"""
Multi-Region Gemini LLM with PyBreaker Circuit Breakers.

Extends ADK's Gemini model class to add:
  1. Multi-region failover (N regions via env vars)
  2. Per-region circuit breakers (pybreaker)
  3. Automatic cooling + half-open probing for recovered regions
  4. Health endpoint for monitoring

Usage in agent.py:
    from .multi_region_gemini import MultiRegionGemini
    agent = Agent(model=MultiRegionGemini(), ...)

Env Vars:
    LLM_PRIMARY_REGION=us-central1
    LLM_FALLBACK_REGIONS=us-east4,us-west1
    LLM_MODEL=gemini-2.5-flash
    CIRCUIT_FAIL_MAX=5
    CIRCUIT_RECOVERY_TIMEOUT=60
    CIRCUIT_SUCCESS_THRESHOLD=2
"""

import os
import logging
from typing import AsyncIterator

import pybreaker
from google import genai
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.utils.variant_utils import GoogleLLMVariant
from google.genai.errors import ClientError
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
LLM_PRIMARY_REGION   = os.environ.get("LLM_PRIMARY_REGION", "us-central1")
LLM_FALLBACK_REGIONS = os.environ.get("LLM_FALLBACK_REGIONS", "us-east4,us-west1").split(",")
LLM_MODEL            = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

CIRCUIT_FAIL_MAX          = int(os.environ.get("CIRCUIT_FAIL_MAX", "5"))
CIRCUIT_RECOVERY_TIMEOUT  = int(os.environ.get("CIRCUIT_RECOVERY_TIMEOUT", "60"))
CIRCUIT_SUCCESS_THRESHOLD = int(os.environ.get("CIRCUIT_SUCCESS_THRESHOLD", "2"))

# Errors that trigger failover
_QUOTA_ERRORS = (ResourceExhausted, ServiceUnavailable)
_FAILOVER_STATUS_CODES = {404, 429, 500, 503}


# ---------------------------------------------------------------------------
# PyBreaker Listener — logs state transitions to Cloud Logging / stdout
# ---------------------------------------------------------------------------
class _CircuitListener(pybreaker.CircuitBreakerListener):
    """Logs circuit breaker events for observability."""

    def state_change(self, cb, old_state, new_state):
        logger.warning(
            "CIRCUIT [%s] %s → %s",
            cb.name, old_state.name, new_state.name,
        )

    def failure(self, cb, exc):
        logger.warning(
            "CIRCUIT [%s] failure %d/%d: %s",
            cb.name, cb.fail_counter, cb.fail_max, str(exc)[:100],
        )

    def success(self, cb):
        logger.info("CIRCUIT [%s] success — counter reset", cb.name)


# ---------------------------------------------------------------------------
# Per-region circuit breaker registry (module-level, survives across requests)
# ---------------------------------------------------------------------------
_BREAKERS: dict[str, pybreaker.CircuitBreaker] = {}


def _get_or_create_breaker(region: str) -> pybreaker.CircuitBreaker:
    """Get or create a pybreaker CircuitBreaker for a Gemini region."""
    if region not in _BREAKERS:
        _BREAKERS[region] = pybreaker.CircuitBreaker(
            fail_max=CIRCUIT_FAIL_MAX,
            reset_timeout=CIRCUIT_RECOVERY_TIMEOUT,
            success_threshold=CIRCUIT_SUCCESS_THRESHOLD,
            exclude=[ValueError, TypeError, PermissionError],
            listeners=[_CircuitListener()],
            name=f"gemini-{region}",
        )
        logger.info(
            "Created pybreaker for %s (fail_max=%d, recovery=%ds, success_threshold=%d)",
            region, CIRCUIT_FAIL_MAX, CIRCUIT_RECOVERY_TIMEOUT, CIRCUIT_SUCCESS_THRESHOLD,
        )
    return _BREAKERS[region]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_retryable_error(exc: Exception) -> bool:
    """Returns True if the error should trigger region failover."""
    if isinstance(exc, _QUOTA_ERRORS):
        return True
    if isinstance(exc, ClientError):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in _FAILOVER_STATUS_CODES:
            return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "quota", "resource exhausted", "unavailable"))


def _make_regional_gemini(region: str) -> Gemini:
    """
    Create a Gemini instance wired to a specific Vertex AI region.

    Injects a regional genai.Client into the Gemini model's cached_property
    slot before ADK accesses it, ensuring LLM calls go to the target region.
    """
    regional_client = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=region,
    )

    gemini = Gemini(model=LLM_MODEL)

    # Inject regional client before ADK's lazy init touches it
    gemini.__dict__["api_client"] = regional_client
    gemini.__dict__["_api_backend"] = (
        GoogleLLMVariant.VERTEX_AI
        if regional_client.vertexai
        else GoogleLLMVariant.GEMINI_API
    )

    logger.info("Regional Gemini ready: region=%s model=%s", region, LLM_MODEL)
    return gemini


# ---------------------------------------------------------------------------
# Call stats (module-level, shared across instances)
# ---------------------------------------------------------------------------
_STATS: dict = {
    "total_calls": 0,
    "total_retries": 0,
    "total_exhausted": 0,
    "total_circuit_skips": 0,
    "per_region": {},
}


def _init_region_stats(region: str):
    if region not in _STATS["per_region"]:
        _STATS["per_region"][region] = {
            "attempts": 0, "successes": 0, "failures": 0, "skips": 0,
        }


# ---------------------------------------------------------------------------
# MultiRegionGemini — drop-in replacement for Gemini in LlmAgent
# ---------------------------------------------------------------------------
class MultiRegionGemini(Gemini):
    """
    ADK Gemini model with multi-region failover + pybreaker circuit breakers.

    Pass this as `model=` to any ADK Agent / LlmAgent:

        from multi_region_agent.multi_region_gemini import MultiRegionGemini

        agent = Agent(
            name="my_agent",
            model=MultiRegionGemini(),
            instruction="...",
        )

    Regions are configured via env vars:
        LLM_PRIMARY_REGION=us-central1
        LLM_FALLBACK_REGIONS=us-east4,us-west1

    Each region gets its own pybreaker.CircuitBreaker:
        CLOSED    → normal, failures counted
        OPEN      → region skipped (cooling period)
        HALF_OPEN → one probe allowed, success→CLOSED, fail→OPEN
    """

    def __init__(self):
        super().__init__(model=LLM_MODEL)

        # Build ordered region list
        regions = []
        if LLM_PRIMARY_REGION:
            regions.append(LLM_PRIMARY_REGION)
        regions.extend([r.strip() for r in LLM_FALLBACK_REGIONS if r.strip()])

        if not regions:
            raise ValueError("No regions configured. Set LLM_PRIMARY_REGION or LLM_FALLBACK_REGIONS.")

        # Each entry: (region_name, gemini_instance, circuit_breaker)
        self.__dict__["_regional_models"] = [
            (region, _make_regional_gemini(region), _get_or_create_breaker(region))
            for region in regions
        ]

        for region in regions:
            _init_region_stats(region)

        logger.info(
            "MultiRegionGemini | regions=%s | circuit: fail_max=%d recovery=%ds",
            [r for r in regions], CIRCUIT_FAIL_MAX, CIRCUIT_RECOVERY_TIMEOUT,
        )

    # --- Stats / Health ---

    @staticmethod
    def get_stats() -> dict:
        """Returns call stats. Wire into a /stats endpoint for monitoring."""
        return _STATS

    @staticmethod
    def get_circuit_health() -> dict:
        """
        Returns per-region circuit breaker state.

        Example:
            {
                "us-central1": {"state": "open",   "fail_count": 5, "fail_max": 5},
                "us-east4":    {"state": "closed", "fail_count": 0, "fail_max": 5},
            }
        """
        return {
            region: {
                "state": str(breaker.current_state),
                "fail_count": breaker.fail_counter,
                "fail_max": breaker.fail_max,
            }
            for region, breaker in _BREAKERS.items()
        }

    @staticmethod
    def _log_stats():
        parts = []
        for region, data in _STATS["per_region"].items():
            breaker = _BREAKERS.get(region)
            state = str(breaker.current_state) if breaker else "?"
            parts.append(
                f"{region}:{data['attempts']}att/{data['successes']}ok/"
                f"{data['failures']}fail/{data['skips']}skip/{state}"
            )
        logger.warning(
            "LLM STATS | calls=%d retries=%d exhausted=%d skips=%d | %s",
            _STATS["total_calls"], _STATS["total_retries"],
            _STATS["total_exhausted"], _STATS["total_circuit_skips"],
            " | ".join(parts),
        )

    # --- Core: generate_content_async with failover + circuit breaker ---

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncIterator[LlmResponse]:
        """
        Override ADK's generate_content_async with multi-region failover.

        For each region in order:
          1. Check circuit breaker — if OPEN, skip instantly
          2. If CLOSED or HALF_OPEN, attempt the Gemini call
          3. On success → record success, return response
          4. On retryable error (429/503) → record failure, try next region
          5. On non-retryable error → raise immediately
          6. If all regions exhausted → raise last error
        """
        _STATS["total_calls"] += 1
        call_num = _STATS["total_calls"]
        retries = 0
        last_exc = None

        regional_models = self.__dict__.get("_regional_models", [])

        for region, gemini_instance, breaker in regional_models:
            region_stats = _STATS["per_region"][region]

            # ── Circuit Breaker Gate ──
            if breaker.current_state == "open":
                region_stats["skips"] += 1
                _STATS["total_circuit_skips"] += 1
                logger.warning(
                    "LLM #%d | SKIP %s — circuit OPEN", call_num, region,
                )
                continue

            if breaker.current_state == "half-open":
                logger.warning(
                    "LLM #%d | %s circuit HALF_OPEN — sending probe", call_num, region,
                )

            # ── Attempt LLM Call ──
            region_stats["attempts"] += 1
            logger.info(
                "LLM #%d | Trying %s [circuit:%s] (attempt %d/%d)",
                call_num, region, breaker.current_state,
                retries + 1, len(regional_models),
            )

            try:
                async for response in gemini_instance.generate_content_async(
                    llm_request, stream=stream
                ):
                    yield response

                # Success
                breaker.call_succeeded()
                region_stats["successes"] += 1
                logger.info(
                    "LLM #%d | ✅ %s [circuit:%s] retries=%d",
                    call_num, region, breaker.current_state, retries,
                )
                self._log_stats()
                return

            except Exception as exc:
                if _is_retryable_error(exc):
                    breaker.call_failed()
                    last_exc = exc
                    region_stats["failures"] += 1
                    _STATS["total_retries"] += 1
                    retries += 1
                    logger.warning(
                        "LLM #%d | ❌ %s [circuit:%s]: %s → next region",
                        call_num, region, breaker.current_state, str(exc)[:120],
                    )
                    continue
                else:
                    # Non-retryable (bad prompt, safety, auth) → raise immediately
                    logger.error(
                        "LLM #%d | Non-retryable error on %s: %s", call_num, region, exc,
                    )
                    raise

        # All regions exhausted
        _STATS["total_exhausted"] += 1
        logger.error(
            "LLM #%d | ALL %d regions exhausted. retries=%d. Last: %s",
            call_num, len(regional_models), retries, last_exc,
        )
        self._log_stats()
        raise last_exc
