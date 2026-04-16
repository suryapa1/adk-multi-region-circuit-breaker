# ADK Multi-Region Gemini with PyBreaker Circuit Breaker

A reusable pattern that combines **multi-region Gemini routing** with **per-region circuit breakers** (`pybreaker`) to make Google ADK agents resilient to `429 RESOURCE_EXHAUSTED` and `503 SERVICE_UNAVAILABLE` errors. Drops into any existing ADK agent as a one-line model swap.

> **TL;DR** — Gemini quotas are per-region. Wrap each region in a circuit breaker, route around the hot one for a cooling window, auto-recover when it's healthy. Code below; one-line change in your agent.

---

## 1. The Problem

### 1.1 Quota Model

Vertex AI Gemini quotas are scoped per **(project, region, model)** tuple. The 60 RPM quota in `us-central1` is independent of the 60 RPM available in `us-east4`. They are independent buckets.

When traffic concentrates on the primary region:

1. The first wave of requests succeeds.
2. The minute-window fills and the next request returns `429 RESOURCE_EXHAUSTED`.
3. Naive retry-with-backoff burns 3–8 seconds per request before giving up.
4. p99 latency degrades cliff-style; SLOs are missed.

### 1.2 Why Plain Retry Fails

A retry loop has no **shared knowledge** that a region is currently exhausted. Each request rediscovers the failure independently. With N concurrent requests, you pay the discovery cost N times per minute — a textbook **thundering-herd** amplification.

### 1.3 Why Multi-Region Alone Isn't Enough

Failing over to a secondary region on each `429` helps that one request, but the next request still tries the primary first and pays the full timeout cost again. The system needs **memory** of which regions are currently bad.

---

## 2. The Solution: Region-Wide Circuit Breakers

A **circuit breaker** is a small in-memory state machine, originally from Michael Nygard's *Release It!*, that tracks the health of a downstream dependency and short-circuits calls when it is known to be failing. Applied per-region, it gives us a coordinated throttle across the whole request stream within a worker process.

### 2.1 State Machine

```
                ┌─────────┐
                │ CLOSED  │  Normal operation. Calls execute.
                │         │  Failures are incrementally counted.
                └────┬────┘
                     │  fail_max consecutive failures
                     ▼
                ┌─────────┐
                │  OPEN   │  Calls SKIPPED instantly (~0 ms).
                │         │  No API request issued.
                └────┬────┘
                     │  recovery_timeout elapses
                     ▼
                ┌──────────┐
                │HALF-OPEN │  Single probe call permitted.
                └────┬─────┘
            success │ │ failure
                    ▼ ▼
        success_threshold  → CLOSED   (region restored)
        any failure        → OPEN     (extend cooling)
```

The crucial property: when a breaker is **OPEN**, the call does not happen. There is no API request, no timeout, no latency cost. The system has *learned* that the region is bad and acts on it.

### 2.2 Why `pybreaker`

- Mature, thread-safe, ~200 KB pure Python.
- First-class listener API for observability hooks.
- Configurable failure-counting, exclusion lists, and recovery timing.
- No external dependencies; works in any worker (Cloud Run, GKE, Lambda, Cloud Functions).

---

## 3. Implementation

### 3.1 Component Overview

| Component | Responsibility |
|---|---|
| `MultiRegionGemini` | Subclass of ADK's `Gemini` model. Drives the failover loop. |
| Per-region `Gemini` clients | One ADK `Gemini` instance per region, each with its own `genai.Client(location=region)`. |
| Per-region `pybreaker.CircuitBreaker` | One breaker per region, registered at module scope so state persists across requests. |
| `_CircuitListener` | `pybreaker` listener that logs every state transition, failure, and success. |
| `_STATS` registry | Module-level counters for total calls, retries, exhaustions, and per-region metrics. |

### 3.2 Request Flow

For every call to `generate_content_async`:

1. Iterate regions in configured priority order.
2. **Inspect the breaker first.**
   - `OPEN` → record skip, continue to next region (zero API cost).
   - `HALF_OPEN` → log probe, proceed cautiously.
   - `CLOSED` → proceed normally.
3. Invoke the regional Gemini client.
4. On success → `breaker.call_succeeded()`, yield response, return.
5. On retryable failure (`429`, `503`, `ResourceExhausted`, `ServiceUnavailable`, status `404/500`) → `breaker.call_failed()`, fall through to next region.
6. On non-retryable failure (auth, safety filter, bad prompt) → raise immediately. No failover, no breaker mutation.
7. If all regions exhausted → raise the last captured exception.

### 3.3 Key Design Choices

- **Breakers live at module scope.** State persists across requests within a worker. Each worker discovers an OPEN region once, not once per request.
- **One breaker per region.** Opening `us-central1` does not affect `us-east4`. They are independent failure domains.
- **Non-retryable errors raise immediately.** Auth failures, safety blocks, and bad prompts do not trip breakers.
- **Listener-based logging.** State changes, failure counts, recoveries are all visible in Cloud Logging without extra plumbing.

---

## 4. Drop-In Usage

The whole point is that this should be a one-line swap in any existing ADK agent.

**Before:**

```python
from google.adk.agents import Agent

root_agent = Agent(
    name="my_agent",
    model="gemini-2.5-flash",
    tools=[...],
)
```

**After:**

```python
from google.adk.agents import Agent
from .multi_region_gemini import MultiRegionGemini

root_agent = Agent(
    name="my_agent",
    model=MultiRegionGemini(),
    tools=[...],
)
```

Tools, instructions, sub-agents, callbacks — everything else works exactly the same.

---

## 5. Setup

```bash
# Clone
git clone https://github.com/suryapa1/adk-multi-region-circuit-breaker.git
cd adk-multi-region-circuit-breaker

# Install
pip install -r requirements.txt

# Configure
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="true"

# Run with ADK dev UI
adk web
```

---

## 6. Configuration

All knobs are environment variables — no code change needed to retune.

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | (required) | GCP project ID |
| `LLM_PRIMARY_REGION` | `us-central1` | First region attempted |
| `LLM_FALLBACK_REGIONS` | `us-east4,us-west1` | Comma-separated failover order |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `CIRCUIT_FAIL_MAX` | `5` | Consecutive failures before OPEN |
| `CIRCUIT_RECOVERY_TIMEOUT` | `60` | Seconds OPEN before HALF-OPEN probe |
| `CIRCUIT_SUCCESS_THRESHOLD` | `2` | Probes needed to fully close |

---

## 7. Observability

### 7.1 Structured Logs

Every state transition and failure is logged via `_CircuitListener`:

```
LLM #1   | Trying us-central1 [circuit:closed]
LLM #1   | us-central1 succeeded

LLM #5   | Trying us-central1 [circuit:closed]
CIRCUIT [gemini-us-central1] failure 5/5: 429 Resource exhausted
CIRCUIT [gemini-us-central1] closed → open
LLM #5   | us-central1 failed → next region
LLM #5   | Trying us-east4 [circuit:closed]
LLM #5   | us-east4 succeeded

LLM #8   | SKIP us-central1 — circuit OPEN
LLM #8   | Trying us-east4 [circuit:closed]
LLM #8   | us-east4 succeeded

LLM #15  | us-central1 circuit HALF_OPEN — sending probe
CIRCUIT [gemini-us-central1] half-open → closed
LLM #15  | us-central1 succeeded (recovered)
```

Notice request `#8`: `us-central1` is OPEN, so it is skipped without an API call. That is the saved 7 seconds, multiplied by every concurrent request, every minute the region is throttled.

### 7.2 Health Endpoints

Two static methods provide machine-readable telemetry suitable for `/health` and `/stats` endpoints, scrape jobs, or Cloud Monitoring custom metrics:

```python
MultiRegionGemini.get_circuit_health()
# {
#   "us-central1": {"state": "open",   "fail_count": 5, "fail_max": 5},
#   "us-east4":    {"state": "closed", "fail_count": 0, "fail_max": 5},
#   "us-west1":    {"state": "closed", "fail_count": 0, "fail_max": 5},
# }

MultiRegionGemini.get_stats()
# {
#   "total_calls": 142,
#   "total_retries": 8,
#   "total_exhausted": 0,
#   "total_circuit_skips": 19,
#   "per_region": { ... }
# }
```

### 7.3 Recommended Alerts

- `total_exhausted > 0` over 5 min → page on-call (every region down).
- Any region in `OPEN` state for > 5 min → ticket (sustained regional issue).
- `total_circuit_skips / total_calls > 0.10` → review quota allocation.

---

## 8. Operational Behavior

### 8.1 Failure Modes Handled

| Scenario | Behavior |
|---|---|
| Single 429 burst on primary | First N requests retried via fallback; circuit opens; subsequent requests skip primary instantly. |
| Sustained primary outage | Primary stays OPEN; traffic flows to secondaries; periodic probes detect recovery. |
| Two regions down simultaneously | Third region carries traffic; alerts fire on `OPEN` state durations. |
| All regions exhausted | Raises last exception with full context; counter increments for alerting. |
| Auth / safety / bad-prompt error | Raised immediately on first region; no failover, no false breaker trips. |

### 8.2 What This Pattern Does *Not* Solve

- **Cross-process coordination.** Breakers live in the worker's memory. With many workers, each rediscovers the OPEN state independently for the first batch of failures. For shared state, use a Redis-backed implementation (`pybreaker.CircuitRedisStorage` or build something on Memorystore).
- **Quota allocation.** This pattern routes around bad regions; it does not increase your total quota. Right-size your quotas as a separate workstream.
- **Cost optimization.** All regions are tried until one succeeds. If certain regions are pricier, weight them in the fallback order.

---

## 9. Generalization

The reusable abstraction is *"one breaker per failure-isolation unit, plus an ordered fallback list."* The Gemini case is one instantiation. The same shape works for:

- **Other LLM providers** (OpenAI, Anthropic, Cohere) — one breaker per deployment region or per API key.
- **Embedding and image-gen services** — same shape, different client.
- **Internal microservices** with regional pools — wrap each upstream pool.
- **Any third-party API** that throttles per credential.

If you are consuming a sharded API and getting bitten by throttles, this pattern likely fits.

---

## 10. Project Structure

```
multi_region_agent/
├── __init__.py              # ADK agent entry point
├── agent.py                 # Agent definition using MultiRegionGemini
├── multi_region_gemini.py   # Multi-region model with pybreaker
└── tools.py                 # Sample tools for the agent
```

---

## 11. References

- `pybreaker`: https://github.com/danielfm/pybreaker
- Vertex AI Gemini quotas: https://cloud.google.com/vertex-ai/generative-ai/docs/quotas
- Original circuit breaker pattern: Michael Nygard, *Release It!* (2nd ed., Ch. 5)

---

## License

Apache 2.0
