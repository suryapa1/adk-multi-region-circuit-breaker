# ADK Multi-Region Gemini with PyBreaker Circuit Breaker

A simple Google ADK agent demonstrating multi-region LLM failover with per-region circuit breakers using `pybreaker`. Handles Gemini API throttling (429), service unavailability (503), and regional outages gracefully.

## Problem

Gemini API quotas are **per-project, per-region**. When one region is throttled, you waste time retrying against a dead endpoint. This pattern:

1. Routes LLM calls through multiple Gemini regions
2. Uses **pybreaker** circuit breakers per-region to skip throttled regions instantly
3. Automatically probes recovered regions after a cooling period

## Circuit Breaker Flow

```
CLOSED (normal) ──5 consecutive 429s──→ OPEN (cooling, skip all calls)
                                            │
                                        60s elapsed
                                            │
                                            ▼
                                      HALF_OPEN (probe)
                                        │         │
                                   success ×2    failure
                                        │         │
                                        ▼         ▼
                                     CLOSED      OPEN (another 60s)
```

## Setup

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

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `GOOGLE_CLOUD_PROJECT` | (required) | GCP project ID |
| `LLM_PRIMARY_REGION` | `us-central1` | Primary Gemini region |
| `LLM_FALLBACK_REGIONS` | `us-east4,us-west1` | Comma-separated fallback regions |
| `LLM_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `CIRCUIT_FAIL_MAX` | `5` | Failures before circuit opens |
| `CIRCUIT_RECOVERY_TIMEOUT` | `60` | Seconds before half-open probe |
| `CIRCUIT_SUCCESS_THRESHOLD` | `2` | Successful probes to close circuit |

## Project Structure

```
multi_region_agent/
├── __init__.py              # ADK agent entry point
├── agent.py                 # Agent definition using MultiRegionGemini
├── multi_region_gemini.py   # Multi-region model with pybreaker
└── tools.py                 # Sample tools for the agent
```

## How It Works

```python
# agent.py — just pass MultiRegionGemini() as the model
from .multi_region_gemini import MultiRegionGemini

root_agent = Agent(
    name="multi_region_agent",
    model=MultiRegionGemini(),      # ← handles failover + circuit breaking
    instruction="You are a helpful assistant.",
    tools=[get_weather, get_time],
)
```

The `MultiRegionGemini` class extends ADK's `Gemini` model. When a region returns 429/503:
1. `pybreaker` records the failure
2. After `CIRCUIT_FAIL_MAX` consecutive failures, the circuit **opens**
3. While open, that region is **skipped instantly** (zero latency waste)
4. After `CIRCUIT_RECOVERY_TIMEOUT` seconds, one probe request is allowed
5. If the probe succeeds, the circuit **closes** and the region is back

## Example Logs

```
LLM #1 | Trying us-central1 [circuit: closed]
LLM #1 | ✅ us-central1 succeeded

LLM #5 | Trying us-central1 [circuit: closed]
CIRCUIT [gemini-us-central1] failure 5/5
CIRCUIT [gemini-us-central1] state: closed → open
LLM #5 | ❌ us-central1 429 → trying next
LLM #5 | Trying us-east4 [circuit: closed]
LLM #5 | ✅ us-east4 succeeded

LLM #8 | SKIP us-central1 — circuit OPEN (probe in 42s)
LLM #8 | Trying us-east4 [circuit: closed]
LLM #8 | ✅ us-east4 succeeded

LLM #15 | us-central1 circuit OPEN → HALF_OPEN (60s elapsed)
LLM #15 | Trying us-central1 [circuit: half-open] — probe
CIRCUIT [gemini-us-central1] state: half_open → closed
LLM #15 | ✅ us-central1 succeeded (back to normal)
```

## License

Apache 2.0
