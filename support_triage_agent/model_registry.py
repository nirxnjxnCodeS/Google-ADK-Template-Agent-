"""Model roles (guide §7D, §11). One place maps a ROLE to a concrete model.

Fixes guide gaps: reads model IDs from env (.env) instead of hard-coding, defines
OLLAMA_API_BASE, and builds models lazily so Gemini-only runs never touch LiteLLM.

Reliability policy for hosted calls (Stage 3/4 finding: 3.8 Flash returns 503 /
queues under high demand):
  * per-request timeout so nothing hangs indefinitely
  * bounded retry ONLY for 429 (rate limit) and 503 (overloaded)
  * the runner counts retries; infra failures are never scored as model errors
"""
import os

from google.adk.models import Gemini
from google.genai import types

GEMINI_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "90000"))
RETRY = types.HttpRetryOptions(attempts=3, initial_delay=2.0, max_delay=20.0, http_status_codes=[429, 503])


def _gemini(model_id: str) -> Gemini:
    return Gemini(
        model=model_id,
        retry_options=RETRY,
        client_kwargs={"http_options": types.HttpOptions(timeout=GEMINI_TIMEOUT_MS)},
    )


def fast_model() -> Gemini:
    return _gemini(os.environ.get("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite"))


def strong_model() -> Gemini:
    return _gemini(os.environ.get("GEMINI_STRONG_MODEL", "gemini-3.8-flash"))


def local_model():
    # Imported here so hosted profiles never load LiteLLM.
    from google.adk.models.lite_llm import LiteLlm

    api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    os.environ.setdefault("OLLAMA_API_BASE", api_base)  # ADK docs: LiteLLM reads the env var too
    name = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    # ollama_chat (not ollama): plain `ollama` can loop on tool calls (ADK docs warning).
    return LiteLlm(model=f"ollama_chat/{name}", api_base=api_base)


def describe(profile: str) -> dict:
    """Model IDs per profile, for the experiment log."""
    fast = os.environ.get("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")
    strong = os.environ.get("GEMINI_STRONG_MODEL", "gemini-3.8-flash")
    local = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    return {
        "gemini_pipeline": {"extractor": fast, "triage": strong},
        "gemini_strong_only": {"triage": strong},
        "local": {"triage": f"ollama_chat/{local}"},
    }[profile]
