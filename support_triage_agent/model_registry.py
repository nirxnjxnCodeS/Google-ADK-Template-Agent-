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
# Free tier: 3.6 Flash allows 5 requests/MINUTE and Google asks to retry after ~57 s, so the
# backoff must be able to reach ~60 s (was max 20 s -> all 3 attempts landed inside the same minute).
RETRY = types.HttpRetryOptions(attempts=4, initial_delay=5.0, max_delay=65.0, http_status_codes=[429, 503])


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
    # timeout (s): a stuck local call becomes a classified error instead of an endless wait.
    return LiteLlm(model=f"ollama_chat/{name}", api_base=api_base,
                   timeout=int(os.environ.get("OLLAMA_TIMEOUT_S", "180")))


def glados_model():
    """Office LAN Ollama server (shared). Address/model come from .env -- never hard-coded."""
    from google.adk.models.lite_llm import LiteLlm

    api_base = os.environ.get("GLADOS_API_BASE")
    if not api_base:
        raise RuntimeError("GLADOS_API_BASE is not set (see .env.example); glados profiles need it")
    name = os.environ.get("GLADOS_MODEL", "nemotron-3.5-lightning:30b-a3b")
    extra = {}
    # Experiment switch. Unset = model default (Nemotron thinks). LiteLLM maps reasoning_effort to
    # Ollama's `think`: low/medium/high -> think on; anything else (e.g. "none") -> think OFF.
    effort = os.environ.get("GLADOS_REASONING_EFFORT")
    if effort:
        extra["reasoning_effort"] = effort
    return LiteLlm(model=f"ollama_chat/{name}", api_base=api_base,
                   timeout=int(os.environ.get("GLADOS_TIMEOUT_S", "180")), **extra)


def describe(profile: str) -> dict:
    """Model IDs per profile, for the experiment log."""
    fast = os.environ.get("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")
    strong = os.environ.get("GEMINI_STRONG_MODEL", "gemini-3.8-flash")
    local = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
    glados = os.environ.get("GLADOS_MODEL", "nemotron-3.5-lightning:30b-a3b")
    if os.environ.get("GLADOS_REASONING_EFFORT"):
        glados += f" (reasoning_effort={os.environ['GLADOS_REASONING_EFFORT']})"
    return {
        "gemini_pipeline": {"extractor": fast, "triage": strong},
        "gemini_strong_only": {"triage": strong},
        "local": {"triage": f"ollama_chat/{local}", "formatter": f"ollama_chat/{local}"},
        "glados_only": {"triage": f"glados/{glados}"},
        "local_reviewed": {"triage": f"ollama_chat/{local}", "reviewer": f"glados/{glados}"},
    }[profile]
