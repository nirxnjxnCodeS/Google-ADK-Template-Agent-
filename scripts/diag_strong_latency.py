"""Diagnostic only (not a gate): why is the STRONG model slow?

Same prompt, two thinking settings, 90 s timeout each, no retries (SDK default).
  A: default thinking (gemini-3.8-flash defaults to MEDIUM)
  B: thinking_level=LOW
If B is fast and A slow  -> latency is thinking.
If both slow / time out  -> latency is server-side queueing/capacity, not our config.
"""
import os
import time

from dotenv import load_dotenv

load_dotenv()

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

MODEL = os.environ.get("GEMINI_STRONG_MODEL", "gemini-3.8-flash")
TIMEOUT_MS = 90_000
client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))

RUNS = [
    ("A default", types.GenerateContentConfig(temperature=0)),
    ("B low", types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
    )),
]

for label, cfg in RUNS:
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=MODEL, contents="Reply with exactly: STRONG_SDK_OK", config=cfg
        )
        dt = time.perf_counter() - t0
        u = resp.usage_metadata
        print(f"{label:<10} {dt:7.2f}s  reply={(resp.text or '').strip()[:30]!r}  "
              f"thoughts={getattr(u, 'thoughts_token_count', None)} output={u.candidates_token_count}")
    except Exception as exc:
        dt = time.perf_counter() - t0
        print(f"{label:<10} {dt:7.2f}s  FAIL {type(exc).__name__}: {str(exc)[:200]}")
