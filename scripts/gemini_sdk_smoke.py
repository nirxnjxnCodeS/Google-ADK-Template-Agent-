"""Direct Google Gen AI SDK smoke test -- NO ADK involved (guide §7C).

Proves, one layer at a time:
  1. google-genai imports (prints version)
  2. an API key is present (never printed)
  3. the FAST model ID works  -> expects FAST_SDK_OK
  4. the STRONG model ID works -> expects STRONG_SDK_OK
Each failure is classified: auth / model_availability / rate_limit / network / sdk_usage.
"""
import os
import sys
import time
from importlib.metadata import version

from dotenv import load_dotenv

load_dotenv()  # reads .env from the current directory (the project root)

from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

print(f"google-genai {version('google-genai')}")

key = os.environ.get("GEMINI_API_KEY", "")
if os.environ.get("GOOGLE_API_KEY"):
    print("WARNING: GOOGLE_API_KEY is also set and takes precedence over GEMINI_API_KEY. Unset it.")
if not key or key == "replace_with_your_real_key":
    print("FAIL [auth]: GEMINI_API_KEY missing or still the placeholder in .env")
    sys.exit(1)
print(f"API key detected (GEMINI_API_KEY, length {len(key)}) -- value not printed")


def redact(text: str) -> str:
    """Never let the key leak through an error message."""
    return text.replace(key, "***REDACTED***")


def classify(exc: Exception) -> str:
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if code in (401, 403):
            return "auth"
        if code == 400 and "api key" in str(exc).lower():
            return "auth"
        if code == 404:
            return "model_availability"
        if code == 429:
            return "rate_limit"
        if code and code >= 500:
            return "network/server"
        return "sdk_usage"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout (server queue/capacity)"
    if any(w in name for w in ("connect", "network")):
        return "network"
    return "sdk_usage"


# SDK reads GEMINI_API_KEY from the environment itself. 120 s timeout so a server-side
# queue shows up as a classified failure instead of an indefinite hang. No retries (SDK default).
client = genai.Client(http_options=types.HttpOptions(timeout=120_000))

CHECKS = [
    ("FAST", os.environ.get("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite"), "FAST_SDK_OK"),
    ("STRONG", os.environ.get("GEMINI_STRONG_MODEL", "gemini-3.8-flash"), "STRONG_SDK_OK"),
]

failed = False
for role, model, token in CHECKS:
    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=f"Reply with exactly: {token}",
            config=types.GenerateContentConfig(temperature=0),
        )
        latency = time.perf_counter() - t0
        text = (resp.text or "").strip()
        exact = text == token
        status = "PASS" if token in text else "FAIL [model_output]"
        failed |= token not in text
        print(f"{role:<6} {model:<24} {status}  exact_match={exact}  latency={latency:.2f}s  reply={text[:60]!r}")
        u = resp.usage_metadata
        if u is not None:
            print(f"       tokens: prompt={u.prompt_token_count} thoughts={getattr(u, 'thoughts_token_count', None)} "
                  f"output={u.candidates_token_count}  model_version={getattr(resp, 'model_version', None)}")
    except Exception as exc:
        latency = time.perf_counter() - t0
        failed = True
        print(f"{role:<6} {model:<24} FAIL [{classify(exc)}]  latency={latency:.2f}s")
        print(f"       {type(exc).__name__}: {redact(str(exc))[:300]}")

print()
print("SDK_SMOKE_FAILED" if failed else "SDK_SMOKE_OK")
sys.exit(1 if failed else 0)
