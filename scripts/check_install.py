"""Installation-layer check: versions + every import the project will need.

No API key, no network, no model call. If this fails, it is an INSTALL problem,
not an agent/prompt problem. Prints IMPORT_CHECK_OK only if everything passes.
"""
import importlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["google-adk", "google-genai", "litellm", "pydantic", "pytest", "python-dotenv"]
BAD_LITELLM = {"1.82.7", "1.82.8"}  # compromised releases, March 2026

# (module, attribute or None) -- the exact symbols our code will import later.
IMPORTS = [
    ("google.genai", "Client"),
    ("google.genai.types", "Content"),
    ("google.adk.agents", "LlmAgent"),
    ("google.adk.agents", "SequentialAgent"),
    ("google.adk.agents.callback_context", "CallbackContext"),
    ("google.adk.models", "LlmRequest"),
    ("google.adk.models", "LlmResponse"),
    ("google.adk.models", "Gemini"),
    ("google.adk.models.lite_llm", "LiteLlm"),
    ("google.adk.tools.tool_context", "ToolContext"),
    ("google.adk.runners", "Runner"),
    ("google.adk.sessions", "InMemorySessionService"),
    ("pydantic", "BaseModel"),
    ("dotenv", "load_dotenv"),
]

failed = False
print(f"python        {platform.python_version()} ({sys.executable})")

for pkg in PACKAGES:
    try:
        v = version(pkg)
        flag = ""
        if pkg == "litellm" and v in BAD_LITELLM:
            flag = "  <-- COMPROMISED RELEASE, DO NOT USE"
            failed = True
        print(f"{pkg:<13} {v}{flag}")
    except PackageNotFoundError:
        print(f"{pkg:<13} NOT INSTALLED")
        failed = True

print()
for module, attr in IMPORTS:
    label = f"{module}.{attr}" if attr else module
    try:
        mod = importlib.import_module(module)
        if attr:
            getattr(mod, attr)
        print(f"ok    {label}")
    except Exception as exc:  # report every failure, don't stop at the first
        print(f"FAIL  {label}: {type(exc).__name__}: {exc}")
        failed = True

print()
if failed:
    print("IMPORT_CHECK_FAILED  (installation layer -- fix this before anything else)")
    sys.exit(1)
print("IMPORT_CHECK_OK")
