"""The ONE tool this agent may use. Read-only. (guide §9)

The docstring below is model-facing: ADK sends it to the model as the tool description.
"""
from .policy import POLICIES


def lookup_triage_policy(category: str) -> dict:
    """Return the approved triage policy for one ticket category.

    Args:
        category: One of access, billing, bug, security, or general.
    """
    policy = POLICIES.get(category.lower())
    if not policy:
        return {"ok": False, "error": "unknown_category"}
    return {"ok": True, "category": category.lower(), **policy}
