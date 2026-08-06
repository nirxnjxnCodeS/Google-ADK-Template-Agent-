"""Deterministic guardrails (guide §10). Executable controls, not instructions.

Layering (deviation from guide, see note):
  1. before_agent_guardrail  -> on the ROOT agent. Screens the user's ticket once,
     before ANY model or tool runs. Returning Content ends the whole invocation.
     This replaces the guide's undefined `before_extractor_guardrail` and avoids its
     problem (a blocked reply cannot satisfy the extractor's TicketFacts schema, and
     the triage stage would still run afterwards).
  2. before_model_guardrail  -> on the triage LlmAgent. Guide's original backstop.
  3. before_tool_guardrail   -> validates tool name + arguments before execution.
"""
import json
import re
from typing import Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .policy import CATEGORIES

ALLOWED_TOOLS = {"lookup_triage_policy"}

# v1 patterns = guide verbatim. Known gaps are listed in the Stage 5 notes; changes
# must come with a test (positive + negative) first.
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_ -]?key|access[_ -]?token|password)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
EXFIL_VERBS = ["show", "reveal", "print", "ignore previous"]

BLOCKED_OUTPUT = {
    "category": "general",
    "priority": "UNKNOWN",
    "human_review": True,
    "needs_more_info": True,
    "reason": "Input blocked by safety policy.",
    "next_action": "Remove secrets or prompt-exfiltration instructions and resubmit the support issue.",
}
BLOCKED_JSON = json.dumps(BLOCKED_OUTPUT)


def screen_text(text: str) -> Optional[str]:
    """Return a block reason ('secret' / 'prompt_exfiltration') or None if allowed."""
    if any(p.search(text) for p in SECRET_PATTERNS):
        return "secret"
    lower = text.lower()
    if "system prompt" in lower and any(v in lower for v in EXFIL_VERBS):
        return "prompt_exfiltration"
    return None


def _content_text(content: Optional[types.Content]) -> str:
    if not content:
        return ""
    return "\n".join(p.text for p in (content.parts or []) if getattr(p, "text", None))


def _request_text(llm_request: LlmRequest) -> str:
    return "\n".join(_content_text(c) for c in (llm_request.contents or []))


def before_agent_guardrail(callback_context: CallbackContext) -> Optional[types.Content]:
    """Pipeline entry gate: block before any model call or tool call happens."""
    reason = screen_text(_content_text(callback_context.user_content))
    if reason is None:
        return None
    callback_context.state["guardrail_blocked"] = reason  # visible in eval traces
    return types.Content(role="model", parts=[types.Part(text=BLOCKED_JSON)])


def before_model_guardrail(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """Backstop: block obvious secret submission and prompt-exfiltration attempts."""
    reason = screen_text(_request_text(llm_request))
    if reason is None:
        return None
    callback_context.state["guardrail_blocked"] = reason
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=BLOCKED_JSON)]))


def before_tool_guardrail(tool, args: dict, tool_context: ToolContext) -> Optional[dict]:
    """Validate tool choice and arguments before any tool executes."""
    name = getattr(tool, "name", "")
    if name not in ALLOWED_TOOLS:
        return {"ok": False, "error": "tool_not_allowed"}
    category = str(args.get("category", "")).lower()
    if category not in CATEGORIES:
        return {"ok": False, "error": "invalid_category"}
    return None
