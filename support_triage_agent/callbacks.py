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

# v2 (2026-10-27). v1 = guide verbatim; v1 had 4 gaps (tests GAP-1..4).
# Changes: `label = value` always blocks; `label: value` blocks only if the value looks
# like a secret (>=6 chars incl. a digit/symbol) -> fixes GAP-1 false block. Bare key
# formats added (GAP-4). Exfiltration needs a verb aimed at the prompt/instructions, or
# an "ignore previous instructions" phrase (GAP-2, GAP-3). Validated on a HOLDOUT set
# in tests/test_guardrails.py that is not in the eval file.
GUARDRAIL_VERSION = "v2"
_LABEL = r"\b(api[_ -]?key|access[_ -]?token|secret|password|passwd)"
SECRET_PATTERNS = [
    re.compile(rf"(?i){_LABEL}\s*=\s*\S+"),
    re.compile(rf"(?i){_LABEL}\s*:\s*(?=\S*[\d!@#$%^&*_\-])\S{{6,}}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),   # Google API key
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),        # AWS access key id
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),    # sk- style API keys
]
EXFIL_PATTERNS = [
    re.compile(r"(?i)\bignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|messages)"),
    re.compile(r"(?i)\b(show|reveal|print|display|repeat|output|tell me|give me|what (is|are))\b"
               r"[^.?!\n]{0,40}\b(system prompt|your (instructions|prompt|rules))"),
]

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
    if any(p.search(text) for p in EXFIL_PATTERNS):
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


def store_ticket_text(callback_context: CallbackContext) -> Optional[types.Content]:
    """Runs AFTER the entry guardrail (only for allowed tickets): keeps the original ticket in
    state so a later tool-less reviewer step can see it (it runs with include_contents='none')."""
    callback_context.state["ticket_text"] = _content_text(callback_context.user_content)
    return None


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


def enforce_human_review(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> Optional[LlmResponse]:
    """Hard rule in CODE (guide §21: "enforce in code"): security or P1 => human_review=true.

    Runs after the FINAL-answer model call. Only touches a reply that is already a valid JSON
    object; anything else is left alone (the schema check will fail it honestly).
    What the model decided by itself is kept in state so the eval can still measure it.
    """
    if not llm_response.content or not llm_response.content.parts:
        return None
    parts = llm_response.content.parts
    text_idx = [i for i, p in enumerate(parts) if getattr(p, "text", None) and not getattr(p, "thought", False)]
    if len(text_idx) != 1 or any(getattr(p, "function_call", None) for p in parts):
        return None
    try:
        data = json.loads(parts[text_idx[0]].text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    must_review = data.get("category") == "security" or data.get("priority") == "P1"
    if not must_review or data.get("human_review") is True:
        return None
    callback_context.state["hr_enforced"] = True
    callback_context.state["model_human_review_raw"] = data.get("human_review")
    data["human_review"] = True
    new_parts = list(parts)
    new_parts[text_idx[0]] = types.Part(text=json.dumps(data))
    return LlmResponse(content=types.Content(role="model", parts=new_parts))


def before_tool_guardrail(tool, args: dict, tool_context: ToolContext) -> Optional[dict]:
    """Validate tool choice and arguments before any tool executes."""
    name = getattr(tool, "name", "")
    if name not in ALLOWED_TOOLS:
        return {"ok": False, "error": "tool_not_allowed"}
    category = str(args.get("category", "")).lower()
    if category not in CATEGORIES:
        return {"ok": False, "error": "invalid_category"}
    return None
