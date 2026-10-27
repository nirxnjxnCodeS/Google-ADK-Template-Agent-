"""Hard rule in code: security or P1 => human_review=true (no model needed)."""
import json
from types import SimpleNamespace

from google.adk.models import LlmResponse
from google.genai import types

from support_triage_agent.callbacks import enforce_human_review


def _resp(obj_or_text):
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _answer(**kw):
    base = {"category": "bug", "priority": "P3", "human_review": False, "needs_more_info": False,
            "reason": "r", "next_action": "n"}
    base.update(kw)
    return base


def test_security_without_review_is_forced_true():
    ctx = SimpleNamespace(state={})
    out = enforce_human_review(ctx, _resp(_answer(category="security", priority="P2")))
    assert json.loads(out.content.parts[0].text)["human_review"] is True
    assert ctx.state["hr_enforced"] is True
    assert ctx.state["model_human_review_raw"] is False  # model's own decision preserved


def test_p1_without_review_is_forced_true():
    ctx = SimpleNamespace(state={})
    out = enforce_human_review(ctx, _resp(_answer(priority="P1")))
    assert json.loads(out.content.parts[0].text)["human_review"] is True


def test_correct_answer_untouched():
    ctx = SimpleNamespace(state={})
    assert enforce_human_review(ctx, _resp(_answer(category="security", priority="P1", human_review=True))) is None
    assert "hr_enforced" not in ctx.state


def test_ordinary_ticket_untouched():
    """Code must not add reviews the policy doesn't require (no over-escalation by code)."""
    ctx = SimpleNamespace(state={})
    assert enforce_human_review(ctx, _resp(_answer(category="bug", priority="P3"))) is None


def test_invalid_json_left_for_schema_check():
    """Code never 'repairs' broken output -- the eval must still see it fail."""
    ctx = SimpleNamespace(state={})
    assert enforce_human_review(ctx, _resp('thora {"category":"security"}')) is None


def test_tool_call_turn_untouched():
    ctx = SimpleNamespace(state={})
    call = types.Part(function_call=types.FunctionCall(name="lookup_triage_policy", args={"category": "security"}))
    resp = LlmResponse(content=types.Content(role="model", parts=[call]))
    assert enforce_human_review(ctx, resp) is None
