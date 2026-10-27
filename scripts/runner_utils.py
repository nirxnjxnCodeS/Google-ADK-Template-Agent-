"""Direct ADK execution, no HTTP server (guide §14). Every preflight/eval goes through run_one().

Captures: final text + author, tool calls/args, stage authors, latency, SDK retries,
infra errors (503/timeout...) and whether the entry guardrail fired.
Infra errors are reported separately so they are never scored as model mistakes.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import ValidationError

from support_triage_agent.agent import build_root_agent
from support_triage_agent.schemas import TriageOutput

APP_NAME = "support_triage_agent"
USER_ID = "eval"
# Fix #3 (all profiles): cap model calls per ticket. A normal run needs <=5
# (extractor/formatter + tool call + answer). Local run2 looped 36 tool calls / 631 s.
MAX_LLM_CALLS = 10


class _RetryCounter(logging.Handler):
    """Counts google-genai retry sleeps (tenacity logs 'Retrying ...' at INFO)."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.count = 0

    def emit(self, record):
        if "retrying" in record.getMessage().lower():
            self.count += 1


async def run_one(prompt: str, profile: str = "gemini_pipeline") -> dict:
    counter = _RetryCounter()
    sdk_logger = logging.getLogger("google_genai._api_client")
    sdk_logger.setLevel(logging.INFO)
    sdk_logger.addHandler(counter)

    session_service = InMemorySessionService()
    runner = Runner(agent=build_root_agent(profile), app_name=APP_NAME, session_service=session_service)
    session_id = str(uuid.uuid4())
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    result = {
        "profile": profile, "text": "", "final_author": None, "authors": [],
        "tool_calls": [], "tool_responses": [], "latency_s": None, "retries": 0,
        "infra_error": None, "event_error": None, "guardrail_blocked": None,
        "stage_texts": {},  # final text per stage -> lets us audit whether a formatter changed a decision
        "agent_error": None,  # agent-caused abort (e.g. loop limit) -> scored as a FAILURE, not infra
    }
    t0 = time.perf_counter()
    try:
        msg = types.Content(role="user", parts=[types.Part(text=prompt)])
        run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS)
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=msg,
                                            run_config=run_config):
            if event.author and (not result["authors"] or result["authors"][-1] != event.author):
                result["authors"].append(event.author)
            for fc in event.get_function_calls() or []:
                result["tool_calls"].append({"name": fc.name, "args": dict(fc.args or {})})
            for fr in event.get_function_responses() or []:
                result["tool_responses"].append({"name": fr.name, "response": fr.response})
            if getattr(event, "error_code", None):
                result["event_error"] = f"{event.error_code}: {getattr(event, 'error_message', '')}"[:300]
            if event.is_final_response() and event.content and event.content.parts:
                text = "".join(p.text for p in event.content.parts if p.text and not getattr(p, "thought", False))
                if text.strip():
                    result["text"] = text.strip()
                    result["final_author"] = event.author
                    result["stage_texts"][event.author] = text.strip()[:600]
    except Exception as exc:
        if "LlmCallsLimitExceeded" in type(exc).__name__:
            result["agent_error"] = f"loop_limit: >{MAX_LLM_CALLS} model calls"
            result["text"] = ""  # no valid final answer -> fails every field
        else:  # 503 after retries, timeouts, connection refused, ...
            result["infra_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        result["latency_s"] = round(time.perf_counter() - t0, 2)
        result["retries"] = counter.count
        sdk_logger.removeHandler(counter)

    session = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    if session is not None:
        result["guardrail_blocked"] = session.state.get("guardrail_blocked")
    return result


def run_sync(prompt: str, profile: str = "gemini_pipeline") -> dict:
    return asyncio.run(run_one(prompt, profile))


def parse_output(text: str) -> tuple[Optional[TriageOutput], Optional[str]]:
    """STRICT contract: the final text must be exactly one JSON object matching TriageOutput.
    Code fences or extra prose count as failures (guide §12)."""
    try:
        return TriageOutput.model_validate(json.loads(text)), None
    except json.JSONDecodeError as exc:
        hint = " (wrapped in code fences)" if text.lstrip().startswith("```") else ""
        return None, f"invalid_json{hint}: {exc}"
    except ValidationError as exc:
        return None, f"schema: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}"
