"""Learning tool: show EXACTLY what the agent sends to the model and what comes back.

Uses an ADK *plugin* (a Runner-level callback) so the agent code is unchanged.
For every model call it prints: which agent is calling, the system instruction,
the tool descriptions, the full conversation (text / tool calls / tool results),
then the model's reply and token usage. Also saved to results/peek_<profile>.txt.

    python scripts/peek_request.py --profile local "One user cannot log in after changing phones."
"""
import argparse
import asyncio
import json
import uuid
from pathlib import Path

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from support_triage_agent.agent import PROFILES, build_root_agent

LINES: list[str] = []


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


def show_part(p) -> str:
    if getattr(p, "function_call", None):
        return f"TOOL CALL   -> {p.function_call.name}({json.dumps(dict(p.function_call.args or {}))})"
    if getattr(p, "function_response", None):
        return f"TOOL RESULT <- {p.function_response.name}: {json.dumps(p.function_response.response)[:400]}"
    if getattr(p, "thought", False):
        return f"(thinking) {(p.text or '')[:300]}"
    return f"TEXT: {p.text}"


class PeekPlugin(BasePlugin):
    def __init__(self):
        super().__init__(name="peek")
        self.n = 0

    async def before_model_callback(self, *, callback_context, llm_request):
        self.n += 1
        out("=" * 90)
        out(f"MODEL CALL #{self.n}   agent={callback_context.agent_name}   model={llm_request.model}")
        out("-" * 90)
        out("[1] SYSTEM INSTRUCTION (sent every call):")
        out(str(llm_request.config.system_instruction or "(none)"))
        decls = [d for t in (llm_request.config.tools or []) for d in (getattr(t, "function_declarations", None) or [])]
        out("-" * 90)
        out(f"[2] TOOLS OFFERED ({len(decls)}):")
        for d in decls:
            # ADK 2.x usually fills parameters_json_schema (plain JSON Schema), not parameters.
            params = (getattr(d, "parameters_json_schema", None)
                      or (d.parameters.model_dump(exclude_none=True) if d.parameters else {}))
            out(f"  - {d.name}: {(d.description or '').strip()}")
            out(f"    parameters: {json.dumps(params, default=str)}")
        if llm_request.config.response_schema or getattr(llm_request.config, "response_json_schema", None):
            out("  (plus a RESPONSE SCHEMA: output is constrained to that JSON shape)")
        out("-" * 90)
        out(f"[3] CONVERSATION SO FAR ({len(llm_request.contents)} message(s)):")
        for c in llm_request.contents:
            for p in c.parts or []:
                out(f"  {c.role:>5}: {show_part(p)}")
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        out("-" * 90)
        out("[4] MODEL REPLY:")
        for p in (llm_response.content.parts if llm_response.content else []) or []:
            out(f"  model: {show_part(p)}")
        u = llm_response.usage_metadata
        if u:
            out(f"  tokens: prompt={u.prompt_token_count} output={u.candidates_token_count} "
                f"thinking={getattr(u, 'thoughts_token_count', None)}")
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        out("=" * 90)
        out(f"ADK RUNS THE TOOL (not the model): {tool.name}({json.dumps(tool_args)})")
        return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="local", choices=list(PROFILES))
    ap.add_argument("ticket")
    args = ap.parse_args()

    sessions = InMemorySessionService()
    runner = Runner(agent=build_root_agent(args.profile), app_name="peek",
                    session_service=sessions, plugins=[PeekPlugin()])
    sid = str(uuid.uuid4())
    await sessions.create_session(app_name="peek", user_id="me", session_id=sid)
    out(f"TICKET: {args.ticket}\nPROFILE: {args.profile}")
    msg = types.Content(role="user", parts=[types.Part(text=args.ticket)])
    async for event in runner.run_async(user_id="me", session_id=sid, new_message=msg):
        if event.is_final_response() and event.content and event.content.parts:
            final = "".join(p.text or "" for p in event.content.parts if not getattr(p, "thought", False))
            if final.strip():
                out("=" * 90)
                out(f"FINAL ANSWER from {event.author}: {final.strip()}")

    path = Path("results") / f"peek_{args.profile}.txt"
    path.write_text("\n".join(LINES))
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
