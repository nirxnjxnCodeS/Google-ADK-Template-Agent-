"""Every profile builds without calling a model; glados profiles fail clearly without config."""
import pytest

from support_triage_agent import agent
from support_triage_agent.agent import PROFILES, REVIEWER_INSTRUCTION, build_root_agent


@pytest.mark.parametrize("profile", PROFILES)
def test_every_profile_builds(profile, monkeypatch):
    monkeypatch.setenv("GLADOS_API_BASE", "http://example.invalid:11434")
    root = build_root_agent(profile)
    assert root.name


def test_glados_profile_needs_config(monkeypatch):
    monkeypatch.delenv("GLADOS_API_BASE", raising=False)
    with pytest.raises(RuntimeError, match="GLADOS_API_BASE"):
        build_root_agent("glados_only")


def test_reviewer_pipeline_guards_before_storing_ticket(monkeypatch):
    monkeypatch.setenv("GLADOS_API_BASE", "http://example.invalid:11434")
    root = build_root_agent("local_reviewed")
    assert [s.name for s in root.sub_agents] == ["support_triage_decision", "triage_reviewer"]
    cbs = root.before_agent_callback
    assert cbs[0] is agent.before_agent_guardrail  # guardrail must run first


def test_reviewer_prompt_has_only_intended_placeholders():
    assert "{ticket_text}" in REVIEWER_INSTRUCTION and "{triage_draft}" in REVIEWER_INSTRUCTION
    stripped = REVIEWER_INSTRUCTION.replace("{ticket_text}", "").replace("{triage_draft}", "")
    assert "{" not in stripped and "}" not in stripped
