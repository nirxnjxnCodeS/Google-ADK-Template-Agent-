import pytest

from support_triage_agent.policy import CATEGORIES
from support_triage_agent.tools import lookup_triage_policy


def test_known_category():
    result = lookup_triage_policy("security")
    assert result["ok"] is True
    assert result["category"] == "security"


def test_unknown_category():
    assert lookup_triage_policy("other")["ok"] is False


@pytest.mark.parametrize("category", sorted(CATEGORIES))
def test_every_category_resolves(category):
    assert lookup_triage_policy(category.upper())["category"] == category


def test_tool_is_read_only():
    """Calling the tool twice returns identical data (no state change)."""
    assert lookup_triage_policy("bug") == lookup_triage_policy("bug")
