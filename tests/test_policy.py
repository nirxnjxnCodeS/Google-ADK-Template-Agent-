from support_triage_agent.policy import CATEGORIES, POLICIES, POLICY_VERSION


def test_all_categories_have_policy():
    assert set(POLICIES) == CATEGORIES


def test_security_requires_human_review_text():
    assert "Always required" in POLICIES["security"]["human_review"]


def test_policy_version_recorded():
    assert POLICY_VERSION == "v1"
