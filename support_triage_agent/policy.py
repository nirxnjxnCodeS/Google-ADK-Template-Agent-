"""Frozen triage policy (guide §4, §9). Ground truth for the tool, instructions and eval labels.

Do not edit this to make a model look better. Any change bumps POLICY_VERSION and
requires re-running the locked eval.
"""

POLICY_VERSION = "v1"

CATEGORIES = {"access", "billing", "bug", "security", "general"}

POLICIES = {
    "access": {
        "description": "Login, permissions, account lockout, MFA, or role access.",
        "priority_rules": "P2 if multiple users cannot access a core system; P3 for one user blocked; P4 for a how-to question.",
        "human_review": "Required if privileged/admin access is requested or identity cannot be verified.",
    },
    "billing": {
        "description": "Invoice, charge, subscription, refund, or payment questions.",
        "priority_rules": "P2 if service is suspended for multiple users; P3 for a disputed charge; P4 for a normal invoice question.",
        "human_review": "Required for refunds, charge disputes, or account ownership ambiguity.",
    },
    "bug": {
        "description": "Unexpected product behavior, errors, crashes, or broken workflows.",
        "priority_rules": "P1 for a full production outage; P2 for major multi-user degradation without workaround; P3 for a single-user/core-workflow bug; P4 for cosmetic issues.",
        "human_review": "Required for P1 and P2.",
    },
    "security": {
        "description": "Suspected compromise, leaked credentials, malicious activity, phishing, data exposure, or unauthorized access.",
        "priority_rules": "P1 for active compromise/data exposure; P2 for credible suspicious activity not yet confirmed.",
        "human_review": "Always required.",
    },
    "general": {
        "description": "Questions that do not fit another category.",
        "priority_rules": "Usually P4. Use UNKNOWN when impact is too unclear to judge.",
        "human_review": "Not normally required.",
    },
}
