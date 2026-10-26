"""Output contracts (guide §12). Executable validators, not prompts."""
from typing import Literal

from pydantic import BaseModel, Field


class TicketFacts(BaseModel):
    """Stage-1 extractor output (gemini_pipeline profile only)."""
    issue_summary: str
    category_hint: Literal["access", "billing", "bug", "security", "general", "unknown"]
    affected_scope: Literal["single_user", "multiple_users", "all_users", "unknown"]
    security_signal: bool
    billing_signal: bool
    missing_facts: list[str] = Field(default_factory=list)


class TriageOutput(BaseModel):
    """Final answer contract. The eval parses every final response with this."""
    category: Literal["access", "billing", "bug", "security", "general"]
    priority: Literal["P1", "P2", "P3", "P4", "UNKNOWN"]
    human_review: bool
    needs_more_info: bool
    reason: str = Field(min_length=1, max_length=240)
    next_action: str = Field(min_length=1, max_length=240)
