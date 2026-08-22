"""
Query classification.

Rule-based first: cheap, deterministic, and testable, and documentation
questions have fairly reliable surface patterns for COMPARISON and
MULTI_HOP. An LLM-based fallback classifier is used only when the rules are
genuinely ambiguous, so the router doesn't add an LLM round-trip (and a new
point of failure) to every single query.
"""
from __future__ import annotations

import re
from typing import Protocol

from app.models.schemas import QueryType

_COMPARISON_PATTERNS = [
    r"\bdifference between\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bwhich (is|one) (better|should)\b",
]

_MULTI_HOP_PATTERNS = [
    r"\band how does\b.*\baffect\b",
    r"\binteract with\b",
    r"\bwhen .* and .* happens?\b",
    r"\bhow does .* (relate|connect) to\b",
    r"\bwhat happens when\b.*\band\b",
]

_MULTI_HOP_CONNECTOR_THRESHOLD = 2  # "and"/"then"/"which in turn" count


class LLMClassifier(Protocol):
    def classify(self, question: str) -> QueryType: ...


class QueryRouter:
    def __init__(self, llm_fallback: LLMClassifier | None = None):
        self.llm_fallback = llm_fallback

    def classify(self, question: str) -> QueryType:
        q = question.strip().lower()

        for pattern in _COMPARISON_PATTERNS:
            if re.search(pattern, q):
                return QueryType.COMPARISON

        for pattern in _MULTI_HOP_PATTERNS:
            if re.search(pattern, q):
                return QueryType.MULTI_HOP

        # heuristic: multiple independent clauses chained with connectors
        # tends to indicate a question that needs more than one retrieval hop
        connector_count = len(re.findall(r"\band\b|\bthen\b|\bafter that\b", q))
        if connector_count >= _MULTI_HOP_CONNECTOR_THRESHOLD:
            return QueryType.MULTI_HOP

        if self.llm_fallback is not None and self._is_ambiguous(q):
            try:
                return self.llm_fallback.classify(question)
            except Exception:  # noqa: BLE001 - never let classification crash the request
                pass

        return QueryType.SIMPLE_LOOKUP

    @staticmethod
    def _is_ambiguous(q: str) -> bool:
        # Only defer to the LLM for longer, structurally complex questions;
        # short direct lookups ("what is Depends?") are never ambiguous.
        return len(q.split()) > 18


def extract_comparison_concepts(question: str) -> tuple[str, str] | None:
    """Best-effort extraction of the two things being compared, e.g.
    "difference between Depends and middleware" -> ("Depends", "middleware").
    Returns None if it can't confidently split the question."""
    q = question.strip().rstrip("?")
    match = re.search(
        r"(?:difference between|compare)\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+)",
        q,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" `"), match.group(2).strip(" `")

    match = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", q, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip(" `"), match.group(2).strip(" `")
    return None
