"""
Evidence sufficiency evaluation.

Deterministic by default: combines retrieval score strength, chunk count,
and query-term coverage of the retrieved text into a single confidence
score. This is intentionally not "assume top-k retrieval is enough" -- a
query can retrieve 8 chunks that all score low, or that all miss half the
query's key terms, and this evaluator is what catches that.

An LLM-based evaluator is provided behind the same interface for cases
where deterministic signals are ambiguous, but is not required for the
system to function -- deterministic evaluation has no external dependency
and is what the retrieval agent uses by default.
"""
from __future__ import annotations

import re
from typing import Protocol

from app.models.schemas import EvidenceEvaluation, RetrievedChunk

_STOPWORDS = {
    "the", "a", "an", "is", "are", "does", "do", "how", "what", "when", "from",
    "why", "of", "to", "in", "on", "for", "and", "or", "with", "that",
    "this", "it", "its", "be", "can", "you", "your", "which",
    "give", "summary", "summarize", "pdf", "provided", "please", "tell",
    "show", "explain", "document", "file",
}


def _key_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class EvidenceEvaluator(Protocol):
    def evaluate(
        self, query: str, results: list[RetrievedChunk], required_subqueries: list[str] | None = None
    ) -> EvidenceEvaluation: ...


class DeterministicEvidenceEvaluator:
    def __init__(self, sufficiency_threshold: float = 0.55, min_chunks: int = 1):
        self.sufficiency_threshold = sufficiency_threshold
        self.min_chunks = min_chunks

    def evaluate(
        self,
        query: str,
        results: list[RetrievedChunk],
        required_subqueries: list[str] | None = None,
    ) -> EvidenceEvaluation:
        if not results:
            return EvidenceEvaluation(
                sufficient=False,
                confidence=0.0,
                missing_information=[query],
                reason="No documents were retrieved for this query.",
            )

        top_score = max(r.score for r in results)
        # scores here may already be normalized (hybrid) or not (raw bm25);
        # clamp defensively so confidence stays in [0, 1]
        top_score_clamped = max(0.0, min(top_score, 1.0)) if top_score <= 1.0 else 1.0

        query_terms = _key_terms(query)
        covered_text = " ".join(r.chunk.text.lower() for r in results)
        covered_terms = {t for t in query_terms if t in covered_text}
        coverage_ratio = (len(covered_terms) / len(query_terms)) if query_terms else 1.0

        chunk_count_factor = min(len(results) / max(self.min_chunks, 1), 1.0)

        # subquestion coverage matters for multi-hop / comparison: each
        # required subquery should have at least one term hit in the results
        missing: list[str] = []
        subquery_coverage = 1.0
        if required_subqueries:
            hits = 0
            for sub in required_subqueries:
                sub_terms = _key_terms(sub)
                if sub_terms and (sub_terms & covered_terms):
                    hits += 1
                else:
                    missing.append(sub)
            subquery_coverage = hits / len(required_subqueries)

        confidence = (
            0.45 * top_score_clamped
            + 0.30 * coverage_ratio
            + 0.10 * chunk_count_factor
            + 0.15 * subquery_coverage
        )
        confidence = round(min(max(confidence, 0.0), 1.0), 4)

        missing_terms = sorted(query_terms - covered_terms)
        if missing_terms and not missing:
            missing = [f"coverage for term(s): {', '.join(missing_terms[:5])}"]

        sufficient = confidence >= self.sufficiency_threshold and len(results) >= self.min_chunks

        reason = (
            f"top_score={top_score_clamped:.2f}, term_coverage={coverage_ratio:.2f}, "
            f"chunks={len(results)}, subquery_coverage={subquery_coverage:.2f}"
        )
        return EvidenceEvaluation(
            sufficient=sufficient,
            confidence=confidence,
            missing_information=missing,
            reason=reason,
        )
