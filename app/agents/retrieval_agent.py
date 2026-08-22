"""
The retrieval agent: implements the bounded retry loop and the three
retrieval plans (simple hybrid, per-concept comparison, multi-hop
decomposition). This is the module that makes retrieval "agentic" rather
than a single fixed call -- it decides how to retrieve, checks whether the
result is good enough, and reformulates/retries when it isn't, up to
MAX_RETRIEVAL_ATTEMPTS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.agents.evidence_evaluator import EvidenceEvaluator
from app.agents.query_router import extract_comparison_concepts
from app.agents.query_rewriter import reformulate
from app.logging_config import get_logger, log_event
from app.models.schemas import (
    QueryType,
    RetrievalAttempt,
    RetrievalStrategy,
    RetrievedChunk,
)
from app.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


def decompose_multi_hop(question: str) -> list[str]:
    """Best-effort decomposition of a multi-hop question into subquestions.

    Simple, explainable heuristic: split on the connective that signals the
    interaction ("interact with", "affect", "and how does ... relate to"),
    producing "how does X work" for each side plus a synthesis question. This
    is intentionally not an LLM call -- it's fast, deterministic, and good
    enough for documentation questions that follow "how does A interact with
    B" phrasing; genuinely novel phrasing falls back to using the whole
    question as a single hop.
    """
    q = question.strip().rstrip("?")
    lowered = q.lower()

    for connector in ("interact with", "affect", "relate to", "connect to"):
        if connector in lowered:
            idx = lowered.index(connector)
            left = q[:idx].strip()
            right = q[idx + len(connector):].strip()
            left = _to_how_does_question(left)
            right = _to_how_does_question(right)
            if left and right:
                return [
                    left,
                    right,
                    f"How do {_bare_subject(left)} and {_bare_subject(right)} work together in the same request?",
                ]

    return [question]


def _to_how_does_question(fragment: str) -> str:
    fragment = fragment.strip().strip(",")
    fragment = fragment.split(" and how does ")[-1]
    fragment = re.sub(r"^(how does|does|how)\s+", "", fragment, flags=re.IGNORECASE).strip()
    if not fragment:
        return ""
    return f"How does {fragment} work?"


def _bare_subject(how_question: str) -> str:
    m = re.match(r"How does (.+) work\?*$", how_question)
    return m.group(1) if m else how_question


@dataclass
class RetrievalOutcome:
    strategy: RetrievalStrategy
    results: list[RetrievedChunk]
    attempts: list[RetrievalAttempt] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    hop_evidence: dict[str, list[RetrievedChunk]] = field(default_factory=dict)


class RetrievalAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        evaluator: EvidenceEvaluator,
        max_attempts: int = 3,
        hybrid_alpha: float = 0.6,
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.max_attempts = max_attempts
        self.hybrid_alpha = hybrid_alpha

    # -- public entrypoint -------------------------------------------------

    def run(self, question: str, query_type: QueryType, top_k: int) -> RetrievalOutcome:
        if query_type == QueryType.COMPARISON:
            return self._run_comparison(question, top_k)
        if query_type == QueryType.MULTI_HOP:
            return self._run_multi_hop(question, top_k)
        return self._run_simple(question, top_k)

    # -- SIMPLE_LOOKUP -------------------------------------------------------

    def _run_simple(self, question: str, top_k: int) -> RetrievalOutcome:
        attempts: list[RetrievalAttempt] = []
        query = question
        strategy = RetrievalStrategy.HYBRID
        results: list[RetrievedChunk] = []

        for attempt_num in range(1, self.max_attempts + 1):
            strategy = RetrievalStrategy.HYBRID if attempt_num < 3 else RetrievalStrategy.DENSE
            results = self._retrieve(query, strategy, top_k)
            evaluation = self.evaluator.evaluate(question, results)
            attempts.append(
                RetrievalAttempt(
                    attempt=attempt_num,
                    query=query,
                    strategy=strategy,
                    documents_retrieved=len(results),
                    evidence=evaluation,
                )
            )
            log_event(
                logger, "retrieval_attempt",
                attempt=attempt_num, query=query, strategy=strategy.value,
                documents_retrieved=len(results), sufficient=evaluation.sufficient,
                confidence=evaluation.confidence,
            )
            if evaluation.sufficient:
                break
            if attempt_num < self.max_attempts:
                query = reformulate(question, evaluation, attempt_num + 1)

        return RetrievalOutcome(strategy=strategy, results=results, attempts=attempts)

    # -- COMPARISON -----------------------------------------------------------

    def _run_comparison(self, question: str, top_k: int) -> RetrievalOutcome:
        concepts = extract_comparison_concepts(question)
        if concepts is None:
            # fall back to simple retrieval if we can't split the concepts
            return self._run_simple(question, top_k)

        concept_a, concept_b = concepts
        attempts: list[RetrievalAttempt] = []
        hop_evidence: dict[str, list[RetrievedChunk]] = {}
        per_concept_top_k = max(top_k // 2, 3)

        for i, concept in enumerate((concept_a, concept_b), start=1):
            results = self._retrieve(concept, RetrievalStrategy.HYBRID, per_concept_top_k)
            evaluation = self.evaluator.evaluate(concept, results)
            attempts.append(
                RetrievalAttempt(
                    attempt=i,
                    query=concept,
                    strategy=RetrievalStrategy.HYBRID,
                    documents_retrieved=len(results),
                    evidence=evaluation,
                )
            )
            hop_evidence[concept] = results
            log_event(
                logger, "comparison_retrieval", concept=concept,
                documents_retrieved=len(results), sufficient=evaluation.sufficient,
            )

        combined = _dedupe_by_chunk_id(
            [r for results in hop_evidence.values() for r in results]
        )
        return RetrievalOutcome(
            strategy=RetrievalStrategy.HYBRID,
            results=combined,
            attempts=attempts,
            subqueries=[concept_a, concept_b],
            hop_evidence=hop_evidence,
        )

    # -- MULTI_HOP ------------------------------------------------------------

    def _run_multi_hop(self, question: str, top_k: int) -> RetrievalOutcome:
        subqueries = decompose_multi_hop(question)
        attempts: list[RetrievalAttempt] = []
        hop_evidence: dict[str, list[RetrievedChunk]] = {}
        per_hop_top_k = max(top_k // max(len(subqueries), 1), 3)

        for i, subq in enumerate(subqueries, start=1):
            results = self._retrieve(subq, RetrievalStrategy.HYBRID, per_hop_top_k)
            evaluation = self.evaluator.evaluate(subq, results)
            attempts.append(
                RetrievalAttempt(
                    attempt=i,
                    query=subq,
                    strategy=RetrievalStrategy.HYBRID,
                    documents_retrieved=len(results),
                    evidence=evaluation,
                )
            )
            hop_evidence[subq] = results
            log_event(
                logger, "multi_hop_retrieval", hop=i, question=subq,
                documents_retrieved=len(results), sufficient=evaluation.sufficient,
            )

        combined = _dedupe_by_chunk_id(
            [r for results in hop_evidence.values() for r in results]
        )

        # overall sufficiency check across all hops, used by the orchestrator
        overall_eval = self.evaluator.evaluate(question, combined, required_subqueries=subqueries)
        attempts.append(
            RetrievalAttempt(
                attempt=len(attempts) + 1,
                query=question,
                strategy=RetrievalStrategy.HYBRID,
                documents_retrieved=len(combined),
                evidence=overall_eval,
            )
        )

        return RetrievalOutcome(
            strategy=RetrievalStrategy.HYBRID,
            results=combined,
            attempts=attempts,
            subqueries=subqueries,
            hop_evidence=hop_evidence,
        )

    # -- shared --------------------------------------------------------------

    def _retrieve(self, query: str, strategy: RetrievalStrategy, top_k: int) -> list[RetrievedChunk]:
        if strategy == RetrievalStrategy.BM25:
            return self.retriever.search_bm25(query, top_k)
        if strategy == RetrievalStrategy.DENSE:
            return self.retriever.search_dense(query, top_k)
        return self.retriever.search_hybrid(query, top_k, alpha=self.hybrid_alpha)


def _dedupe_by_chunk_id(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: dict[str, RetrievedChunk] = {}
    for r in results:
        existing = seen.get(r.chunk.chunk_id)
        if existing is None or r.score > existing.score:
            seen[r.chunk.chunk_id] = r
    return sorted(seen.values(), key=lambda r: r.score, reverse=True)
