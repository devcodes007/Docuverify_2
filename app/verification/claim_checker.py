"""
Optional claim-level verification.

Splits a generated answer into individual sentence-level claims and runs the
groundedness classifier on each claim independently against the retrieved
context, so a partially-correct answer can be flagged claim-by-claim rather
than as a single pass/fail. Implemented as an optional module on top of the
existing GroundednessClassifier interface -- it adds no new model, just
finer-grained application of the same one.
"""
from __future__ import annotations

import re

from app.models.schemas import ClaimVerification
from app.verification.groundedness import GroundednessClassifier

_CITATION_RE = re.compile(r"\[(\d+)\]")


def split_into_claims(answer: str) -> list[str]:
    # simple sentence segmentation; good enough for generated documentation
    # answers, which tend to be short declarative sentences with citations
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s.strip() for s in sentences if s.strip()]


def verify_claims(
    question: str,
    context: str,
    answer: str,
    classifier: GroundednessClassifier,
    evidence_by_index: dict[int, str] | None = None,
) -> list[ClaimVerification]:
    claims = split_into_claims(answer)
    results: list[ClaimVerification] = []
    for claim in claims:
        cited_indices = [int(m) for m in _CITATION_RE.findall(claim)]
        evidence_ids = (
            [f"[{i}]" for i in cited_indices]
            if cited_indices
            else []
        )
        verdict = classifier.predict(question=question, context=context, answer=claim)
        results.append(
            ClaimVerification(claim=claim, status=verdict.label, evidence=evidence_ids)
        )
    return results
