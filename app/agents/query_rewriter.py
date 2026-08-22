"""
Query reformulation.

Uses the *previous* attempt's evidence gap (missing_information from the
evaluator) rather than a generic paraphrase, so each retry is targeted at
what specifically failed rather than just rephrasing the same question.
"""
from __future__ import annotations

import re

from app.models.schemas import EvidenceEvaluation

_QUESTION_WORDS = {"how", "what", "why", "when", "does", "do", "is", "are", "the", "a", "an"}


def _strip_stopwords(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    kept = [w for w in words if w not in _QUESTION_WORDS]
    return " ".join(kept)


def reformulate(original_query: str, previous_evaluation: EvidenceEvaluation, attempt: int) -> str:
    """Attempt 2: strip question-word framing and append missing-term hints
    (pure keyword form tends to help BM25 & retrieval on docs corpora).
    Attempt 3: fall back to a keyword-only form of just the missing terms,
    to try to surface documents the previous attempts entirely missed."""
    base = _strip_stopwords(original_query)

    if attempt == 2:
        extra_terms = []
        for item in previous_evaluation.missing_information:
            if item.startswith("coverage for term(s):"):
                extra_terms.extend(t.strip() for t in item.split(":", 1)[1].split(","))
            else:
                extra_terms.append(_strip_stopwords(item))
        extra = " ".join(t for t in extra_terms if t)
        return f"{base} {extra}".strip()

    # attempt >= 3: keyword-only, prioritizing whatever was still missing
    extra_terms = []
    for item in previous_evaluation.missing_information:
        if item.startswith("coverage for term(s):"):
            extra_terms.extend(t.strip() for t in item.split(":", 1)[1].split(","))
        else:
            extra_terms.append(_strip_stopwords(item))
    if extra_terms:
        return " ".join(dict.fromkeys(extra_terms))  # dedupe, preserve order
    return base
