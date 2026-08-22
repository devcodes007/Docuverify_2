"""
Groundedness / hallucination detection.

The real classifier (`TransformerGroundednessClassifier`) is a fine-tuned
sequence classifier loaded from `GROUNDEDNESS_MODEL` (see training/), fed
`[CLS] question [SEP] context [SEP] answer [SEP]` and predicting
SUPPORTED/CONTRADICTED/UNSUPPORTED -- this is the deep-learning centerpiece
described in the project spec, not another LLM prompt.

`HeuristicGroundednessClassifier` is a lexical-overlap fallback used when no
fine-tuned model is available yet (e.g. before training has been run, or in
this sandbox where model weights can't be downloaded). It is clearly weaker
than a trained classifier and is only meant to keep the end-to-end pipeline
runnable/testable; `build_groundedness_classifier` prefers the real model
and only falls back with a logged warning.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from app.logging_config import get_logger
from app.models.schemas import GroundednessLabel, GroundednessResult

logger = get_logger(__name__)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "does", "do", "how", "what", "when",
    "why", "of", "to", "in", "on", "for", "and", "or", "with", "that",
    "this", "it", "its", "be", "can", "you", "your", "which", "not",
}

_NEGATION_WORDS = {"not", "never", "cannot", "can't", "won't", "isn't", "doesn't", "don't", "no"}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class GroundednessClassifier(Protocol):
    def predict(self, question: str, context: str, answer: str) -> GroundednessResult: ...


class TransformerGroundednessClassifier:
    """Loads a fine-tuned sequence classification model produced by
    training/train.py. See README for how to train and where the model is
    expected on disk (GROUNDEDNESS_MODEL)."""

    LABELS = [GroundednessLabel.CONTRADICTED, GroundednessLabel.SUPPORTED, GroundednessLabel.UNSUPPORTED]

    def __init__(self, model_path: str, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not Path(model_path).exists():
            raise FileNotFoundError(f"No fine-tuned model found at {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.max_length = max_length
        self._torch = torch

    def predict(self, question: str, context: str, answer: str) -> GroundednessResult:
        inputs = self.tokenizer(
            f"{question} [SEP] {context}",
            answer,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = self._torch.softmax(logits, dim=-1).tolist()

        per_class = {label.value: round(p, 4) for label, p in zip(self.LABELS, probs)}
        best_idx = max(range(len(probs)), key=lambda i: probs[i])
        return GroundednessResult(
            label=self.LABELS[best_idx], confidence=round(probs[best_idx], 4), per_class_scores=per_class
        )


class HeuristicGroundednessClassifier:
    """Deterministic fallback: checks (a) whether the answer's claim terms
    are covered by the retrieved context (-> UNSUPPORTED if not), and
    (b) whether the answer negates something the context asserts, or vice
    versa, using simple negation-word matching on shared subjects (->
    CONTRADICTED). Otherwise SUPPORTED. This is intentionally simple and is
    not a substitute for the trained classifier -- it exists so /query and
    /evaluate are exercisable without a trained model present.
    """

    def predict(self, question: str, context: str, answer: str) -> GroundednessResult:
        context_terms = _terms(context)
        answer_terms = _terms(answer)
        if not answer_terms:
            return GroundednessResult(label=GroundednessLabel.UNSUPPORTED, confidence=0.5)

        overlap = answer_terms & context_terms
        coverage = len(overlap) / len(answer_terms)

        answer_negated = bool(_NEGATION_WORDS & set(answer.lower().split()))
        context_negated = bool(_NEGATION_WORDS & set(context.lower().split()))
        shares_subject = len(overlap) >= max(2, len(answer_terms) // 3)

        if shares_subject and answer_negated != context_negated and coverage > 0.3:
            confidence = round(0.5 + min(coverage, 0.4), 4)
            return GroundednessResult(
                label=GroundednessLabel.CONTRADICTED,
                confidence=confidence,
                per_class_scores={"CONTRADICTED": confidence},
            )

        if coverage >= 0.5:
            confidence = round(0.5 + min(coverage, 0.45), 4)
            return GroundednessResult(
                label=GroundednessLabel.SUPPORTED,
                confidence=confidence,
                per_class_scores={"SUPPORTED": confidence},
            )

        confidence = round(0.5 + (0.5 - coverage), 4)
        return GroundednessResult(
            label=GroundednessLabel.UNSUPPORTED,
            confidence=min(confidence, 0.95),
            per_class_scores={"UNSUPPORTED": min(confidence, 0.95)},
        )


def build_groundedness_classifier(model_path: str, max_length: int = 512) -> GroundednessClassifier:
    try:
        return TransformerGroundednessClassifier(model_path, max_length=max_length)
    except Exception as exc:  # noqa: BLE001 - capability probe, same pattern as build_embedder
        logger.warning(
            "falling back to HeuristicGroundednessClassifier: no usable model at %s (%s)",
            model_path, exc,
        )
        return HeuristicGroundednessClassifier()
