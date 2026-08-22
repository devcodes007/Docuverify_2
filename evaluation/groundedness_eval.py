"""
Evaluates the classifier that would actually be loaded by the running app
(build_groundedness_classifier against GROUNDEDNESS_MODEL -- the fine-tuned
transformer if present, otherwise the heuristic fallback) against a small
hand-labeled spot-check set (evaluation/groundedness_eval_set.json).

This is distinct from training/evaluate.py, which reports the fine-tuned
model's performance on its own held-out test split from the training data
generation pipeline; this script is meant to be re-run against whatever
classifier is actually configured for a deployment (including the
heuristic fallback, so its accuracy is visible and not just assumed) as a
lightweight regression check independent of the training pipeline's own
data.

Usage:
    python -m evaluation.groundedness_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from app.config import get_settings
from app.verification.groundedness import (
    TransformerGroundednessClassifier,
    build_groundedness_classifier,
)

LABELS = ["CONTRADICTED", "SUPPORTED", "UNSUPPORTED"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", default="evaluation/groundedness_eval_set.json")
    parser.add_argument("--out", default="evaluation/results/groundedness_live_eval_report.json")
    args = parser.parse_args()

    settings = get_settings()
    classifier = build_groundedness_classifier(settings.groundedness_model, settings.groundedness_max_length)
    classifier_kind = (
        "TransformerGroundednessClassifier" if isinstance(classifier, TransformerGroundednessClassifier)
        else "HeuristicGroundednessClassifier"
    )

    examples = json.loads(Path(args.eval_set).read_text())
    y_true, y_pred = [], []
    for ex in examples:
        result = classifier.predict(ex["question"], ex["context"], ex["answer"])
        y_true.append(LABEL2IDX[ex["label"]])
        y_pred.append(LABEL2IDX[result.label.value])

    report = classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))

    result = {
        "classifier": classifier_kind,
        "note": (
            "HeuristicGroundednessClassifier is a lexical-overlap fallback, not the "
            "fine-tuned model; its accuracy here should not be reported as the "
            "project's groundedness classifier performance -- see training/evaluate.py "
            "for the fine-tuned model's held-out test metrics."
        ) if classifier_kind == "HeuristicGroundednessClassifier" else None,
        "eval_examples": len(examples),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"classifier: {classifier_kind}")
    if result["note"]:
        print(f"NOTE: {result['note']}")
    print(f"accuracy: {result['accuracy']:.4f}")
    print(f"macro_f1: {result['macro_f1']:.4f}")
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
