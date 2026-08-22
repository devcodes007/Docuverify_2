"""
Evaluate a fine-tuned groundedness classifier on the held-out test split.

Reports accuracy, macro F1, per-class precision/recall/F1, and a confusion
matrix, and writes them to evaluation/results/groundedness_test_report.json
so results are never just printed and lost. Never fabricates numbers: if
this hasn't been run, there is no report file, and the README says so.

Usage:
    python -m training.evaluate --model models/groundedness-classifier \
        --test data/processed/groundedness_test.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

from training.train import LABEL2ID, LABELS, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/groundedness-classifier")
    parser.add_argument("--test", default="data/processed/groundedness_test.jsonl")
    parser.add_argument("--out", default="evaluation/results/groundedness_test_report.json")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    test_examples = load_jsonl(args.test)
    if not test_examples:
        raise SystemExit(f"No test examples found in {args.test}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    y_true, y_pred = [], []
    for ex in test_examples:
        inputs = tokenizer(
            f"{ex['question']} [SEP] {ex['context']}", ex["answer"],
            truncation=True, max_length=args.max_length, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        pred_id = int(torch.argmax(logits).item())
        y_true.append(LABEL2ID[ex["label"]])
        y_pred.append(pred_id)

    report = classification_report(
        y_true, y_pred, labels=list(range(len(LABELS))), target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))

    result = {
        "model_path": args.model,
        "test_examples": len(test_examples),
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {label: report[label] for label in LABELS},
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm.tolist(),  # rows = true label, cols = predicted label
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"accuracy: {result['accuracy']:.4f}")
    print(f"macro_f1: {result['macro_f1']:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print(f"  {LABELS}")
    for label, row in zip(LABELS, cm.tolist()):
        print(f"  {label}: {row}")
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
