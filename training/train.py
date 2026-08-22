"""
Fine-tune a transformer sequence classifier for groundedness detection.

Input format at [CLS]question [SEP] context[SEP]answer[SEP]: the tokenizer's
text/text_pair mechanism is used with `text=f"{question} [SEP] {context}"`
and `text_pair=answer`, so the model sees question+context as one segment
and the candidate answer as the other -- this lets the model attend
specifically to whether the answer segment is entailed by the
question+context segment, which is the actual groundedness question.

Base model defaults to `microsoft/deberta-v3-small`: chosen because it is
small enough to fine-tune and run inference on CPU in a reasonable time
(~140M params) while its ELECTRA-style pretraining and disentangled
attention give it strong out-of-the-box performance on NLI-style
entailment/contradiction tasks, which is structurally what groundedness
classification is. DistilBERT and MiniLM are documented alternatives in
config (GROUNDEDNESS_BASE_MODEL) if an even smaller/faster model is needed
for constrained hardware -- both trade some accuracy for speed.

Usage:
    python -m training.train \
        --train data/processed/groundedness_train.jsonl \
        --val data/processed/groundedness_val.jsonl \
        --base-model microsoft/deberta-v3-small \
        --out models/groundedness-classifier \
        --epochs 3 --batch-size 8
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

LABELS = ["CONTRADICTED", "SUPPORTED", "UNSUPPORTED"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_hf_dataset(examples: list[dict], tokenizer, max_length: int):
    from datasets import Dataset

    def tokenize(batch):
        first = [f"{q} [SEP] {c}" for q, c in zip(batch["question"], batch["context"])]
        return tokenizer(
            first, batch["answer"], truncation=True, max_length=max_length, padding="max_length"
        )

    ds = Dataset.from_list(
        [{"question": e["question"], "context": e["context"], "answer": e["answer"],
          "label": LABEL2ID[e["label"]]} for e in examples]
    )
    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["question", "context", "answer"])
    ds.set_format(type="torch")
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(LABELS))), zero_division=0
    )
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    metrics = {"accuracy": accuracy_score(labels, preds), "macro_f1": macro_f1}
    for i, label in enumerate(LABELS):
        metrics[f"precision_{label}"] = precision[i]
        metrics[f"recall_{label}"] = recall[i]
        metrics[f"f1_{label}"] = f1[i]
    return metrics


def report_class_balance(examples: list[dict], split_name: str) -> None:
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex["label"]] = counts.get(ex["label"], 0) + 1
    print(f"[{split_name}] label distribution: {counts}")
    if counts and (max(counts.values()) / max(min(counts.values()), 1)) > 3:
        print(
            f"[{split_name}] WARNING: class imbalance detected (largest/smallest class "
            f"ratio > 3x). Consider class weighting or generating more minority-class "
            f"examples via training/generate_dataset.py before trusting raw accuracy."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/processed/groundedness_train.jsonl")
    parser.add_argument("--val", default="data/processed/groundedness_val.jsonl")
    parser.add_argument("--base-model", default="microsoft/deberta-v3-small")
    parser.add_argument("--out", default="models/groundedness-classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    train_examples = load_jsonl(args.train)
    val_examples = load_jsonl(args.val)
    if not train_examples:
        raise SystemExit(f"No training examples found in {args.train}")
    report_class_balance(train_examples, "train")
    report_class_balance(val_examples, "val")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_ds = build_hf_dataset(train_examples, tokenizer, args.max_length)
    val_ds = build_hf_dataset(val_examples, tokenizer, args.max_length) if val_examples else None

    training_args = TrainingArguments(
        output_dir=f"{args.out}-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="macro_f1" if val_ds is not None else None,
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics if val_ds is not None else None,
    )

    train_result = trainer.train()
    print(f"Training loss (final): {train_result.training_loss:.4f}")

    if val_ds is not None:
        val_metrics = trainer.evaluate()
        print("Validation metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
