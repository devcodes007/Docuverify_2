"""
Split the raw groundedness dataset into train/val/test.

Splits by `document_id`, not randomly per-example: every example generated
from the same source document goes entirely into one split. Splitting
randomly at the example level would leak near-duplicate context (many
examples share the same underlying chunk text) between train and
test/val, inflating reported accuracy. This is a document-level split by
design, per the project's "avoid train/test leakage" requirement.

Usage:
    python -m training.prepare_dataset --in data/processed/groundedness_raw.jsonl --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(13)


def load_examples(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def split_by_document(
    examples: list[dict], train_frac: float = 0.7, val_frac: float = 0.15
) -> tuple[list[dict], list[dict], list[dict]]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_doc[ex["document_id"]].append(ex)

    doc_ids = list(by_doc.keys())
    random.shuffle(doc_ids)
    n = len(doc_ids)

    if n == 1:
        train_ids, val_ids, test_ids = set(doc_ids), set(), set()
    elif n == 2:
        train_ids, val_ids, test_ids = {doc_ids[0]}, set(), {doc_ids[1]}
    elif n <= 4:
        # too few documents for a proportional split to be meaningful;
        # guarantee exactly one held-out doc each for val and test
        train_ids = set(doc_ids[:-2])
        val_ids = {doc_ids[-2]}
        test_ids = {doc_ids[-1]}
    else:
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        n_train = min(n_train, n - 2)  # always leave >=1 for val and >=1 for test
        train_ids = set(doc_ids[:n_train])
        val_ids = set(doc_ids[n_train:n_train + n_val])
        test_ids = set(doc_ids[n_train + n_val:])

    train = [ex for doc_id in train_ids for ex in by_doc[doc_id]]
    val = [ex for doc_id in val_ids for ex in by_doc[doc_id]]
    test = [ex for doc_id in test_ids for ex in by_doc[doc_id]]
    return train, val, test


def write_jsonl(path: str | Path, examples: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", default="data/processed/groundedness_raw.jsonl")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    examples = load_examples(args.input_path)
    if not examples:
        raise SystemExit(f"No examples found in {args.input_path}")

    train, val, test = split_by_document(examples, args.train_frac, args.val_frac)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "groundedness_train.jsonl", train)
    write_jsonl(out_dir / "groundedness_val.jsonl", val)
    write_jsonl(out_dir / "groundedness_test.jsonl", test)

    train_docs = {ex["document_id"] for ex in train}
    val_docs = {ex["document_id"] for ex in val}
    test_docs = {ex["document_id"] for ex in test}
    assert not (train_docs & val_docs), "leakage: a document appears in both train and val"
    assert not (train_docs & test_docs), "leakage: a document appears in both train and test"
    assert not (val_docs & test_docs), "leakage: a document appears in both val and test"

    print(f"train: {len(train)} examples from {len(train_docs)} documents")
    print(f"val:   {len(val)} examples from {len(val_docs)} documents")
    print(f"test:  {len(test)} examples from {len(test_docs)} documents")
    print("Leakage check passed: no document appears in more than one split.")


if __name__ == "__main__":
    main()
