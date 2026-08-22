"""
Smoke-tests the actual training mechanics used by training/train.py
(dataset building, tokenization scheme, Trainer wiring, compute_metrics)
against a tiny, from-scratch BERT config + WordLevel tokenizer, so the
training loop is proven to run correctly end-to-end without requiring a
network download of a real pretrained model. `training/train.py` itself
targets a real base model (default microsoft/deberta-v3-small) via
AutoTokenizer/AutoModelForSequenceClassification.from_pretrained -- this
test only substitutes what the base model *is*, not how it's trained.
"""
from __future__ import annotations

import json

import pytest

from training.train import LABEL2ID, LABELS, build_hf_dataset, compute_metrics


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture()
def tiny_examples():
    return [
        {"question": "How does Depends work?", "context": "FastAPI uses Depends for dependency injection.",
         "answer": "FastAPI uses Depends for dependency injection.", "label": "SUPPORTED"},
        {"question": "How does Depends work?", "context": "Middleware wraps every request in FastAPI.",
         "answer": "FastAPI uses Depends for dependency injection.", "label": "UNSUPPORTED"},
        {"question": "How does Depends work?", "context": "FastAPI uses Depends for dependency injection.",
         "answer": "FastAPI does not use Depends for dependency injection.", "label": "CONTRADICTED"},
        {"question": "What is middleware?", "context": "Middleware wraps every request and response.",
         "answer": "Middleware wraps every request and response.", "label": "SUPPORTED"},
    ]


@pytest.fixture()
def tiny_tokenizer(tmp_path):
    """A small WordLevel tokenizer trained on the fly -- no network access,
    no Hugging Face Hub download -- just enough vocabulary to tokenize the
    tiny_examples fixture, so build_hf_dataset/Trainer can run for real."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"])
    corpus = [
        "FastAPI uses Depends for dependency injection",
        "Middleware wraps every request and response in FastAPI",
        "FastAPI does not use Depends for dependency injection",
        "How does Depends work",
        "What is middleware",
    ]
    tok.train_from_iterator(corpus, trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", pad_token="[PAD]",
    )
    return fast_tokenizer


def test_build_hf_dataset_produces_expected_labels_and_fields(tiny_examples, tiny_tokenizer):
    ds = build_hf_dataset(tiny_examples, tiny_tokenizer, max_length=32)
    assert len(ds) == len(tiny_examples)
    assert "input_ids" in ds.column_names
    assert "label" in ds.column_names
    expected_labels = [LABEL2ID[e["label"]] for e in tiny_examples]
    assert [int(x) for x in ds["label"]] == expected_labels


def test_compute_metrics_reports_all_expected_keys():
    import numpy as np

    # 4 examples, 3 classes; construct logits that get 3/4 correct
    logits = np.array([
        [0.1, 5.0, 0.1],   # predicts SUPPORTED (label idx 1)
        [0.1, 0.1, 5.0],   # predicts UNSUPPORTED (label idx 2)
        [5.0, 0.1, 0.1],   # predicts CONTRADICTED (label idx 0)
        [0.1, 5.0, 0.1],   # predicts SUPPORTED
    ])
    labels = np.array([LABEL2ID["SUPPORTED"], LABEL2ID["UNSUPPORTED"], LABEL2ID["CONTRADICTED"], LABEL2ID["UNSUPPORTED"]])
    metrics = compute_metrics((logits, labels))
    assert metrics["accuracy"] == 0.75
    assert "macro_f1" in metrics
    for label in LABELS:
        assert f"precision_{label}" in metrics
        assert f"recall_{label}" in metrics
        assert f"f1_{label}" in metrics


def test_real_trainer_training_loop_runs_end_to_end(tmp_path, tiny_examples, tiny_tokenizer):
    """Actually instantiates a (tiny, from-scratch, untrained) transformer
    and runs transformers.Trainer for a handful of steps, proving the
    training loop wiring in train.py's build_hf_dataset/compute_metrics
    works with the real HF Trainer API, not just that it imports."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments

    from training.train import ID2LABEL, LABEL2ID

    config = BertConfig(
        vocab_size=tiny_tokenizer.vocab_size + 5,
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64,
        num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model = BertForSequenceClassification(config)

    train_ds = build_hf_dataset(tiny_examples, tiny_tokenizer, max_length=32)
    val_ds = build_hf_dataset(tiny_examples, tiny_tokenizer, max_length=32)

    args = TrainingArguments(
        output_dir=str(tmp_path / "ckpt"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    assert train_result.training_loss is not None

    eval_metrics = trainer.evaluate()
    assert "eval_accuracy" in eval_metrics
    assert "eval_macro_f1" in eval_metrics

    out_dir = tmp_path / "saved-model"
    trainer.save_model(str(out_dir))
    tiny_tokenizer.save_pretrained(str(out_dir))
    assert (out_dir / "config.json").exists()

    # verify a saved model round-trips through the same load path
    # app/verification/groundedness.py uses (AutoModel/AutoTokenizer.from_pretrained)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    reloaded_tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
    reloaded_model = AutoModelForSequenceClassification.from_pretrained(str(out_dir))
    inputs = reloaded_tokenizer("How does Depends work? [SEP] FastAPI uses Depends.", "FastAPI uses Depends.", return_tensors="pt")
    with torch.no_grad():
        logits = reloaded_model(**inputs).logits
    assert logits.shape[-1] == 3
