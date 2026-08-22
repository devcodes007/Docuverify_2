from training.generate_dataset import _negate, generate_template_examples
from training.prepare_dataset import split_by_document
from app.models.schemas import Chunk, ContentType


def make_chunk(chunk_id: str, document_id: str, text: str, heading_path=None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id=document_id, source_url=f"file:///{document_id}.md",
        title=document_id, heading_path=heading_path or [document_id],
        text=text, content_type=ContentType.PROSE,
    )


def test_negate_inserts_negation_for_known_patterns():
    assert _negate("FastAPI validates the request body automatically.") is not None
    assert "does not validate" in _negate("FastAPI validates the request body automatically.")


def test_negate_returns_none_when_no_pattern_matches():
    assert _negate("The sky is blue today.") is None


def test_generate_template_examples_produces_all_three_labels():
    chunks = [
        make_chunk("c1", "doc1", "FastAPI validates the request body automatically using Pydantic models for every request."),
        make_chunk("c2", "doc2", "FastAPI resolves dependencies automatically before running the path operation."),
    ]
    examples = generate_template_examples(chunks)
    labels = {ex.label for ex in examples}
    assert "SUPPORTED" in labels
    assert "UNSUPPORTED" in labels
    assert "CONTRADICTED" in labels


def test_generate_template_unsupported_uses_different_context_than_question_chunk():
    chunks = [
        make_chunk("c1", "doc1", "FastAPI validates the request body automatically using Pydantic models for every request."),
        make_chunk("c2", "doc2", "Middleware runs before and after every request in the application stack."),
    ]
    examples = generate_template_examples(chunks)
    unsupported = [e for e in examples if e.label == "UNSUPPORTED"]
    assert unsupported
    for ex in unsupported:
        assert ex.context != chunks[0].text or ex.document_id != "doc1"


# ---------------------------------------------------------------------------
# split_by_document: leakage safety across a range of document counts
# ---------------------------------------------------------------------------

def _fake_examples(doc_count: int, per_doc: int = 3) -> list[dict]:
    examples = []
    for d in range(doc_count):
        for i in range(per_doc):
            examples.append({"question": f"q{d}-{i}", "context": "c", "answer": "a", "label": "SUPPORTED", "document_id": f"doc{d}"})
    return examples


def test_split_no_leakage_across_various_doc_counts():
    for doc_count in (1, 2, 3, 4, 5, 10, 20):
        examples = _fake_examples(doc_count)
        train, val, test = split_by_document(examples)
        train_docs = {e["document_id"] for e in train}
        val_docs = {e["document_id"] for e in val}
        test_docs = {e["document_id"] for e in test}
        assert not (train_docs & val_docs), f"leakage at doc_count={doc_count}"
        assert not (train_docs & test_docs), f"leakage at doc_count={doc_count}"
        assert not (val_docs & test_docs), f"leakage at doc_count={doc_count}"
        # every example must land in exactly one split
        assert len(train) + len(val) + len(test) == len(examples)


def test_split_with_many_documents_is_roughly_proportional():
    examples = _fake_examples(doc_count=20, per_doc=5)
    train, val, test = split_by_document(examples, train_frac=0.7, val_frac=0.15)
    total = len(examples)
    assert 0.5 < len(train) / total < 0.85
    assert len(val) > 0
    assert len(test) > 0
