from app.models.schemas import Chunk, ContentType, RetrievedChunk
from evaluation.retrieval_eval import precision_at_k, recall_at_k, reciprocal_rank


def make_result(chunk_id: str, document_id: str, score: float) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id, document_id=document_id, source_url="file:///x.md",
        title="X", text="text", content_type=ContentType.PROSE,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def test_recall_at_k_full_and_partial():
    results = [make_result("c1", "docA", 0.9), make_result("c2", "docB", 0.8)]
    assert recall_at_k(results, {"docA", "docB"}) == 1.0
    assert recall_at_k(results, {"docA", "docC"}) == 0.5


def test_recall_at_k_empty_gold_returns_one():
    results = [make_result("c1", "docA", 0.9)]
    assert recall_at_k(results, set()) == 1.0


def test_precision_at_k():
    results = [make_result("c1", "docA", 0.9), make_result("c2", "docB", 0.8), make_result("c3", "docC", 0.7)]
    assert precision_at_k(results, {"docA"}) == 1 / 3
    assert precision_at_k([], {"docA"}) == 0.0


def test_reciprocal_rank():
    results = [make_result("c1", "docB", 0.9), make_result("c2", "docA", 0.8)]
    assert reciprocal_rank(results, {"docA"}) == 0.5
    assert reciprocal_rank(results, {"docZ"}) == 0.0
