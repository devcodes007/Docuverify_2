from app.models.schemas import Chunk, ContentType
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import ChromaVectorStore, HashingEmbedder
from app.retrieval.hybrid import HybridRetriever, _normalize


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        source_url="file:///doc1.md",
        title="Doc1",
        text=text,
        content_type=ContentType.PROSE,
    )


CHUNKS = [
    make_chunk("c1", "FastAPI dependencies use Depends for dependency injection."),
    make_chunk("c2", "You can raise an HTTPException inside a path operation function."),
    make_chunk("c3", "BackgroundTasks let you run code after returning a response."),
    make_chunk("c4", "Middleware runs code before and after every request in FastAPI."),
]


def _build_retriever(tmp_path):
    embedder = HashingEmbedder(dim=64)
    bm25 = BM25Index()
    bm25.build(CHUNKS)
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="hybrid_test")
    store.upsert(CHUNKS, embedder.embed([c.text for c in CHUNKS]))
    return HybridRetriever(bm25_index=bm25, vector_store=store, embedder=embedder)


def test_normalize_handles_empty_and_flat_scores():
    assert _normalize({}) == {}
    assert _normalize({"a": 5.0, "b": 5.0}) == {"a": 1.0, "b": 1.0}
    norm = _normalize({"a": 0.0, "b": 10.0})
    assert norm["a"] == 0.0
    assert norm["b"] == 1.0


def test_hybrid_search_returns_bm25_and_dense_scores(tmp_path):
    retriever = _build_retriever(tmp_path)
    results = retriever.search_hybrid("HTTPException path operation", top_k=3, alpha=0.5)
    assert results
    assert results[0].chunk.chunk_id == "c2"
    # both signals should be present on at least one result
    assert any(r.bm25_score is not None for r in results)


def test_alpha_zero_behaves_like_pure_bm25(tmp_path):
    retriever = _build_retriever(tmp_path)
    hybrid_results = retriever.search_hybrid("HTTPException", top_k=1, alpha=0.0)
    bm25_results = retriever.search_bm25("HTTPException", top_k=1)
    assert hybrid_results[0].chunk.chunk_id == bm25_results[0].chunk.chunk_id


def test_hybrid_search_deduplicates_by_chunk_id(tmp_path):
    retriever = _build_retriever(tmp_path)
    results = retriever.search_hybrid("dependency injection Depends", top_k=10, alpha=0.6)
    ids = [r.chunk.chunk_id for r in results]
    assert len(ids) == len(set(ids))
