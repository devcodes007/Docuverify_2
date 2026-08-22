from app.models.schemas import Chunk, ContentType
from app.retrieval.bm25 import BM25Index, tokenize


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
    make_chunk("c1", "FastAPI dependencies use the Depends function for dependency injection."),
    make_chunk("c2", "You can raise an HTTPException from inside a path operation function."),
    make_chunk("c3", "BackgroundTasks let you run code after returning a response."),
    make_chunk("c4", "APIRouter groups related path operations into a single router object."),
]


def test_tokenize_keeps_identifiers_and_splits_camelcase():
    tokens = tokenize("Use HTTPException and BackgroundTasks")
    assert "httpexception" in tokens
    assert "background" in tokens
    assert "tasks" in tokens


def test_bm25_exact_identifier_match_ranks_first():
    index = BM25Index()
    index.build(CHUNKS)
    results = index.search("HTTPException", top_k=4)
    assert results, "expected at least one result"
    assert results[0].chunk.chunk_id == "c2"


def test_bm25_returns_empty_for_unbuilt_index():
    index = BM25Index()
    assert index.search("Depends") == []
    assert index.is_ready() is False


def test_bm25_ready_after_build():
    index = BM25Index()
    index.build(CHUNKS)
    assert index.is_ready() is True


def test_bm25_save_and_load_roundtrip(tmp_path):
    index = BM25Index()
    index.build(CHUNKS)
    path = tmp_path / "bm25.pkl"
    index.save(path)

    loaded = BM25Index()
    loaded.load(path)
    results = loaded.search("APIRouter", top_k=1)
    assert results[0].chunk.chunk_id == "c4"
