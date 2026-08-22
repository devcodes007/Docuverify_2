from app.models.schemas import Chunk, ContentType
from app.retrieval.dense import ChromaVectorStore, HashingEmbedder


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
    make_chunk("c2", "You can raise an HTTPException inside a path operation."),
    make_chunk("c3", "BackgroundTasks let you run code after returning a response."),
]


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dim=64)
    v1 = embedder.embed(["dependency injection"])[0]
    v2 = embedder.embed(["dependency injection"])[0]
    assert v1 == v2


def test_hashing_embedder_similar_text_more_similar_than_dissimilar():
    embedder = HashingEmbedder(dim=128)
    a, b, c = embedder.embed(
        [
            "FastAPI dependency injection with Depends",
            "dependency injection in FastAPI using Depends",
            "BackgroundTasks run after the response is sent",
        ]
    )

    def cosine(x, y):
        return sum(xi * yi for xi, yi in zip(x, y))

    assert cosine(a, b) > cosine(a, c)


def test_chroma_vector_store_upsert_and_query(tmp_path):
    embedder = HashingEmbedder(dim=64)
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="test_collection")
    embeddings = embedder.embed([c.text for c in CHUNKS])
    store.upsert(CHUNKS, embeddings)

    assert store.is_ready() is True

    [query_emb] = embedder.embed(["FastAPI dependencies Depends dependency injection"])
    results = store.query(query_emb, top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id in {"c1", "c2", "c3"}
    # top result should be the most textually similar chunk about dependencies
    # (query shares the most raw tokens with c1's text)
    assert results[0].chunk.chunk_id == "c1"


def test_chroma_vector_store_empty_returns_no_results(tmp_path):
    store = ChromaVectorStore(path=tmp_path / "chroma_empty", collection_name="empty")
    assert store.is_ready() is False
    embedder = HashingEmbedder(dim=32)
    [q] = embedder.embed(["anything"])
    assert store.query(q, top_k=5) == []
