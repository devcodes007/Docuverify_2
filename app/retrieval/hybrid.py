"""
Hybrid retrieval: combines BM25 (exact terminology / identifiers) and dense
(semantic) retrieval via a transparent, configurable score combination.

    hybrid_score = alpha * normalize(dense_score) + (1 - alpha) * normalize(bm25_score)

Each score list is min-max normalized to [0, 1] independently before
combining, since BM25 and cosine-similarity scores live on very different,
unbounded-vs-bounded scales and are not otherwise comparable. This is why
hybrid retrieval helps here: BM25 alone misses paraphrased/semantic queries
("how does FastAPI inject dependencies") and dense retrieval alone can miss
exact identifier matches ("HTTPException") when the identifier is rare in
the embedding model's training distribution. Combining both lets a single
query benefit from whichever signal is stronger for it.
"""
from __future__ import annotations

from app.models.schemas import Chunk, RetrievedChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import EmbeddingProvider, VectorStore


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    def __init__(self, bm25_index: BM25Index, vector_store: VectorStore, embedder: EmbeddingProvider | None):
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.embedder = embedder

    def search_bm25(self, query: str, top_k: int) -> list[RetrievedChunk]:
        results = self.bm25_index.search(query, top_k=top_k)
        return [RetrievedChunk(chunk=r.chunk, score=r.score, bm25_score=r.score) for r in results]

    def search_dense(self, query: str, top_k: int) -> list[RetrievedChunk]:
        embedding = self._embed_query(query)
        results = self.vector_store.query(embedding, top_k=top_k, query_text=query)
        return [RetrievedChunk(chunk=r.chunk, score=r.score, dense_score=r.score) for r in results]

    def search_hybrid(self, query: str, top_k: int, alpha: float = 0.6) -> list[RetrievedChunk]:
        bm25_results = self.bm25_index.search(query, top_k=max(top_k * 3, top_k))
        embedding = self._embed_query(query)
        dense_results = self.vector_store.query(embedding, top_k=max(top_k * 3, top_k), query_text=query)

        bm25_by_id = {r.chunk.chunk_id: r for r in bm25_results}
        dense_by_id = {r.chunk.chunk_id: r for r in dense_results}

        bm25_norm = _normalize({cid: r.score for cid, r in bm25_by_id.items()})
        dense_norm = _normalize({cid: r.score for cid, r in dense_by_id.items()})

        all_ids = set(bm25_by_id) | set(dense_by_id)
        combined: list[RetrievedChunk] = []
        for cid in all_ids:
            chunk: Chunk = (dense_by_id.get(cid) or bm25_by_id[cid]).chunk
            b = bm25_norm.get(cid, 0.0)
            d = dense_norm.get(cid, 0.0)
            score = alpha * d + (1 - alpha) * b
            combined.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    bm25_score=bm25_by_id[cid].score if cid in bm25_by_id else None,
                    dense_score=dense_by_id[cid].score if cid in dense_by_id else None,
                )
            )
        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[:top_k]

    def _embed_query(self, query: str) -> list[float] | None:
        if not self.vector_store.requires_embeddings:
            return None
        if self.embedder is None:
            raise ValueError("A local embedder is required for this vector store.")
        [embedding] = self.embedder.embed([query])
        return embedding
