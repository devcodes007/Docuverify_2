"""
Dense retrieval.

Two things are deliberately kept behind narrow interfaces so they can be
swapped without touching any other module:

  * EmbeddingProvider  -- SentenceTransformerEmbedder (real, requires the
    model weights, see README for the network caveat in this sandbox) and
    HashingEmbedder (a deterministic, dependency-free fallback used for
    tests/offline dev so the rest of the pipeline is exercisable without a
    model download).

  * VectorStore -- ChromaVectorStore is the default; swapping to
    Qdrant/Pinecone/Weaviate means implementing the same three methods
    (upsert, query, is_ready) against their client.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.models.schemas import Chunk


# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    """Wraps a Hugging Face sentence-transformers model. Requires network
    access to huggingface.co (or a locally cached model dir) the first time
    a given model name is used -- see README known limitations."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # local import: optional heavy dep

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class HashingEmbedder:
    """Deterministic, dependency-free embedding fallback.

    Not semantically meaningful the way a trained model is -- it's a bag-of-
    tokens hashed into a fixed-size vector (a "feature hashing" / hashing
    trick embedding) -- but it is stable, fast, needs no network access, and
    is enough to exercise upsert/query/hybrid-scoring end to end in tests
    and offline development. Swap to SentenceTransformerEmbedder for real
    semantic retrieval quality.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def build_embedder(model_name: str, fallback_dim: int = 384) -> EmbeddingProvider:
    """Tries the real sentence-transformers model; falls back to the hashing
    embedder (with a warning) if the model can't be loaded (e.g. no network
    access to download weights). This keeps the app runnable end-to-end in
    restricted environments while making the real embedder the default for
    normal deployments."""
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, this is a capability probe
        from app.logging_config import get_logger

        get_logger(__name__).warning(
            "falling back to HashingEmbedder: could not load %s (%s)", model_name, exc
        )
        return HashingEmbedder(dim=fallback_dim)


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

@dataclass
class DenseResult:
    chunk: Chunk
    score: float  # cosine similarity, higher is better


class VectorStore(Protocol):
    requires_embeddings: bool

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None: ...
    def replace(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None: ...
    def query(
        self,
        embedding: list[float] | None,
        top_k: int,
        query_text: str | None = None,
    ) -> list[DenseResult]: ...
    def is_ready(self) -> bool: ...


class ChromaVectorStore:
    requires_embeddings = True

    def __init__(self, path: str | Path, collection_name: str):
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None:
        if not chunks:
            return
        if embeddings is None:
            raise ValueError("ChromaVectorStore requires precomputed embeddings.")
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "source_url": c.source_url,
                    "title": c.title,
                    "section": c.section,
                    "subsection": c.subsection,
                    "heading_path": " > ".join(c.heading_path),
                    "content_type": c.content_type.value,
                }
                for c in chunks
            ],
        )

    def replace(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None:
        existing = self._collection.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        self.upsert(chunks, embeddings)

    def query(
        self,
        embedding: list[float] | None,
        top_k: int,
        query_text: str | None = None,
    ) -> list[DenseResult]:
        if embedding is None:
            raise ValueError("ChromaVectorStore requires a query embedding.")
        if self._collection.count() == 0:
            return []
        top_k = min(top_k, self._collection.count())
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        out: list[DenseResult] = []
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        distances = result["distances"][0]  # cosine distance, lower is better
        for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            similarity = 1.0 - dist
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=meta["document_id"],
                source_url=meta["source_url"],
                title=meta["title"],
                section=meta.get("section", ""),
                subsection=meta.get("subsection", ""),
                heading_path=[p for p in meta.get("heading_path", "").split(" > ") if p],
                content_type=meta.get("content_type", "prose"),
                text=doc_text,
            )
            out.append(DenseResult(chunk=chunk, score=similarity))
        return out

    def is_ready(self) -> bool:
        try:
            return self._collection.count() > 0
        except Exception:  # noqa: BLE001
            return False


class QdrantVectorStore:
    """Qdrant-backed vector store.

    With ``cloud_inference=True``, embeddings are generated by Qdrant Cloud
    from text inference objects, so the FastAPI process does not need to load
    sentence-transformers for dense search.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection_name: str,
        vector_size: int,
        inference_model: str,
        cloud_inference: bool = False,
        client: Any | None = None,
    ):
        if not url and client is None:
            raise ValueError("QDRANT_URL is required when VECTOR_STORE_BACKEND=qdrant.")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.inference_model = inference_model
        self.cloud_inference = cloud_inference
        self.requires_embeddings = not cloud_inference

        if client is not None:
            self._client = client
            self._models = None
        else:
            from qdrant_client import QdrantClient, models

            self._client = QdrantClient(url=url, api_key=api_key, cloud_inference=cloud_inference)
            self._models = models

        self._ensure_collection()

    def _ensure_collection(self, recreate: bool = False) -> None:
        models = self._models
        if models is None:
            return

        exists = self._client.collection_exists(self.collection_name)
        if exists and recreate:
            self._client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None:
        if not chunks:
            return
        if self.requires_embeddings and embeddings is None:
            raise ValueError("QdrantVectorStore requires embeddings unless QDRANT_CLOUD_INFERENCE=true.")
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("Embeddings count must match chunks count.")

        points = []
        for index, chunk in enumerate(chunks):
            vector = embeddings[index] if embeddings is not None else self._document_vector(chunk.text)
            points.append(
                self._point_struct(
                    point_id=_stable_point_id(chunk.chunk_id),
                    vector=vector,
                    payload=_chunk_payload(chunk),
                )
            )
        self._client.upsert(collection_name=self.collection_name, points=points)

    def replace(self, chunks: list[Chunk], embeddings: list[list[float]] | None = None) -> None:
        self._ensure_collection(recreate=True)
        self.upsert(chunks, embeddings)

    def query(
        self,
        embedding: list[float] | None,
        top_k: int,
        query_text: str | None = None,
    ) -> list[DenseResult]:
        if self.requires_embeddings:
            if embedding is None:
                raise ValueError("QdrantVectorStore requires a query embedding.")
            query = embedding
        else:
            if not query_text:
                raise ValueError("query_text is required when QDRANT_CLOUD_INFERENCE=true.")
            query = self._document_vector(query_text)

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query,
            with_payload=True,
            limit=top_k,
        )
        points = getattr(response, "points", response)
        return [_dense_result_from_qdrant_point(point) for point in points]

    def is_ready(self) -> bool:
        try:
            if self._models is not None and not self._client.collection_exists(self.collection_name):
                return False
            count = self._client.count(collection_name=self.collection_name, exact=False)
            return getattr(count, "count", 0) > 0
        except Exception:  # noqa: BLE001
            return False

    def _document_vector(self, text: str):
        if self._models is None:
            return {"text": text, "model": self.inference_model}
        return self._models.Document(text=text, model=self.inference_model)

    def _point_struct(self, point_id: str, vector, payload: dict):
        if self._models is None:
            return {"id": point_id, "vector": vector, "payload": payload}
        return self._models.PointStruct(id=point_id, vector=vector, payload=payload)


def _stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _chunk_payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_url": chunk.source_url,
        "title": chunk.title,
        "section": chunk.section,
        "subsection": chunk.subsection,
        "heading_path": chunk.heading_path,
        "content_type": chunk.content_type.value,
        "content": chunk.text,
    }


def _dense_result_from_qdrant_point(point) -> DenseResult:
    payload = getattr(point, "payload", None) or point.get("payload", {})
    score = getattr(point, "score", None)
    if score is None:
        score = point.get("score", 0.0)
    chunk = Chunk(
        chunk_id=payload["chunk_id"],
        document_id=payload["document_id"],
        source_url=payload["source_url"],
        title=payload["title"],
        section=payload.get("section", ""),
        subsection=payload.get("subsection", ""),
        heading_path=payload.get("heading_path", []),
        content_type=payload.get("content_type", "prose"),
        text=payload.get("content", ""),
    )
    return DenseResult(chunk=chunk, score=float(score))
