"""
Optional reranking stage.

No-op by default (identity passthrough) -- hybrid retrieval's score
combination already does most of the work needed for this corpus size.
This module exists as an explicit extension point: a cross-encoder reranker
can be dropped in by implementing `Reranker.rerank` without touching the
retrieval agent or API layer.
"""
from __future__ import annotations

from typing import Protocol

from app.models.schemas import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ...


class NoOpReranker:
    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return results[:top_k]
