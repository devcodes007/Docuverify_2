"""
Singleton/lazy-loaded component wiring.

Embedding models, the vector store, the BM25 index, and the groundedness
classifier are all expensive to construct (model loads, disk-backed
indices), so they're built once per process and reused across requests via
`lru_cache`, rather than reconstructed per-request. FastAPI route handlers
depend on these via `Depends(get_orchestrator)` etc.
"""
from __future__ import annotations

from functools import lru_cache

from app.agents.evidence_evaluator import DeterministicEvidenceEvaluator
from app.agents.orchestrator import Orchestrator
from app.agents.query_router import QueryRouter
from app.agents.retrieval_agent import RetrievalAgent
from app.config import get_settings
from app.generation.llm import build_llm_provider
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import ChromaVectorStore, build_embedder
from app.retrieval.hybrid import HybridRetriever
from app.verification.groundedness import build_groundedness_classifier


@lru_cache
def get_embedder(_settings_marker: int = 0):
    settings = get_settings()
    return build_embedder(settings.embedding_model, fallback_dim=settings.embedding_dim)


@lru_cache
def get_vector_store() -> ChromaVectorStore:
    settings = get_settings()
    return ChromaVectorStore(path=settings.vector_db_path, collection_name=settings.vector_db_collection)


@lru_cache
def get_bm25_index() -> BM25Index:
    settings = get_settings()
    index = BM25Index()
    try:
        index.load(settings.bm25_index_path)
    except FileNotFoundError:
        pass  # index will be empty until /ingest is called
    return index


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        bm25_index=get_bm25_index(), vector_store=get_vector_store(), embedder=get_embedder()
    )


@lru_cache
def get_evaluator() -> DeterministicEvidenceEvaluator:
    settings = get_settings()
    return DeterministicEvidenceEvaluator(sufficiency_threshold=settings.evidence_sufficiency_threshold)


@lru_cache
def get_retrieval_agent() -> RetrievalAgent:
    settings = get_settings()
    return RetrievalAgent(
        retriever=get_hybrid_retriever(),
        evaluator=get_evaluator(),
        max_attempts=settings.max_retrieval_attempts,
        hybrid_alpha=settings.hybrid_alpha,
    )


@lru_cache
def get_llm_provider():
    settings = get_settings()
    return build_llm_provider(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_base=settings.llm_api_base,
        api_key=settings.openai_api_key,
    )


@lru_cache
def get_groundedness_classifier():
    settings = get_settings()
    return build_groundedness_classifier(settings.groundedness_model, settings.groundedness_max_length)


@lru_cache
def get_orchestrator() -> Orchestrator:
    return Orchestrator(
        router=QueryRouter(),
        retrieval_agent=get_retrieval_agent(),
        llm=get_llm_provider(),
        groundedness=get_groundedness_classifier(),
        evaluator=get_evaluator(),
    )


def reset_caches() -> None:
    """Used by /ingest after rebuilding indexes, and by tests, to force
    singletons to be rebuilt against fresh data."""
    for fn in (
        get_embedder, get_vector_store, get_bm25_index, get_hybrid_retriever,
        get_evaluator, get_retrieval_agent, get_llm_provider,
        get_groundedness_classifier, get_orchestrator,
    ):
        fn.cache_clear()
