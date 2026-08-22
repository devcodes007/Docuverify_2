from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.dependencies import get_bm25_index, get_groundedness_classifier, get_vector_store
from app.models.schemas import HealthResponse, InfoResponse
from app.verification.groundedness import TransformerGroundednessClassifier

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    vector_store = get_vector_store()
    bm25_index = get_bm25_index()
    groundedness = get_groundedness_classifier()
    return HealthResponse(
        status="ok",
        vector_store_ready=vector_store.is_ready(),
        bm25_index_ready=bm25_index.is_ready(),
        groundedness_model_ready=isinstance(groundedness, TransformerGroundednessClassifier),
    )


@router.get("/info", response_model=InfoResponse)
def info(settings: Settings = Depends(get_settings)) -> InfoResponse:
    return InfoResponse(
        app_name=settings.app_name,
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
        max_retrieval_attempts=settings.max_retrieval_attempts,
        hybrid_alpha=settings.hybrid_alpha,
    )
