from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.orchestrator import Orchestrator
from app.config import Settings, get_settings
from app.dependencies import get_groundedness_classifier, get_hybrid_retriever, get_orchestrator
from app.logging_config import get_logger, log_event
from app.models.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    QueryRequest,
    QueryResponse,
    RetrievalStrategy,
    RetrieveRequest,
    RetrieveResponse,
)

router = APIRouter(tags=["query"])
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, orchestrator: Orchestrator = Depends(get_orchestrator)) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    try:
        return orchestrator.answer(request.question, top_k=request.top_k, debug=request.debug)
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "query_failed", error=str(type(exc).__name__))
        raise HTTPException(status_code=500, detail="internal error while processing the query") from exc


@router.post("/query/stream")
def query_stream(request: QueryRequest, orchestrator: Orchestrator = Depends(get_orchestrator)):
    """Streams coarse-grained progress events followed by the final answer.
    This is not token-level LLM streaming (the LLM provider abstraction
    doesn't require streaming support from every backend) -- it streams the
    *agent's* steps as they complete, which is what's most useful for
    showing the agentic behavior live in the UI."""

    def event_stream():
        yield _sse({"event": "classification_started"})
        response = orchestrator.answer(request.question, top_k=request.top_k, debug=True)
        if response.trace:
            for step in response.trace.steps:
                yield _sse({"event": "step", "step": step.step, "detail": step.detail})
        yield _sse({"event": "final", "response": json.loads(response.model_dump_json())})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest, settings: Settings = Depends(get_settings)) -> RetrieveResponse:
    retriever = get_hybrid_retriever()
    if request.strategy == RetrievalStrategy.BM25:
        results = retriever.search_bm25(request.query, request.top_k)
    elif request.strategy == RetrievalStrategy.DENSE:
        results = retriever.search_dense(request.query, request.top_k)
    else:
        results = retriever.search_hybrid(request.query, request.top_k, alpha=settings.hybrid_alpha)
    return RetrieveResponse(results=results)


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    classifier = get_groundedness_classifier()
    result = classifier.predict(request.question, request.context, request.answer)
    return EvaluateResponse(groundedness=result)
