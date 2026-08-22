"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_debug, routes_health, routes_ingestion, routes_query
from app.config import get_settings
from app.logging_config import configure_logging, get_logger, log_event

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


def _auto_ingest_if_empty() -> None:
    """Render's free-tier filesystem (and most PaaS free tiers) is
    ephemeral: it does not survive a redeploy or a restart after idling.
    Rather than requiring paid persistent disk just to avoid shipping an
    empty vector database, the app rebuilds its indexes from the bundled
    data/raw/ corpus on startup whenever they're missing -- cheap for the
    bundled sample corpus, and the same mechanism works unchanged if a
    persistent disk is mounted (it just becomes a no-op after the first
    boot, since the indexes are then already present)."""
    from app.dependencies import get_bm25_index, get_vector_store, reset_caches
    from app.ingestion.loader import LocalMarkdownSource
    from app.ingestion.metadata import build_chunks
    from app.retrieval.dense import build_embedder

    if not settings.auto_ingest_on_startup:
        return

    bm25 = get_bm25_index()
    vector_store = get_vector_store()
    if bm25.is_ready() and vector_store.is_ready():
        log_event(logger, "auto_ingest_skipped", reason="indexes already populated")
        return

    documents = LocalMarkdownSource(settings.raw_data_dir).load()
    if not documents:
        log_event(logger, "auto_ingest_skipped", reason=f"no documents found in {settings.raw_data_dir}")
        return

    log_event(logger, "auto_ingest_started", documents=len(documents))
    chunks = build_chunks(documents, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    bm25.build(chunks)
    bm25.save(settings.bm25_index_path)
    embedder = build_embedder(settings.embedding_model, fallback_dim=settings.embedding_dim)
    vector_store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    reset_caches()
    log_event(logger, "auto_ingest_complete", documents=len(documents), chunks=len(chunks))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _auto_ingest_if_empty()
    except Exception as exc:  # noqa: BLE001 - never let startup ingestion crash the app
        log_event(logger, "auto_ingest_failed", error=str(exc))
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Agentic RAG system over software documentation: adaptive retrieval, "
        "evidence self-evaluation, query reformulation, and a fine-tuned "
        "groundedness classifier that verifies answers before returning them."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_ingestion.router)
app.include_router(routes_query.router)
app.include_router(routes_debug.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak internal stack traces to the client in production; log
    them server-side instead."""
    log_event(
        logger, "unhandled_exception",
        path=str(request.url.path), error_type=type(exc).__name__,
        traceback=traceback.format_exc() if settings.environment != "production" else None,
    )
    detail = str(exc) if settings.environment != "production" else "internal server error"
    return JSONResponse(status_code=500, content={"detail": detail})


@app.get("/")
def root() -> dict:
    return {"app": settings.app_name, "docs": "/docs", "health": "/health"}
