from __future__ import annotations

import time

from fastapi import APIRouter

from app.dependencies import get_bm25_index, get_vector_store

router = APIRouter(tags=["debug"])

_START_TIME = time.time()


@router.get("/metrics")
def metrics() -> dict:
    """Minimal Prometheus-style-friendly metrics endpoint. Kept intentionally
    simple (no external metrics infra) per the project's "don't add
    infrastructure without a concrete reason" guidance -- this is enough to
    see index size and process uptime without standing up Prometheus."""
    vector_store = get_vector_store()
    bm25_index = get_bm25_index()
    return {
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "vector_store_ready": vector_store.is_ready(),
        "bm25_ready": bm25_index.is_ready(),
        "bm25_chunk_count": len(bm25_index._chunks) if bm25_index.is_ready() else 0,
    }
