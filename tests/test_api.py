import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest
from fastapi.testclient import TestClient

from app.dependencies import reset_caches


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point every disk-backed component at a throwaway tmp dir so tests
    # never touch the real ./data directory, and force the mock LLM
    # provider so no live LLM call is required.
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25_index.pkl"))
    monkeypatch.setenv("RAW_DATA_DIR", "./data/raw")
    monkeypatch.setenv("GROUNDEDNESS_MODEL", str(tmp_path / "no-model-here"))

    from app.config import get_settings

    get_settings.cache_clear()
    reset_caches()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
    reset_caches()


def test_health_endpoint_before_ingestion(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # auto-ingest-on-startup means the index is already populated by the
    # time /health is first hit, unless AUTO_INGEST_ON_STARTUP=false
    assert body["vector_store_ready"] is True


def test_info_endpoint(client):
    resp = client.get("/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["app_name"] == "DocuVerify v2"
    assert "llm_provider" in body


def test_ingest_then_query_end_to_end(client):
    ingest_resp = client.post("/ingest", json={"source": "local"})
    assert ingest_resp.status_code == 200
    ingest_body = ingest_resp.json()
    assert ingest_body["documents_ingested"] >= 3
    assert ingest_body["chunks_created"] > 0

    health_resp = client.get("/health")
    assert health_resp.json()["vector_store_ready"] is True
    assert health_resp.json()["bm25_index_ready"] is True

    query_resp = client.post("/query", json={"question": "What does Depends do?", "top_k": 5})
    assert query_resp.status_code == 200
    body = query_resp.json()
    assert body["query_type"] == "SIMPLE_LOOKUP"
    assert body["answer"]
    assert isinstance(body["retrieval_attempts"], int)


def test_pdf_ingest_rebuilds_indexes(client, monkeypatch):
    from app.api import routes_ingestion
    from app.ingestion.loader import RawDocument

    def fake_pdf_load(self):
        return [
            RawDocument(
                document_id="pdf-upload",
                source_url="uploaded-pdf://guide.pdf",
                title="Guide",
                raw_html_or_markdown="# Guide\n\n## Page 1\n\nDepends declares reusable FastAPI dependency logic.",
                is_html=False,
            )
        ]

    monkeypatch.setattr(routes_ingestion.UploadedPdfSource, "load", fake_pdf_load)

    ingest_resp = client.post(
        "/ingest/pdf",
        files={"file": ("guide.pdf", b"%PDF-pretend", "application/pdf")},
    )
    assert ingest_resp.status_code == 200
    ingest_body = ingest_resp.json()
    assert ingest_body["documents_ingested"] == 1
    assert ingest_body["chunks_created"] > 0

    query_resp = client.post("/query", json={"question": "What does Depends do?", "top_k": 5})
    assert query_resp.status_code == 200
    assert query_resp.json()["sources"][0]["source_url"] == "uploaded-pdf://guide.pdf"


def test_pdf_ingest_rejects_non_pdf_file(client):
    resp = client.post(
        "/ingest/pdf",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert resp.status_code == 400


def test_query_debug_returns_trace(client):
    client.post("/ingest", json={"source": "local"})
    resp = client.post(
        "/query",
        json={"question": "What is the difference between Depends and middleware?", "debug": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_type"] == "COMPARISON"
    assert body["trace"] is not None
    assert len(body["trace"]["steps"]) > 0


def test_multi_hop_query_end_to_end(client):
    client.post("/ingest", json={"source": "local"})
    resp = client.post(
        "/query",
        json={
            "question": "How does FastAPI dependency injection interact with request validation?",
            "debug": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_type"] == "MULTI_HOP"
    assert len(body["trace"]["subqueries"]) >= 2


def test_malformed_query_request_rejected(client):
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code in (422, 400)


def test_query_missing_field_returns_422(client):
    resp = client.post("/query", json={"top_k": 5})
    assert resp.status_code == 422


def test_retrieve_endpoint_before_ingestion_returns_empty(tmp_path, monkeypatch):
    # Standalone (not nested in the `client` fixture) so it gets its own
    # untouched paths and can verify the genuinely-empty-index behavior
    # with auto-ingest explicitly disabled.
    import importlib
    import sys

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("AUTO_INGEST_ON_STARTUP", "false")
    monkeypatch.setenv("RAW_DATA_DIR", str(tmp_path / "no-docs-here"))
    monkeypatch.setenv("VECTOR_DB_PATH", str(tmp_path / "chroma_isolated"))
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25_isolated.pkl"))
    monkeypatch.setenv("GROUNDEDNESS_MODEL", str(tmp_path / "no-model-here"))

    from app.config import get_settings
    from app.dependencies import reset_caches
    get_settings.cache_clear()
    reset_caches()

    sys.modules.pop("app.main", None)  # force settings = get_settings() to re-run on import
    import app.main as fresh_main_module
    importlib.reload(fresh_main_module)

    from fastapi.testclient import TestClient
    with TestClient(fresh_main_module.app) as fresh_client:
        resp = fresh_client.post("/retrieve", json={"query": "Depends", "strategy": "HYBRID"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    get_settings.cache_clear()
    reset_caches()
    sys.modules.pop("app.main", None)


def test_retrieve_endpoint_after_ingestion(client):
    client.post("/ingest", json={"source": "local"})
    resp = client.post("/retrieve", json={"query": "HTTPException", "strategy": "BM25", "top_k": 3})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) > 0


def test_evaluate_endpoint(client):
    resp = client.post(
        "/evaluate",
        json={
            "question": "What does Depends do?",
            "context": "FastAPI dependencies use Depends to declare reusable logic.",
            "answer": "FastAPI dependencies use Depends to declare reusable logic.",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["groundedness"]["label"] in {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "uptime_seconds" in resp.json()


def test_ingest_rejects_unknown_source(client):
    resp = client.post("/ingest", json={"source": "not-a-real-source"})
    assert resp.status_code == 400


def test_root_endpoint(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "docs" in resp.json()
