from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import Settings, get_settings
from app.dependencies import get_bm25_index, get_vector_store, reset_caches
from app.ingestion.loader import LocalMarkdownSource, PdfExtractionError, UploadedPdfSource
from app.ingestion.metadata import build_chunks
from app.logging_config import get_logger, log_event
from app.models.schemas import Chunk, IngestRequest, IngestResponse
from app.retrieval.dense import build_embedder

router = APIRouter(tags=["ingestion"])
logger = get_logger(__name__)

MAX_PDF_UPLOAD_BYTES = 15 * 1024 * 1024


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, settings: Settings = Depends(get_settings)) -> IngestResponse:
    if request.source != "local":
        raise HTTPException(
            status_code=400,
            detail="Only the 'local' source is wired into this endpoint by default; "
            "see app/ingestion/loader.py to register a WebDocSource with an allowed-domain list.",
        )

    source = LocalMarkdownSource(settings.raw_data_dir)
    documents = source.load()
    if not documents:
        raise HTTPException(status_code=404, detail=f"No documents found in {settings.raw_data_dir}")

    response = _rebuild_indexes(documents, settings)
    log_event(logger, "ingestion_complete", documents=response.documents_ingested, chunks=response.chunks_created)
    return response


@router.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
    if len(content) > MAX_PDF_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF upload is too large. Maximum size is 15 MB.")

    try:
        documents = UploadedPdfSource(filename, content).load()
    except PdfExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not documents:
        raise HTTPException(
            status_code=422,
            detail="No extractable text was found in this PDF. Scanned PDFs need OCR before ingestion.",
        )

    response = _rebuild_indexes(documents, settings)
    log_event(
        logger,
        "pdf_ingestion_complete",
        filename=filename,
        documents=response.documents_ingested,
        chunks=response.chunks_created,
    )
    return response


def _rebuild_indexes(documents, settings: Settings) -> IngestResponse:
    chunks: list[Chunk] = build_chunks(
        documents,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )

    bm25_index = get_bm25_index()
    bm25_index.build(chunks)
    bm25_index.save(settings.bm25_index_path)

    vector_store = get_vector_store()
    embeddings = None
    if vector_store.requires_embeddings:
        embedder = build_embedder(settings.embedding_model, fallback_dim=settings.embedding_dim)
        embeddings = embedder.embed([c.text for c in chunks])
    vector_store.replace(chunks, embeddings)

    reset_caches()
    return IngestResponse(documents_ingested=len(documents), chunks_created=len(chunks))


@router.get("/documents")
def list_documents(settings: Settings = Depends(get_settings)) -> dict:
    source = LocalMarkdownSource(settings.raw_data_dir)
    documents = source.load()
    return {
        "count": len(documents),
        "documents": [
            {"document_id": d.document_id, "title": d.title, "source_url": d.source_url}
            for d in documents
        ],
    }
