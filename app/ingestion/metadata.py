"""Attaches document/source/section metadata to raw chunk drafts, producing
final `Chunk` objects. This is the glue between loader -> cleaner -> chunker
and the rest of the app (retrieval, generation, verification all consume
`Chunk` objects only)."""
from __future__ import annotations

from app.ingestion.chunker import ChunkDraft, chunk_document
from app.ingestion.cleaner import clean
from app.ingestion.loader import RawDocument
from app.models.schemas import Chunk


def build_chunks_for_document(
    doc: RawDocument,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> list[Chunk]:
    blocks = clean(doc.raw_html_or_markdown, doc.is_html)
    drafts: list[ChunkDraft] = chunk_document(
        blocks, chunk_size_tokens=chunk_size_tokens, chunk_overlap_tokens=chunk_overlap_tokens
    )

    chunks: list[Chunk] = []
    for i, draft in enumerate(drafts):
        section = draft.heading_path[0] if draft.heading_path else ""
        subsection = draft.heading_path[1] if len(draft.heading_path) > 1 else ""
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}::chunk-{i}",
                document_id=doc.document_id,
                source_url=doc.source_url,
                title=doc.title,
                section=section,
                subsection=subsection,
                heading_path=draft.heading_path,
                content_type=draft.content_type,
                text=draft.text,
            )
        )
    return chunks


def build_chunks(
    documents: list[RawDocument],
    chunk_size_tokens: int = 350,
    chunk_overlap_tokens: int = 50,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in documents:
        all_chunks.extend(
            build_chunks_for_document(doc, chunk_size_tokens, chunk_overlap_tokens)
        )
    return all_chunks
