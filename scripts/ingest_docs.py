"""
Convenience CLI wrapper: ingests data/raw and builds both the BM25 and
dense indexes, without needing the FastAPI server running (this calls the
same ingestion/metadata/retrieval modules the /ingest endpoint uses).

Usage:
    python scripts/ingest_docs.py --raw-dir data/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.ingestion.loader import LocalMarkdownSource
from app.ingestion.metadata import build_chunks
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import ChromaVectorStore, QdrantVectorStore, build_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=None, help="defaults to RAW_DATA_DIR from settings")
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = args.raw_dir or settings.raw_data_dir

    print(f"Loading documents from {raw_dir} ...")
    documents = LocalMarkdownSource(raw_dir).load()
    if not documents:
        raise SystemExit(f"No documents found in {raw_dir}")
    print(f"Loaded {len(documents)} documents.")

    chunks = build_chunks(documents, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    print(f"Produced {len(chunks)} chunks (chunk_size={settings.chunk_size_tokens} tokens, "
          f"overlap={settings.chunk_overlap_tokens}).")

    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(settings.bm25_index_path)
    print(f"BM25 index saved to {settings.bm25_index_path}.")

    if settings.vector_store_backend == "qdrant":
        store = QdrantVectorStore(
            url=settings.qdrant_url or "",
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
            vector_size=settings.embedding_dim,
            inference_model=settings.qdrant_inference_model,
            cloud_inference=settings.qdrant_cloud_inference,
        )
        embeddings = None
        if store.requires_embeddings:
            print(f"Loading embedding model: {settings.embedding_model} ...")
            embedder = build_embedder(settings.embedding_model, fallback_dim=settings.embedding_dim)
            embeddings = embedder.embed([c.text for c in chunks])
        store.replace(chunks, embeddings)
        print(f"Dense index written to Qdrant collection={settings.qdrant_collection}.")
    else:
        print(f"Loading embedding model: {settings.embedding_model} ...")
        embedder = build_embedder(settings.embedding_model, fallback_dim=settings.embedding_dim)
        store = ChromaVectorStore(path=settings.vector_db_path, collection_name=settings.vector_db_collection)
        embeddings = embedder.embed([c.text for c in chunks])
        store.replace(chunks, embeddings)
        print(f"Dense index written to {settings.vector_db_path} (collection={settings.vector_db_collection}).")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
