"""
Retrieval evaluation: Recall@k, Precision@k, and MRR for BM25, Dense, and
Hybrid retrieval, compared side by side, using evaluation/eval_queries.json
(relevance judged at the document level: a retrieved chunk counts as
relevant if its document_id is in the query's gold_document_ids).

Usage:
    python -m evaluation.retrieval_eval --k 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.ingestion.loader import LocalMarkdownSource
from app.ingestion.metadata import build_chunks
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import ChromaVectorStore, build_embedder
from app.retrieval.hybrid import HybridRetriever
from app.models.schemas import RetrievedChunk


def load_eval_queries(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def recall_at_k(results: list[RetrievedChunk], gold_docs: set[str]) -> float:
    if not gold_docs:
        return 1.0  # nothing to recall; not counted in aggregate averaging below
    retrieved_docs = {r.chunk.document_id for r in results}
    return len(retrieved_docs & gold_docs) / len(gold_docs)


def precision_at_k(results: list[RetrievedChunk], gold_docs: set[str]) -> float:
    if not results:
        return 0.0
    relevant = sum(1 for r in results if r.chunk.document_id in gold_docs)
    return relevant / len(results)


def reciprocal_rank(results: list[RetrievedChunk], gold_docs: set[str]) -> float:
    for i, r in enumerate(results, start=1):
        if r.chunk.document_id in gold_docs:
            return 1.0 / i
    return 0.0


def evaluate_strategy(retriever: HybridRetriever, strategy: str, queries: list[dict], k: int, alpha: float) -> dict:
    recalls, precisions, rrs = [], [], []
    for q in queries:
        gold_docs = set(q.get("gold_document_ids") or [])
        if not gold_docs:
            continue  # skip the intentionally-unanswerable query for retrieval metrics
        if strategy == "BM25":
            results = retriever.search_bm25(q["question"], top_k=k)
        elif strategy == "DENSE":
            results = retriever.search_dense(q["question"], top_k=k)
        else:
            results = retriever.search_hybrid(q["question"], top_k=k, alpha=alpha)

        recalls.append(recall_at_k(results, gold_docs))
        precisions.append(precision_at_k(results, gold_docs))
        rrs.append(reciprocal_rank(results, gold_docs))

    n = max(len(recalls), 1)
    return {
        "strategy": strategy,
        f"recall@{k}": round(sum(recalls) / n, 4),
        f"precision@{k}": round(sum(precisions) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        "queries_evaluated": len(recalls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--eval-queries", default="evaluation/eval_queries.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="evaluation/results/retrieval_eval_report.json")
    args = parser.parse_args()

    settings = get_settings()
    documents = LocalMarkdownSource(args.raw_dir).load()
    if not documents:
        raise SystemExit(f"No documents found in {args.raw_dir}")
    chunks = build_chunks(documents, settings.chunk_size_tokens, settings.chunk_overlap_tokens)

    embedder = build_embedder(settings.embedding_model, fallback_dim=settings.embedding_dim)
    bm25 = BM25Index()
    bm25.build(chunks)
    store = ChromaVectorStore(path="./data/processed/eval_chroma", collection_name="retrieval_eval")
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(bm25_index=bm25, vector_store=store, embedder=embedder)

    queries = load_eval_queries(args.eval_queries)

    report = {
        "k": args.k,
        "embedding_model": settings.embedding_model,
        "results": [
            evaluate_strategy(retriever, "BM25", queries, args.k, settings.hybrid_alpha),
            evaluate_strategy(retriever, "DENSE", queries, args.k, settings.hybrid_alpha),
            evaluate_strategy(retriever, "HYBRID", queries, args.k, settings.hybrid_alpha),
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(f"Retrieval evaluation (k={args.k}):")
    for r in report["results"]:
        print(f"  {r}")
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
