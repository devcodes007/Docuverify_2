"""
Compares the baseline single-shot RAG pipeline against the full DocuVerify
agentic pipeline on evaluation/eval_queries.json, reporting per-query and
aggregate:

  * context_recall           -- did retrieval surface all gold documents
  * answer_correctness proxy -- expected-keyword coverage in the answer
  * refusal behavior         -- baseline never refuses (it has no
    groundedness check); DocuVerify should refuse on the intentionally
    unanswerable query rather than hallucinate.

Usage:
    python -m evaluation.benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.generation.llm import build_llm_provider
from app.ingestion.loader import LocalMarkdownSource
from app.ingestion.metadata import build_chunks
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import ChromaVectorStore, build_embedder
from app.retrieval.hybrid import HybridRetriever
from evaluation.baseline_rag import BaselineRAG
from evaluation.rag_eval import build_orchestrator_for_eval, evaluate_query
from evaluation.retrieval_eval import load_eval_queries


def evaluate_baseline_query(baseline: BaselineRAG, q: dict) -> dict:
    response = baseline.answer(q["question"])
    gold_docs = set(q.get("gold_document_ids") or [])
    context_recall = (
        len(response.sources_document_ids & gold_docs) / len(gold_docs) if gold_docs else None
    )
    expected_keywords = q.get("expected_keywords") or []
    answer_lower = response.answer.lower()
    hit_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    answer_correctness = len(hit_keywords) / len(expected_keywords) if expected_keywords else None
    return {
        "question": q["question"],
        "context_recall": round(context_recall, 4) if context_recall is not None else None,
        "answer_correctness": round(answer_correctness, 4) if answer_correctness is not None else None,
        "refused": False,  # the baseline has no verification step and can never refuse
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--eval-queries", default="evaluation/eval_queries.json")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out", default="evaluation/results/baseline_vs_agentic_report.json")
    args = parser.parse_args()

    settings = get_settings()
    documents = LocalMarkdownSource(args.raw_dir).load()
    chunks = build_chunks(documents, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    embedder = build_embedder(settings.embedding_model, fallback_dim=settings.embedding_dim)
    bm25 = BM25Index()
    bm25.build(chunks)
    store = ChromaVectorStore(path="./data/processed/benchmark_chroma", collection_name="benchmark")
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(bm25_index=bm25, vector_store=store, embedder=embedder)
    llm = build_llm_provider(settings.llm_provider, settings.llm_model, settings.llm_api_base, settings.openai_api_key)

    baseline = BaselineRAG(retriever=retriever, llm=llm, top_k=args.top_k)
    orchestrator = build_orchestrator_for_eval(args.raw_dir, "./data/processed/benchmark_orchestrator_chroma")

    queries = load_eval_queries(args.eval_queries)
    baseline_results = [evaluate_baseline_query(baseline, q) for q in queries]
    agentic_results = [evaluate_query(orchestrator, q, args.top_k) for q in queries]

    def _avg(rows: list[dict], key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    unanswerable_idx = [i for i, q in enumerate(queries) if not q.get("gold_document_ids")]

    summary = {
        "baseline": {
            "avg_context_recall": _avg(baseline_results, "context_recall"),
            "avg_answer_correctness": _avg(baseline_results, "answer_correctness"),
            "refused_on_unanswerable": any(baseline_results[i]["refused"] for i in unanswerable_idx),
        },
        "docuverify_agentic": {
            "avg_context_recall": _avg(agentic_results, "context_recall"),
            "avg_answer_correctness": _avg(agentic_results, "answer_correctness"),
            "refused_on_unanswerable": any(agentic_results[i]["refused"] for i in unanswerable_idx),
            "avg_retrieval_attempts": _avg(agentic_results, "retrieval_attempts"),
        },
    }

    report = {"summary": summary, "baseline_per_query": baseline_results, "agentic_per_query": agentic_results}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("Baseline vs DocuVerify agentic comparison:")
    print(json.dumps(summary, indent=2))
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
