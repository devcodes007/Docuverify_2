"""
RAG-level evaluation.

Measures, per query in evaluation/eval_queries.json, using the live
Orchestrator:

  * context_relevance -- fraction of retrieved chunks whose document_id is
    in gold_document_ids (precision of what was retrieved).
  * context_recall     -- fraction of gold documents that appear anywhere
    in the retrieved set (did retrieval find everything it needed to).
  * answer_correctness -- fraction of the query's expected_keywords that
    appear in the generated answer (a cheap, dependency-free proxy for
    "did the answer actually address the question"; a real deployment
    would supplement this with human judgement or an external LLM-judge
    on top of, not instead of, the groundedness classifier).
  * groundedness       -- the label returned by the orchestrator's own
    groundedness classifier for that answer.
  * refused            -- whether the system refused rather than answering
    (expected to be True for the intentionally-unanswerable eval query).

Usage:
    python -m evaluation.rag_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.evidence_evaluator import DeterministicEvidenceEvaluator
from app.agents.orchestrator import Orchestrator
from app.agents.query_router import QueryRouter
from app.agents.retrieval_agent import RetrievalAgent
from app.config import get_settings
from app.generation.llm import build_llm_provider
from app.ingestion.loader import LocalMarkdownSource
from app.ingestion.metadata import build_chunks
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import ChromaVectorStore, build_embedder
from app.retrieval.hybrid import HybridRetriever
from app.verification.groundedness import build_groundedness_classifier
from evaluation.retrieval_eval import load_eval_queries


def build_orchestrator_for_eval(raw_dir: str, chroma_path: str):
    settings = get_settings()
    documents = LocalMarkdownSource(raw_dir).load()
    chunks = build_chunks(documents, settings.chunk_size_tokens, settings.chunk_overlap_tokens)

    embedder = build_embedder(settings.embedding_model, fallback_dim=settings.embedding_dim)
    bm25 = BM25Index()
    bm25.build(chunks)
    store = ChromaVectorStore(path=chroma_path, collection_name="rag_eval")
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(bm25_index=bm25, vector_store=store, embedder=embedder)

    evaluator = DeterministicEvidenceEvaluator(sufficiency_threshold=settings.evidence_sufficiency_threshold)
    agent = RetrievalAgent(
        retriever=retriever, evaluator=evaluator,
        max_attempts=settings.max_retrieval_attempts, hybrid_alpha=settings.hybrid_alpha,
    )
    llm = build_llm_provider(settings.llm_provider, settings.llm_model, settings.llm_api_base, settings.openai_api_key)
    groundedness = build_groundedness_classifier(settings.groundedness_model, settings.groundedness_max_length)
    return Orchestrator(
        router=QueryRouter(), retrieval_agent=agent, llm=llm, groundedness=groundedness, evaluator=evaluator,
    )


def evaluate_query(orchestrator: Orchestrator, q: dict, top_k: int) -> dict:
    response = orchestrator.answer(q["question"], top_k=top_k, debug=True)
    gold_docs = set(q.get("gold_document_ids") or [])
    retrieved_docs = {src.chunk_id.split("::")[0] for src in response.sources}

    context_relevance = (
        len(retrieved_docs & gold_docs) / len(retrieved_docs) if retrieved_docs and gold_docs else 0.0
    )
    context_recall = len(retrieved_docs & gold_docs) / len(gold_docs) if gold_docs else None

    expected_keywords = q.get("expected_keywords") or []
    answer_lower = response.answer.lower()
    hit_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    answer_correctness = len(hit_keywords) / len(expected_keywords) if expected_keywords else None

    return {
        "question": q["question"],
        "query_type": response.query_type.value,
        "retrieval_strategy": response.retrieval_strategy.value,
        "retrieval_attempts": response.retrieval_attempts,
        "context_relevance": round(context_relevance, 4),
        "context_recall": round(context_recall, 4) if context_recall is not None else None,
        "answer_correctness": round(answer_correctness, 4) if answer_correctness is not None else None,
        "groundedness_label": response.groundedness.label.value,
        "groundedness_confidence": response.groundedness.confidence,
        "refused": response.refused,
        "latency_ms": response.latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--eval-queries", default="evaluation/eval_queries.json")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out", default="evaluation/results/rag_eval_report.json")
    args = parser.parse_args()

    orchestrator = build_orchestrator_for_eval(args.raw_dir, "./data/processed/rag_eval_chroma")
    queries = load_eval_queries(args.eval_queries)

    per_query = [evaluate_query(orchestrator, q, args.top_k) for q in queries]

    def _avg(key: str) -> float | None:
        vals = [r[key] for r in per_query if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "avg_context_relevance": _avg("context_relevance"),
        "avg_context_recall": _avg("context_recall"),
        "avg_answer_correctness": _avg("answer_correctness"),
        "avg_latency_ms": _avg("latency_ms"),
        "refusal_rate_on_unanswerable": sum(
            1 for r in per_query if r["refused"] and not (
                (queries[per_query.index(r)].get("gold_document_ids") or [])
            )
        ),
    }

    report = {"summary": summary, "per_query": per_query}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print("RAG evaluation summary:")
    print(json.dumps(summary, indent=2))
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
