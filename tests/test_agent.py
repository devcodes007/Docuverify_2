from app.agents.evidence_evaluator import DeterministicEvidenceEvaluator
from app.agents.query_rewriter import reformulate
from app.agents.retrieval_agent import RetrievalAgent, decompose_multi_hop, _dedupe_by_chunk_id
from app.models.schemas import Chunk, ContentType, EvidenceEvaluation, QueryType, RetrievedChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import ChromaVectorStore, HashingEmbedder
from app.retrieval.hybrid import HybridRetriever


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc1", source_url="file:///doc1.md",
        title="Doc1", text=text, content_type=ContentType.PROSE,
    )


# ---------------------------------------------------------------------------
# Evidence evaluator
# ---------------------------------------------------------------------------

def test_no_results_is_insufficient():
    evaluator = DeterministicEvidenceEvaluator()
    result = evaluator.evaluate("How does Depends work?", [])
    assert result.sufficient is False
    assert result.confidence == 0.0


def test_high_score_full_coverage_is_sufficient():
    evaluator = DeterministicEvidenceEvaluator(sufficiency_threshold=0.5)
    chunk = make_chunk("c1", "FastAPI dependency injection uses Depends to declare dependencies.")
    results = [RetrievedChunk(chunk=chunk, score=0.95, dense_score=0.95, bm25_score=8.0)]
    evaluation = evaluator.evaluate("How does dependency injection work with Depends?", results)
    assert evaluation.sufficient is True
    assert evaluation.confidence > 0.5


def test_pdf_summary_instruction_words_do_not_block_sufficiency():
    evaluator = DeterministicEvidenceEvaluator(sufficiency_threshold=0.55)
    chunk = make_chunk("c1", "Problem 1.3 discusses probability distributions and expected value.")
    results = [RetrievedChunk(chunk=chunk, score=0.75, dense_score=0.75)]
    evaluation = evaluator.evaluate("give me summary of problem 1.3 from the provided pdf", results)
    assert evaluation.sufficient is True
    assert not evaluation.missing_information


def test_low_score_poor_coverage_is_insufficient():
    evaluator = DeterministicEvidenceEvaluator(sufficiency_threshold=0.55)
    chunk = make_chunk("c1", "BackgroundTasks run code after the response.")
    results = [RetrievedChunk(chunk=chunk, score=0.1, dense_score=0.1)]
    evaluation = evaluator.evaluate("How does dependency injection resolve sub-dependencies?", results)
    assert evaluation.sufficient is False
    assert evaluation.missing_information


def test_subquery_coverage_flags_missing_subquestions():
    evaluator = DeterministicEvidenceEvaluator(sufficiency_threshold=0.9)
    chunk = make_chunk("c1", "Dependency injection lets you declare reusable logic with Depends.")
    results = [RetrievedChunk(chunk=chunk, score=0.9)]
    evaluation = evaluator.evaluate(
        "dependency injection and validation",
        results,
        required_subqueries=["How does dependency injection work?", "How does request validation work?"],
    )
    assert "How does request validation work?" in evaluation.missing_information


# ---------------------------------------------------------------------------
# Query reformulation
# ---------------------------------------------------------------------------

def test_reformulate_strips_question_words_on_attempt_2():
    prev_eval = EvidenceEvaluation(sufficient=False, confidence=0.2, missing_information=[], reason="low score")
    result = reformulate("How does dependency injection work?", prev_eval, attempt=2)
    assert "how" not in result.split()
    assert "does" not in result.split()
    assert "dependency" in result


def test_reformulate_uses_missing_terms_on_attempt_3():
    prev_eval = EvidenceEvaluation(
        sufficient=False, confidence=0.1,
        missing_information=["coverage for term(s): validation, pydantic"],
        reason="missing terms",
    )
    result = reformulate("How does dependency injection work?", prev_eval, attempt=3)
    assert "validation" in result
    assert "pydantic" in result


# ---------------------------------------------------------------------------
# Multi-hop decomposition
# ---------------------------------------------------------------------------

def test_decompose_multi_hop_splits_on_interact_with():
    subqueries = decompose_multi_hop(
        "How does dependency injection interact with request validation?"
    )
    assert len(subqueries) == 3
    assert "dependency injection" in subqueries[0].lower()
    assert "request validation" in subqueries[1].lower()


def test_decompose_falls_back_to_original_when_no_connector_found():
    q = "What does Depends do?"
    assert decompose_multi_hop(q) == [q]


# ---------------------------------------------------------------------------
# Retrieval agent: retry limits, dedup
# ---------------------------------------------------------------------------

def _build_agent(tmp_path, max_attempts=3):
    chunks = [
        make_chunk("c1", "FastAPI dependency injection uses Depends for declaring dependencies."),
        make_chunk("c2", "Middleware wraps every request and response in FastAPI."),
        make_chunk("c3", "Sub-dependencies let a dependency depend on another dependency."),
    ]
    embedder = HashingEmbedder(dim=64)
    bm25 = BM25Index()
    bm25.build(chunks)
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="agent_test")
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(bm25_index=bm25, vector_store=store, embedder=embedder)
    evaluator = DeterministicEvidenceEvaluator(sufficiency_threshold=0.99)  # force retries
    return RetrievalAgent(retriever=retriever, evaluator=evaluator, max_attempts=max_attempts)


def test_retrieval_agent_never_exceeds_max_attempts(tmp_path):
    agent = _build_agent(tmp_path, max_attempts=3)
    outcome = agent.run("How does Depends work?", QueryType.SIMPLE_LOOKUP, top_k=5)
    assert len(outcome.attempts) <= 3


def test_retrieval_agent_stops_early_when_sufficient(tmp_path):
    chunks = [make_chunk("c1", "FastAPI dependency injection uses Depends.")]
    embedder = HashingEmbedder(dim=64)
    bm25 = BM25Index()
    bm25.build(chunks)
    store = ChromaVectorStore(path=tmp_path / "chroma2", collection_name="agent_test2")
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(bm25_index=bm25, vector_store=store, embedder=embedder)
    evaluator = DeterministicEvidenceEvaluator(sufficiency_threshold=0.01)  # trivially satisfied
    agent = RetrievalAgent(retriever=retriever, evaluator=evaluator, max_attempts=3)
    outcome = agent.run("How does Depends work?", QueryType.SIMPLE_LOOKUP, top_k=5)
    assert len(outcome.attempts) == 1


def test_dedupe_by_chunk_id_keeps_highest_score():
    c = make_chunk("c1", "text")
    results = [
        RetrievedChunk(chunk=c, score=0.3),
        RetrievedChunk(chunk=c, score=0.9),
    ]
    deduped = _dedupe_by_chunk_id(results)
    assert len(deduped) == 1
    assert deduped[0].score == 0.9
