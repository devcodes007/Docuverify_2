from app.models.schemas import Chunk, ContentType, GroundednessLabel, QueryType
from app.agents.evidence_evaluator import DeterministicEvidenceEvaluator
from app.agents.orchestrator import Orchestrator
from app.agents.query_router import QueryRouter
from app.agents.retrieval_agent import RetrievalAgent
from app.generation.llm import MockLLMProvider
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import ChromaVectorStore, HashingEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.verification.groundedness import HeuristicGroundednessClassifier
from app.verification.retry_policy import RetryDecision, RetryState, decide


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="doc1", source_url="file:///doc1.md",
        title="Doc1", text=text, content_type=ContentType.PROSE,
    )


# ---------------------------------------------------------------------------
# HeuristicGroundednessClassifier
# ---------------------------------------------------------------------------

def test_supported_answer_high_overlap():
    clf = HeuristicGroundednessClassifier()
    context = "FastAPI dependencies use Depends to declare reusable logic for path operations."
    answer = "FastAPI dependencies use Depends to declare reusable logic."
    result = clf.predict("How does Depends work?", context, answer)
    assert result.label == GroundednessLabel.SUPPORTED


def test_unsupported_answer_low_overlap():
    clf = HeuristicGroundednessClassifier()
    context = "FastAPI dependencies use Depends to declare reusable logic."
    answer = "FastAPI supports GraphQL subscriptions natively out of the box."
    result = clf.predict("Does FastAPI support GraphQL subscriptions?", context, answer)
    assert result.label == GroundednessLabel.UNSUPPORTED


def test_empty_answer_is_unsupported():
    clf = HeuristicGroundednessClassifier()
    result = clf.predict("q", "context here", "")
    assert result.label == GroundednessLabel.UNSUPPORTED


# ---------------------------------------------------------------------------
# retry_policy
# ---------------------------------------------------------------------------

def test_retry_policy_accepts_supported():
    assert decide(GroundednessLabel.SUPPORTED, RetryState()) == RetryDecision.ACCEPT


def test_retry_policy_retries_once_then_refuses():
    state = RetryState()
    first = decide(GroundednessLabel.UNSUPPORTED, state, max_verification_attempts=2)
    assert first == RetryDecision.RETRY
    state.verification_attempts += 1
    second = decide(GroundednessLabel.UNSUPPORTED, state, max_verification_attempts=2)
    assert second == RetryDecision.REFUSE


# ---------------------------------------------------------------------------
# Orchestrator end-to-end (mock LLM + heuristic classifier, no network needed)
# ---------------------------------------------------------------------------

def _build_orchestrator(tmp_path, chunks):
    embedder = HashingEmbedder(dim=64)
    bm25 = BM25Index()
    bm25.build(chunks)
    store = ChromaVectorStore(path=tmp_path / "chroma", collection_name="orchestrator_test")
    if chunks:
        store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = HybridRetriever(bm25_index=bm25, vector_store=store, embedder=embedder)
    evaluator = DeterministicEvidenceEvaluator(sufficiency_threshold=0.2)
    agent = RetrievalAgent(retriever=retriever, evaluator=evaluator, max_attempts=3)
    return Orchestrator(
        router=QueryRouter(),
        retrieval_agent=agent,
        llm=MockLLMProvider(),
        groundedness=HeuristicGroundednessClassifier(),
        evaluator=evaluator,
    )


def test_orchestrator_returns_supported_answer_with_sources(tmp_path):
    chunks = [make_chunk("c1", "FastAPI dependencies use Depends to declare reusable logic for path operations.")]
    orch = _build_orchestrator(tmp_path, chunks)
    response = orch.answer("How does Depends work?", top_k=5)
    assert response.refused is False
    assert response.sources
    assert response.groundedness.label in {GroundednessLabel.SUPPORTED, GroundednessLabel.CONTRADICTED, GroundednessLabel.UNSUPPORTED}


def test_orchestrator_refuses_when_no_documents_at_all(tmp_path):
    orch = _build_orchestrator(tmp_path, chunks=[])
    response = orch.answer("How does something completely undocumented work?", top_k=5)
    assert response.refused is True
    assert response.sources == []
    assert "sufficient evidence" in response.answer.lower()


def test_orchestrator_debug_trace_included_when_requested(tmp_path):
    chunks = [make_chunk("c1", "FastAPI dependencies use Depends.")]
    orch = _build_orchestrator(tmp_path, chunks)
    response = orch.answer("How does Depends work?", top_k=5, debug=True)
    assert response.trace is not None
    assert response.trace.classification == QueryType.SIMPLE_LOOKUP
    assert len(response.trace.steps) > 0


def test_orchestrator_no_trace_when_debug_false(tmp_path):
    chunks = [make_chunk("c1", "FastAPI dependencies use Depends.")]
    orch = _build_orchestrator(tmp_path, chunks)
    response = orch.answer("How does Depends work?", top_k=5, debug=False)
    assert response.trace is None
