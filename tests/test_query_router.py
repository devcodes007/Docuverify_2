from app.agents.query_router import QueryRouter, extract_comparison_concepts
from app.models.schemas import QueryType


def test_simple_lookup_classification():
    router = QueryRouter()
    assert router.classify("What does Depends do?") == QueryType.SIMPLE_LOOKUP
    assert router.classify("What is FastAPI?") == QueryType.SIMPLE_LOOKUP


def test_comparison_classification():
    router = QueryRouter()
    assert (
        router.classify("What is the difference between Depends and middleware?")
        == QueryType.COMPARISON
    )
    assert router.classify("Depends vs middleware") == QueryType.COMPARISON


def test_multi_hop_classification():
    router = QueryRouter()
    result = router.classify(
        "How does FastAPI dependency injection interact with request validation?"
    )
    assert result == QueryType.MULTI_HOP


def test_multi_hop_connector_heuristic():
    router = QueryRouter()
    q = (
        "What happens when a dependency raises an exception and how does that "
        "affect the request and then what happens to background tasks?"
    )
    assert router.classify(q) == QueryType.MULTI_HOP


def test_llm_fallback_used_only_for_ambiguous_long_queries():
    calls = []

    class FakeLLM:
        def classify(self, question):
            calls.append(question)
            return QueryType.MULTI_HOP

    router = QueryRouter(llm_fallback=FakeLLM())
    short_q = "What is Depends?"
    router.classify(short_q)
    assert calls == [], "short unambiguous queries should never hit the LLM fallback"

    long_ambiguous_q = (
        "I was reading through some code the other day and I noticed there "
        "was a function decorator thing going on, what is that about exactly"
    )
    result = router.classify(long_ambiguous_q)
    assert calls == [long_ambiguous_q]
    assert result == QueryType.MULTI_HOP


def test_extract_comparison_concepts():
    concepts = extract_comparison_concepts(
        "What is the difference between Depends and middleware?"
    )
    assert concepts == ("Depends", "middleware")

    concepts2 = extract_comparison_concepts("BackgroundTasks vs middleware")
    assert concepts2 == ("BackgroundTasks", "middleware")

    assert extract_comparison_concepts("What is FastAPI?") is None
