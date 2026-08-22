"""
Orchestrator: the single entrypoint the API calls. Wires together

    query_router -> retrieval_agent -> generation.llm -> verification.groundedness
                                                        -> verification.retry_policy

and assembles the AgentTrace + QueryResponse described in the API spec.
Kept intentionally thin -- all the actual decision logic lives in the
modules it calls, so this file is mostly sequencing and trace-building.
"""
from __future__ import annotations

import time

from app.agents.evidence_evaluator import EvidenceEvaluator
from app.agents.query_router import QueryRouter
from app.agents.retrieval_agent import RetrievalAgent
from app.generation.llm import LLMProvider
from app.generation.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from app.logging_config import get_logger, log_event
from app.models.schemas import (
    AgentTrace,
    QueryResponse,
    SourceRef,
    TraceStep,
)
from app.verification.claim_checker import verify_claims
from app.verification.groundedness import GroundednessClassifier
from app.verification.retry_policy import REFUSAL_MESSAGE, RetryDecision, RetryState, decide

logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        router: QueryRouter,
        retrieval_agent: RetrievalAgent,
        llm: LLMProvider,
        groundedness: GroundednessClassifier,
        evaluator: EvidenceEvaluator,
        max_verification_attempts: int = 2,
        enable_claim_checking: bool = False,
    ):
        self.router = router
        self.retrieval_agent = retrieval_agent
        self.llm = llm
        self.groundedness = groundedness
        self.evaluator = evaluator
        self.max_verification_attempts = max_verification_attempts
        self.enable_claim_checking = enable_claim_checking

    def answer(self, question: str, top_k: int, debug: bool = False) -> QueryResponse:
        start = time.perf_counter()
        trace = AgentTrace(query=question)
        request_id = f"req-{int(start * 1000)}"

        query_type = self.router.classify(question)
        trace.classification = query_type
        trace.steps.append(TraceStep(step="classification", detail={"result": query_type.value}))
        log_event(logger, "query_classified", request_id=request_id, query_type=query_type.value)

        outcome = self.retrieval_agent.run(question, query_type, top_k=top_k)
        trace.attempts = outcome.attempts
        trace.subqueries = outcome.subqueries
        trace.steps.append(
            TraceStep(
                step="retrieval",
                detail={"strategy": outcome.strategy.value, "documents_retrieved": len(outcome.results)},
            )
        )

        state = RetryState()
        answer_text = ""
        groundedness_result = None
        results = outcome.results
        refused = False

        for verification_pass in range(self.max_verification_attempts):
            if not results:
                answer_text = REFUSAL_MESSAGE
                groundedness_result = self.groundedness.predict(question, "", answer_text)
                refused = True
                break

            context = "\n\n".join(r.chunk.text for r in results)
            prompt = build_answer_prompt(question, results)
            answer_text = self.llm.generate(ANSWER_SYSTEM_PROMPT, prompt)
            trace.steps.append(TraceStep(step="generation", detail={"pass": verification_pass + 1}))

            groundedness_result = self.groundedness.predict(question, context, answer_text)
            trace.steps.append(
                TraceStep(
                    step="groundedness",
                    detail={"label": groundedness_result.label.value, "confidence": groundedness_result.confidence},
                )
            )
            log_event(
                logger, "groundedness_checked", request_id=request_id,
                label=groundedness_result.label.value, confidence=groundedness_result.confidence,
                pass_number=verification_pass + 1,
            )

            decision = decide(groundedness_result.label, state, self.max_verification_attempts)
            if decision == RetryDecision.ACCEPT:
                break
            if decision == RetryDecision.REFUSE:
                answer_text = REFUSAL_MESSAGE
                refused = True
                trace.steps.append(TraceStep(step="refusal", detail={"reason": groundedness_result.label.value}))
                break

            # RETRY: re-run retrieval once more for this same query before regenerating
            state.verification_attempts += 1
            retry_outcome = self.retrieval_agent.run(question, query_type, top_k=top_k)
            results = retry_outcome.results
            trace.attempts.extend(retry_outcome.attempts)
            trace.steps.append(TraceStep(step="post_groundedness_retry", detail={"attempt": state.verification_attempts}))

        claims = None
        if self.enable_claim_checking and not refused and results:
            context = "\n\n".join(r.chunk.text for r in results)
            claims = verify_claims(question, context, answer_text, self.groundedness)

        sources = [
            SourceRef(
                chunk_id=r.chunk.chunk_id, title=r.chunk.title, section=r.chunk.display_heading(),
                source_url=r.chunk.source_url, score=round(r.score, 4),
            )
            for r in results
        ] if not refused else []

        latency_ms = int((time.perf_counter() - start) * 1000)
        log_event(
            logger, "query_completed", request_id=request_id, latency_ms=latency_ms,
            refused=refused, groundedness=groundedness_result.label.value if groundedness_result else None,
        )

        return QueryResponse(
            answer=answer_text,
            query_type=query_type,
            retrieval_strategy=outcome.strategy,
            groundedness=groundedness_result,
            sources=sources,
            retrieval_attempts=len(trace.attempts),
            latency_ms=latency_ms,
            claims=claims,
            trace=trace if debug else None,
            refused=refused,
        )
