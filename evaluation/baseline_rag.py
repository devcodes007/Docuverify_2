"""
Baseline: the "too basic" pipeline the project spec explicitly says not to
ship as the main system --

    query -> dense retrieval -> LLM -> answer

No query classification, no retry, no evidence evaluation, no groundedness
verification. Used only as a comparison point in evaluation/benchmark.py to
demonstrate what the agentic architecture in app/agents adds.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.generation.llm import LLMProvider
from app.generation.prompts import ANSWER_SYSTEM_PROMPT, build_answer_prompt
from app.retrieval.hybrid import HybridRetriever


@dataclass
class BaselineResponse:
    answer: str
    sources_document_ids: set[str]


class BaselineRAG:
    def __init__(self, retriever: HybridRetriever, llm: LLMProvider, top_k: int = 6):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str) -> BaselineResponse:
        results = self.retriever.search_dense(question, top_k=self.top_k)
        if not results:
            return BaselineResponse(answer="No information found.", sources_document_ids=set())
        prompt = build_answer_prompt(question, results)
        answer_text = self.llm.generate(ANSWER_SYSTEM_PROMPT, prompt)
        return BaselineResponse(
            answer=answer_text, sources_document_ids={r.chunk.document_id for r in results}
        )
