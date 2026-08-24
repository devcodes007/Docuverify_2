"""
LLM provider abstraction.

Everything downstream depends only on `LLMProvider.generate(system, user) ->
str`. Concrete providers:

  * OllamaProvider          -- local models via Ollama's HTTP API (default;
    no API key needed, nothing leaves the machine).
  * OpenAICompatibleProvider -- any OpenAI-compatible chat completions API
    (OpenAI itself, or a self-hosted compatible server); API key is read
    from OPENAI_API_KEY, never hardcoded.
  * MockLLMProvider          -- deterministic canned responses, used in
    tests and for exercising the pipeline without a live LLM.

Provider selection is controlled entirely by `LLM_PROVIDER` in config/env.
"""
from __future__ import annotations

import re
from typing import Protocol

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OllamaProvider:
    def __init__(self, model: str, api_base: str, timeout: int = 60):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = httpx.post(
                f"{self.api_base}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except httpx.HTTPError as exc:
            logger.error("Ollama generation failed: %s", exc)
            raise LLMGenerationError(str(exc)) from exc


class OpenAICompatibleProvider:
    def __init__(self, model: str, api_base: str, api_key: str | None, timeout: int = 60):
        if not api_key:
            raise ValueError("OPENAI_API_KEY (or compatible) is required for this provider")
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = httpx.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPError as exc:
            logger.error("OpenAI-compatible generation failed: %s", exc)
            raise LLMGenerationError(str(exc)) from exc


class MockLLMProvider:
    """Deterministic provider for tests/offline dev.

    It is intentionally not a real language model, but it should still make
    local demos useful: rank evidence sentences by overlap with the question
    so different PDF questions do not all echo the same first chunk.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "Evidence:" not in user_prompt:
            return "I don't have enough information to answer that."

        question = _extract_question(user_prompt)
        evidence_section = user_prompt.split("Evidence:", 1)[1]
        chunks = _parse_evidence_chunks(evidence_section)
        if not chunks:
            return "The retrieved documentation does not address this question."

        terms = _query_terms(question)
        ranked: list[tuple[int, int, str, int]] = []
        for citation, body in chunks:
            for position, sentence in enumerate(_split_sentences(body)):
                score = sum(1 for term in terms if term in sentence.lower())
                ranked.append((score, -position, sentence, citation))

        focused = [item for item in ranked if item[0] > 0]
        if focused:
            max_score = max(item[0] for item in focused)
            focused = [item for item in focused if item[0] == max_score]
        else:
            focused = ranked[:3]

        focused.sort(key=lambda item: (item[0], item[1]), reverse=True)
        lines = []
        for _, _, sentence, citation in focused[:4]:
            clean = sentence.strip().rstrip(".")
            if clean:
                lines.append(f"{clean}. [{citation}]")
        return " ".join(lines)


def _extract_question(prompt: str) -> str:
    match = re.search(r"^Question:\s*(.+?)\n\nEvidence:", prompt, flags=re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_evidence_chunks(evidence_section: str) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    for block in evidence_section.split("\n\n"):
        stripped = block.strip()
        if not stripped.startswith("["):
            continue
        match = re.match(r"\[(\d+)\].*?\n(.+)", stripped, flags=re.DOTALL)
        if not match:
            continue
        chunks.append((int(match.group(1)), match.group(2).strip()))
    return chunks


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]


def _query_terms(question: str) -> set[str]:
    stopwords = {
        "about", "after", "again", "answer", "does", "from", "give", "have",
        "into", "what", "when", "where", "which", "with", "your", "summary",
        "summarize", "provided", "pdf", "please", "tell", "show", "explain",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9.]+", question.lower())
        if len(token) > 2 and token not in stopwords
    }


class LLMGenerationError(RuntimeError):
    pass


def build_llm_provider(
    provider: str, model: str, api_base: str, api_key: str | None
) -> LLMProvider:
    if provider == "ollama":
        return OllamaProvider(model=model, api_base=api_base)
    if provider == "openai_compatible":
        return OpenAICompatibleProvider(model=model, api_base=api_base, api_key=api_key)
    if provider == "mock":
        return MockLLMProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
