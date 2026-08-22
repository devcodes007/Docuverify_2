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
    """Deterministic provider for tests/offline dev: extracts the evidence
    block from the user prompt and produces a templated answer that quotes
    the first sentence of each evidence chunk, so tests can assert on
    citation behavior without a live model."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if "Evidence:" not in user_prompt:
            return "I don't have enough information to answer that."
        evidence_section = user_prompt.split("Evidence:", 1)[1]
        chunks = [c for c in evidence_section.split("\n\n") if c.strip().startswith("[")]
        if not chunks:
            return "The retrieved documentation does not address this question."
        lines = []
        for i, block in enumerate(chunks, start=1):
            body = block.split("\n", 1)[-1]
            first_sentence = body.strip().split(". ")[0]
            lines.append(f"{first_sentence.strip()}. [{i}]")
        return " ".join(lines)


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
