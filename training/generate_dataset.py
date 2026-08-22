"""
Build a SUPPORTED / CONTRADICTED / UNSUPPORTED dataset from ingested
documentation chunks.

Two generation modes, both producing the same output schema:

  * LLM-assisted (`--llm`): uses the configured LLM_PROVIDER to generate a
    question for each chunk, a grounded answer, an unsupported answer (adds
    a claim the chunk doesn't support), and a contradicted answer (negates
    a claim the chunk does support). Higher-quality, more varied examples;
    requires a working LLM provider (Ollama running locally, or an
    OpenAI-compatible endpoint with OPENAI_API_KEY set).

  * Template-based (default): deterministic, no LLM required. Produces:
      - SUPPORTED: a question built from the chunk's heading + a paraphrase
        drawn directly from chunk sentences as the answer.
      - UNSUPPORTED: the SUPPORTED question, but paired with a *different*
        chunk's context (so the answer's claims aren't grounded in what's
        actually retrieved).
      - CONTRADICTED: the SUPPORTED answer with a negation inserted, kept
        paired with its own correct context (so the answer now conflicts
        with, rather than merely being absent from, the context).

This always produces a reproducible dataset from any ingested corpus, and
is what CI / a from-scratch clone can run without any external service.

Usage:
    python -m training.generate_dataset --raw-dir data/raw --out data/processed/groundedness_raw.jsonl
    python -m training.generate_dataset --raw-dir data/raw --out data/processed/groundedness_raw.jsonl --llm
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import get_settings
from app.generation.llm import build_llm_provider
from app.ingestion.loader import LocalMarkdownSource
from app.ingestion.metadata import build_chunks
from app.models.schemas import Chunk

random.seed(13)

_NEGATIONS = [
    (r"\bcan\b", "cannot"),
    (r"\bwill\b", "will not"),
    (r"\buses\b", "does not use"),
    (r"\bruns\b", "does not run"),
    (r"\bvalidates\b", "does not validate"),
    (r"\bresolves\b", "fails to resolve"),
    (r"\bstops\b", "continues"),
    (r"\bautomatically\b", "manually"),
]


@dataclass
class GroundednessExample:
    question: str
    context: str
    answer: str
    label: str  # SUPPORTED | CONTRADICTED | UNSUPPORTED
    document_id: str  # used later for leakage-safe splitting


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 25]


def _question_from_heading(chunk: Chunk) -> str:
    topic = chunk.display_heading() or chunk.title
    return f"How does {topic.split(' > ')[-1].lower()} work in FastAPI?"


def _negate(sentence: str) -> str | None:
    for pattern, replacement in _NEGATIONS:
        if re.search(pattern, sentence):
            return re.sub(pattern, replacement, sentence, count=1)
    return None


def generate_template_examples(chunks: list[Chunk]) -> list[GroundednessExample]:
    examples: list[GroundednessExample] = []
    usable = [(c, _sentences(c.text)) for c in chunks]
    usable = [(c, s) for c, s in usable if s]

    for i, (chunk, sentences) in enumerate(usable):
        question = _question_from_heading(chunk)
        grounded_answer = " ".join(sentences[:2])

        # SUPPORTED
        examples.append(
            GroundednessExample(
                question=question, context=chunk.text, answer=grounded_answer,
                label="SUPPORTED", document_id=chunk.document_id,
            )
        )

        # UNSUPPORTED: same question/answer, wrong (unrelated) context
        other_chunk, _ = usable[(i + 1) % len(usable)] if len(usable) > 1 else (chunk, sentences)
        if other_chunk.chunk_id != chunk.chunk_id:
            examples.append(
                GroundednessExample(
                    question=question, context=other_chunk.text, answer=grounded_answer,
                    label="UNSUPPORTED", document_id=chunk.document_id,
                )
            )

        # CONTRADICTED: negate a claim, keep the correct (now-contradicted) context
        for sentence in sentences:
            negated = _negate(sentence)
            if negated:
                examples.append(
                    GroundednessExample(
                        question=question, context=chunk.text, answer=negated,
                        label="CONTRADICTED", document_id=chunk.document_id,
                    )
                )
                break

    return examples


def generate_llm_examples(chunks: list[Chunk]) -> list[GroundednessExample]:
    settings = get_settings()
    llm = build_llm_provider(
        provider=settings.llm_provider, model=settings.llm_model,
        api_base=settings.llm_api_base, api_key=settings.openai_api_key,
    )
    examples: list[GroundednessExample] = []
    system = (
        "You generate training examples for a groundedness classifier. "
        "Given a documentation excerpt, output exactly three lines, no other text:\n"
        "QUESTION: <a question the excerpt answers>\n"
        "SUPPORTED_ANSWER: <a short answer fully supported by the excerpt>\n"
        "UNSUPPORTED_CLAIM: <a short, plausible-sounding claim the excerpt does NOT support>"
    )
    for chunk in chunks:
        raw = llm.generate(system, f"Documentation excerpt:\n{chunk.text}")
        q = _extract_field(raw, "QUESTION")
        supported = _extract_field(raw, "SUPPORTED_ANSWER")
        unsupported = _extract_field(raw, "UNSUPPORTED_CLAIM")
        if not (q and supported):
            continue
        examples.append(GroundednessExample(q, chunk.text, supported, "SUPPORTED", chunk.document_id))
        if unsupported:
            examples.append(GroundednessExample(q, chunk.text, unsupported, "UNSUPPORTED", chunk.document_id))
        negated = _negate(supported)
        if negated:
            examples.append(GroundednessExample(q, chunk.text, negated, "CONTRADICTED", chunk.document_id))
    return examples


def _extract_field(raw: str, field: str) -> str:
    match = re.search(rf"{field}:\s*(.+)", raw)
    return match.group(1).strip() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/groundedness_raw.jsonl")
    parser.add_argument("--llm", action="store_true", help="use the configured LLM instead of templates")
    parser.add_argument("--chunk-size-tokens", type=int, default=350)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=50)
    args = parser.parse_args()

    documents = LocalMarkdownSource(args.raw_dir).load()
    if not documents:
        raise SystemExit(f"No documents found in {args.raw_dir}; run ingestion first or check the path.")
    chunks = build_chunks(documents, args.chunk_size_tokens, args.chunk_overlap_tokens)

    examples = generate_llm_examples(chunks) if args.llm else generate_template_examples(chunks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")

    counts = {}
    for ex in examples:
        counts[ex.label] = counts.get(ex.label, 0) + 1
    print(f"Wrote {len(examples)} examples to {out_path}")
    print(f"Label distribution: {counts}")


if __name__ == "__main__":
    main()
