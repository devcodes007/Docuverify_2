"""
Structure-aware chunking.

Walks the (heading, paragraph, code) block stream produced by cleaner.py and
groups blocks into chunks that:
  * never split a code block away from the paragraph(s) immediately
    preceding it (the explanation that gives the code meaning),
  * track the heading path (h1 > h2 > h3 ...) so each chunk knows its
    section/subsection,
  * respect a configurable token budget with configurable overlap.

Token counting here uses whitespace-splitting as a cheap, dependency-free
approximation of a real tokenizer -- swapping in a real tokenizer (e.g. the
embedding model's) is a one-line change in `_token_count` if higher accuracy
is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.cleaner import Block
from app.models.schemas import ContentType


def _token_count(text: str) -> int:
    return len(text.split())


@dataclass
class _Unit:
    """One or more blocks that must stay together (e.g. paragraph + the code
    block it introduces)."""
    blocks: list[Block]
    heading_path: list[str]

    @property
    def text(self) -> str:
        parts = []
        for b in self.blocks:
            parts.append(f"```\n{b.text}\n```" if b.kind == "code" else b.text)
        return "\n\n".join(parts)

    @property
    def content_type(self) -> ContentType:
        kinds = {b.kind for b in self.blocks}
        if kinds == {"code"}:
            return ContentType.CODE
        if "code" in kinds:
            return ContentType.MIXED
        return ContentType.PROSE

    @property
    def token_count(self) -> int:
        return _token_count(self.text)


@dataclass
class ChunkDraft:
    text: str
    heading_path: list[str]
    content_type: ContentType


def _build_units(blocks: list[Block]) -> list[_Unit]:
    """Group blocks so a paragraph directly followed by code stays in one
    unit, tracking the current heading path as we go."""
    units: list[_Unit] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    pending: list[Block] = []

    def current_path() -> list[str]:
        return [t for _, t in heading_stack]

    def flush_pending():
        if pending:
            units.append(_Unit(blocks=list(pending), heading_path=current_path()))
            pending.clear()

    for block in blocks:
        if block.kind == "heading":
            flush_pending()
            while heading_stack and heading_stack[-1][0] >= block.level:
                heading_stack.pop()
            heading_stack.append((block.level, block.text))
            continue

        if block.kind == "code" and pending and pending[-1].kind == "paragraph":
            # keep the explanation glued to the code that follows it
            pending.append(block)
            flush_pending()
            continue

        if block.kind == "code" and not pending:
            pending.append(block)
            flush_pending()
            continue

        # paragraph
        flush_pending()
        pending.append(block)

    flush_pending()
    return units


def chunk_document(
    blocks: list[Block],
    chunk_size_tokens: int = 350,
    chunk_overlap_tokens: int = 50,
) -> list[ChunkDraft]:
    """Greedily pack units into chunks up to chunk_size_tokens, carrying
    `chunk_overlap_tokens` worth of trailing text into the next chunk so
    retrieval doesn't lose context at chunk boundaries. A unit larger than
    the whole budget (e.g. a long code block) becomes its own chunk rather
    than being split mid-block."""
    units = _build_units(blocks)
    if not units:
        return []

    drafts: list[ChunkDraft] = []
    current_units: list[_Unit] = []
    current_tokens = 0

    def flush():
        nonlocal current_units, current_tokens
        if not current_units:
            return
        text = "\n\n".join(u.text for u in current_units)
        heading_path = current_units[-1].heading_path
        content_type = (
            ContentType.CODE
            if all(u.content_type == ContentType.CODE for u in current_units)
            else ContentType.MIXED
            if any(u.content_type != ContentType.PROSE for u in current_units)
            else ContentType.PROSE
        )
        drafts.append(ChunkDraft(text=text, heading_path=heading_path, content_type=content_type))

        # carry overlap: keep trailing units whose combined token count is
        # <= chunk_overlap_tokens for the start of the next chunk
        overlap_units: list[_Unit] = []
        overlap_tokens = 0
        for u in reversed(current_units):
            if overlap_tokens + u.token_count > chunk_overlap_tokens:
                break
            overlap_units.insert(0, u)
            overlap_tokens += u.token_count
        current_units = overlap_units
        current_tokens = overlap_tokens

    for unit in units:
        if current_tokens + unit.token_count > chunk_size_tokens and current_units:
            flush()
        current_units.append(unit)
        current_tokens += unit.token_count

    flush()
    return drafts
