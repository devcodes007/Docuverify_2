"""
BM25 lexical retrieval.

Tuned for documentation search over code identifiers: the tokenizer keeps
`snake_case`, `CamelCase`, and dotted names (`app.main`, `HTTPException`)
intact as single tokens in addition to their split forms, so a query for
"HTTPException" or "Depends" matches exactly, while a natural-language query
still gets reasonable token overlap.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.models.schemas import Chunk

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.]+")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        tokens.append(raw.lower())
        # also add split forms of identifiers so "dependency injection"
        # matches "dependency_injection" / "DependencyInjection". Camel
        # splitting must happen before lowercasing, since case is the signal.
        if "_" in raw:
            tokens.extend(p.lower() for p in raw.split("_") if p)
        camel_parts = _CAMEL_SPLIT.split(raw)
        if len(camel_parts) > 1:
            tokens.extend(p.lower() for p in camel_parts if p)
    return tokens


@dataclass
class BM25Result:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def is_ready(self) -> bool:
        return self._bm25 is not None

    def search(self, query: str, top_k: int = 8) -> list[BM25Result]:
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True
        )[:top_k]
        return [BM25Result(chunk=c, score=float(s)) for c, s in ranked if s > 0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks}, f)
        # rebuild bm25 on load to avoid pickling the (large, derivable) index

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.build(data["chunks"])
