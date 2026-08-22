"""
Pluggable document sources.

To ingest a *different* open-source library's docs, implement a new
DocumentSource subclass (e.g. one that reads a git checkout, or hits a
different docs site) and register it in SOURCE_REGISTRY. Nothing else in
the ingestion/retrieval/agent pipeline needs to change.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RawDocument:
    document_id: str
    source_url: str
    title: str
    raw_html_or_markdown: str
    is_html: bool


class DocumentSource(ABC):
    """Interface every ingestion source implements."""

    @abstractmethod
    def load(self) -> list[RawDocument]:
        ...


class LocalMarkdownSource(DocumentSource):
    """Reads .md/.html files from a local directory (default: data/raw).

    This is the source used for the bundled example corpus and for tests,
    since it requires no network access.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def load(self) -> list[RawDocument]:
        docs: list[RawDocument] = []
        if not self.directory.exists():
            logger.warning("raw data directory %s does not exist", self.directory)
            return docs

        for path in sorted(self.directory.glob("**/*")):
            if path.suffix.lower() not in {".md", ".markdown", ".html", ".htm"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            title = _guess_title(text, path)
            docs.append(
                RawDocument(
                    document_id=path.stem,
                    source_url=f"file://{path}",
                    title=title,
                    raw_html_or_markdown=text,
                    is_html=path.suffix.lower() in {".html", ".htm"},
                )
            )
        logger.info("LocalMarkdownSource loaded %d documents from %s", len(docs), self.directory)
        return docs


class WebDocSource(DocumentSource):
    """Fetches a fixed, explicit list of URLs.

    Security: only domains in `allowed_domains` are fetched, so this can't be
    pointed at arbitrary remote content. There is no crawling/link-following;
    the caller supplies the exact URL list.
    """

    def __init__(self, urls: list[str], allowed_domains: list[str], fetcher):
        """`fetcher` is a callable url -> (html_text, ok: bool) so this class
        stays testable without real HTTP calls."""
        self.urls = urls
        self.allowed_domains = set(allowed_domains)
        self.fetcher = fetcher

    def load(self) -> list[RawDocument]:
        docs: list[RawDocument] = []
        for url in self.urls:
            domain = urlparse(url).netloc
            if domain not in self.allowed_domains:
                logger.warning("refusing to fetch disallowed domain: %s", domain)
                continue
            html, ok = self.fetcher(url)
            if not ok or not html:
                logger.warning("fetch failed for %s", url)
                continue
            docs.append(
                RawDocument(
                    document_id=_slugify(url),
                    source_url=url,
                    title=_guess_title(html, Path(url)),
                    raw_html_or_markdown=html,
                    is_html=True,
                )
            )
        return docs


def _guess_title(text: str, path: Path) -> str:
    md_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    if md_match:
        return md_match.group(1).strip()
    html_match = re.search(r"<h1[^>]*>(.*?)</h1>", text, flags=re.IGNORECASE | re.DOTALL)
    if html_match:
        return re.sub(r"<[^>]+>", "", html_match.group(1)).strip()
    return path.stem.replace("-", " ").replace("_", " ").title()


def _slugify(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", url).strip("-").lower()


SOURCE_REGISTRY: dict[str, type[DocumentSource]] = {
    "local": LocalMarkdownSource,
    "web": WebDocSource,
}
