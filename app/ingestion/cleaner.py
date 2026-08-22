"""
Cleaning / normalization.

Strips navigation/sidebar/footer boilerplate from HTML, normalizes
whitespace, and returns a structure that the chunker can walk: an ordered
list of (level, heading_text) and (paragraph|code, text) blocks. Markdown is
parsed directly since it's already close to this structure; HTML is parsed
with BeautifulSoup and known noisy containers are dropped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

NOISY_HTML_SELECTORS = [
    "nav", "footer", "header", "aside",
    "[class*=sidebar]", "[class*=navbar]", "[class*=breadcrumb]",
    "[class*=toc]", "[id*=sidebar]", "script", "style", "noscript",
]

HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}


@dataclass
class Block:
    kind: str  # "heading" | "paragraph" | "code"
    text: str
    level: int = 0  # heading level, 0 for non-headings


def clean_html(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "html.parser")
    for selector in NOISY_HTML_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    blocks: list[Block] = []
    body = soup.body or soup
    for el in body.find_all(list(HEADING_TAGS) + ["p", "pre", "li"]):
        if el.name in HEADING_TAGS:
            text = _normalize_whitespace(el.get_text())
            if text:
                blocks.append(Block("heading", text, HEADING_TAGS[el.name]))
        elif el.name == "pre":
            code_text = el.get_text()
            if code_text.strip():
                blocks.append(Block("code", code_text.rstrip("\n")))
        else:  # p, li
            text = _normalize_whitespace(el.get_text())
            if text and not el.find_parent("pre"):
                blocks.append(Block("paragraph", text))
    return blocks


def clean_markdown(markdown_text: str) -> list[Block]:
    blocks: list[Block] = []
    lines = markdown_text.splitlines()
    in_code = False
    code_buf: list[str] = []
    para_buf: list[str] = []

    def flush_paragraph():
        if para_buf:
            text = _normalize_whitespace(" ".join(para_buf))
            if text:
                blocks.append(Block("paragraph", text))
            para_buf.clear()

    for line in lines:
        fence_match = re.match(r"^\s*```", line)
        if fence_match:
            if in_code:
                blocks.append(Block("code", "\n".join(code_buf)))
                code_buf.clear()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            blocks.append(Block("heading", heading_match.group(2).strip(), level))
            continue

        if not line.strip():
            flush_paragraph()
            continue

        para_buf.append(line.strip())

    flush_paragraph()
    if code_buf:
        blocks.append(Block("code", "\n".join(code_buf)))
    return blocks


def clean(raw_text: str, is_html: bool) -> list[Block]:
    return clean_html(raw_text) if is_html else clean_markdown(raw_text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
