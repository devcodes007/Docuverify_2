from app.ingestion.cleaner import clean_markdown
from app.ingestion.chunker import chunk_document
from app.ingestion.loader import RawDocument
from app.ingestion.metadata import build_chunks_for_document

SAMPLE_MD = """# Dependencies

FastAPI has a very powerful Dependency Injection system.

## Declaring a dependency

You declare a dependency using `Depends`.

```python
from fastapi import Depends

def common_params(q: str = None):
    return {"q": q}
```

That's it. FastAPI will call the function for you.

## Sub-dependencies

Dependencies can also depend on other dependencies.

```python
def sub_dep(commons: dict = Depends(common_params)):
    return commons
```

This allows building a dependency graph.
"""


def test_clean_markdown_extracts_headings_and_code():
    blocks = clean_markdown(SAMPLE_MD)
    kinds = [b.kind for b in blocks]
    assert "heading" in kinds
    assert "code" in kinds
    code_blocks = [b for b in blocks if b.kind == "code"]
    assert any("Depends" in b.text for b in code_blocks)


def test_chunker_keeps_code_glued_to_explanation():
    blocks = clean_markdown(SAMPLE_MD)
    drafts = chunk_document(blocks, chunk_size_tokens=500, chunk_overlap_tokens=0)
    # find the chunk containing the Depends code sample
    target = [d for d in drafts if "def common_params" in d.text]
    assert target, "expected a chunk containing the code example"
    assert "You declare a dependency using" in target[0].text, (
        "explanation must stay in the same chunk as the code it explains"
    )


def test_chunker_respects_small_budget_and_splits_sections():
    blocks = clean_markdown(SAMPLE_MD)
    drafts = chunk_document(blocks, chunk_size_tokens=15, chunk_overlap_tokens=0)
    assert len(drafts) > 1, "a tight token budget should force multiple chunks"


def test_chunker_tracks_heading_path():
    blocks = clean_markdown(SAMPLE_MD)
    drafts = chunk_document(blocks, chunk_size_tokens=500, chunk_overlap_tokens=0)
    sub_dep_chunk = [d for d in drafts if "sub_dep" in d.text][0]
    assert sub_dep_chunk.heading_path == ["Dependencies", "Sub-dependencies"]


def test_build_chunks_for_document_attaches_metadata():
    doc = RawDocument(
        document_id="dependencies",
        source_url="file:///data/raw/dependencies.md",
        title="Dependencies",
        raw_html_or_markdown=SAMPLE_MD,
        is_html=False,
    )
    chunks = build_chunks_for_document(doc, chunk_size_tokens=500, chunk_overlap_tokens=0)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "dependencies"
        assert c.source_url == doc.source_url
        assert c.chunk_id.startswith("dependencies::chunk-")
    sub_dep_chunk = [c for c in chunks if "sub_dep" in c.text][0]
    assert sub_dep_chunk.section == "Dependencies"
    assert sub_dep_chunk.subsection == "Sub-dependencies"


def test_no_blind_fixed_width_splitting_mid_code_block():
    """Regression guard: a code block should never be cut in half even
    under a very tight token budget, since that would produce a chunk with
    unrunnable/meaningless partial code."""
    blocks = clean_markdown(SAMPLE_MD)
    drafts = chunk_document(blocks, chunk_size_tokens=5, chunk_overlap_tokens=0)
    for d in drafts:
        if "```" in d.text:
            assert d.text.count("```") % 2 == 0, "code fence must be balanced, never split"
