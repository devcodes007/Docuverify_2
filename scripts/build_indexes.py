"""
Rebuilds the BM25 and dense indexes from whatever is currently in
data/raw. Functionally identical to scripts/ingest_docs.py -- kept as a
separate, clearly-named entrypoint per the project's requested script
layout (scripts/ingest_docs.py, scripts/build_indexes.py) since "ingest"
and "(re)build indexes" are the operations people reach for at different
points in the workflow (first-time setup vs. after editing chunking
config).

Usage:
    python scripts/build_indexes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ingest_docs import main as ingest_main

if __name__ == "__main__":
    ingest_main()
