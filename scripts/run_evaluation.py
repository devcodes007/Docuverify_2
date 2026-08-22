"""
Runs the full evaluation suite in sequence: retrieval benchmarks, RAG
evaluation, the live groundedness classifier spot-check, and the baseline
vs agentic comparison. Writes all reports to evaluation/results/.

Usage:
    python scripts/run_evaluation.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STEPS = [
    ("Retrieval evaluation (BM25 / Dense / Hybrid)", [sys.executable, "-m", "evaluation.retrieval_eval"]),
    ("RAG evaluation (agentic pipeline)", [sys.executable, "-m", "evaluation.rag_eval"]),
    ("Groundedness classifier spot-check", [sys.executable, "-m", "evaluation.groundedness_eval"]),
    ("Baseline vs agentic benchmark", [sys.executable, "-m", "evaluation.benchmark"]),
]


def main() -> None:
    failures = []
    for name, cmd in STEPS:
        print(f"\n=== {name} ===")
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            failures.append(name)

    print("\n=== Summary ===")
    if failures:
        print(f"{len(failures)} step(s) failed: {failures}")
        sys.exit(1)
    print("All evaluation steps completed. See evaluation/results/ for reports.")


if __name__ == "__main__":
    main()
