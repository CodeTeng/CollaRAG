"""End-to-end smoke test: ingest + 1 query against DeepSeek.

Uses a tiny 5-paragraph / 3-question slice so it's cheap to run
(< 30 DeepSeek calls) and surfaces any wiring issue quickly.

Run::

    uv run python scripts/smoke_deepseek.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chimera_rag import ChimeraRAG  # noqa: E402
from chimera_rag.core.types import Document, Query  # noqa: E402


def main() -> int:
    # Use the intent-only config — it uses defaults for ingestion so the
    # LLM call count is minimal (one extraction per paragraph).
    config_path = PROJECT_ROOT / "configs" / "colla_rag_only.yaml"
    crag = ChimeraRAG.from_config(str(config_path))

    # Tiny corpus: 5 paragraphs hand-picked from the mock wiki dataset.
    docs = [
        Document(
            doc_id="smk-0",
            content="Ada Verdant is a researcher based in London. "
            "Ada Verdant works at Lumentech Corp, where they lead research on machine learning.",
        ),
        Document(
            doc_id="smk-1",
            content="Lumentech Corp is a technology company headquartered in London. "
            "Lumentech Corp was founded in 1995. "
            "In 2021, Lumentech Corp was acquired by Helios Systems.",
        ),
        Document(
            doc_id="smk-2",
            content="Bruno Falk is a researcher based in Berlin. "
            "Bruno Falk works at Helios Systems, where they lead research on quantum computing.",
        ),
        Document(
            doc_id="smk-3",
            content="Helios Systems is an independent technology company headquartered in Berlin.",
        ),
        Document(
            doc_id="smk-4",
            content="Cara Solis is a researcher based in Barcelona. "
            "Cara Solis works at Nimbus Labs, where they lead research on cryptography.",
        ),
    ]

    print("=" * 60)
    print(f"config: {config_path.name}")
    print(f"active: {json.dumps(crag.active_implementations(), indent=2)}")
    print("=" * 60)

    t0 = time.time()
    stats = crag.ingest(docs)
    print(f"\n[ingest] done in {time.time() - t0:.1f}s  stats={stats}")
    print(f"[state]  {crag.stats()}")

    questions = [
        "Where does Ada Verdant work?",
        "Which company acquired the employer of Ada Verdant?",
        "What field does Bruno Falk research?",
    ]
    for q in questions:
        print(f"\n[query] {q}")
        t0 = time.time()
        ans = crag.query(Query(text=q))
        print(f"  -> {ans.text!r}")
        print(f"     intent={ans.intent_label}  "
              f"confidence={ans.confidence}  "
              f"evidence={ans.evidence_chunk_ids}  "
              f"latency={time.time() - t0:.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
