"""Demo: both innovations enabled (full configuration).

This is the intended thesis-grade setup: AdaGraph on ingestion,
IntentAgent on query.

Run::

    uv run python examples/demo_full.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from chimera_rag import ChimeraRAG
from chimera_rag.core.types import Document, Query

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    workspace = Path(os.environ.get("CHIMERA_WORKSPACE") or tempfile.mkdtemp(prefix="chimera-"))
    workspace.mkdir(parents=True, exist_ok=True)

    raw = (REPO / "configs" / "full.yaml").read_text()
    raw = raw.replace("./storage/full", str(workspace / "storage"))
    cfg_path = workspace / "config.yaml"
    cfg_path.write_text(raw)

    crag = ChimeraRAG.from_config(cfg_path)
    print("=== FULL (AdaGraph + IntentAgent) ===")
    print(f"active implementations: {crag.active_implementations()}")

    crag.ingest(
        [
            Document(
                doc_id="nobel",
                content=(
                    "Alfred Nobel was a Swedish chemist and the inventor of dynamite. "
                    "He was born in Stockholm in 1833. "
                    "Later in life, Nobel established the Nobel Prize, first awarded in 1901. "
                    "The prize honours achievements in physics, chemistry, medicine, "
                    "literature and peace."
                ),
            )
        ]
    )

    for q in [
        "Who invented dynamite?",
        "Compare Nobel and Einstein",
        "What is the impact of dynamite on mining?",
    ]:
        ans = crag.query(Query(text=q))
        strategy = ans.trace.get("retrieval_metadata", {}).get("strategy")
        print(f"\nQ: {q}")
        print(f"  intent={ans.intent_label}  strategy={strategy}")
        print(f"  chunks={len(ans.evidence_chunk_ids)}")
        print(f"  answer={ans.text[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
