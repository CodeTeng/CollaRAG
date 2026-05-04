"""Demo: vanilla KG-RAG baseline (both plugins disabled).

Run::

    uv run python examples/demo_vanilla.py

Uses only MockProviders so no API key is needed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from chimera_rag import ChimeraRAG
from chimera_rag.core.types import Document, Query

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    # Workspace: honour CHIMERA_WORKSPACE (set by tests) else a fresh tmp dir.
    workspace = Path(os.environ.get("CHIMERA_WORKSPACE") or tempfile.mkdtemp(prefix="chimera-"))
    workspace.mkdir(parents=True, exist_ok=True)

    # Patch the vanilla config's workspace_dir on the fly.
    raw = (REPO / "configs" / "vanilla.yaml").read_text()
    raw = raw.replace("./storage/vanilla", str(workspace / "storage"))
    cfg_path = workspace / "config.yaml"
    cfg_path.write_text(raw)

    crag = ChimeraRAG.from_config(cfg_path)

    print("=== VANILLA KG-RAG ===")
    print(f"active implementations: {crag.active_implementations()}")

    crag.ingest(
        [
            Document(
                doc_id="nobel",
                content=(
                    "Alfred Nobel was a Swedish chemist. "
                    "He invented dynamite in 1867. "
                    "He later founded the Nobel Prize."
                ),
            )
        ]
    )
    print(f"ingested: {crag.stats()}")

    for q in ["Who invented dynamite?", "When was the Nobel Prize founded?"]:
        ans = crag.query(Query(text=q))
        print(f"\nQ: {q}")
        print(f"  intent={ans.intent_label}")
        print(f"  answer={ans.text}")
        print(f"  evidence_chunks={ans.evidence_chunk_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
