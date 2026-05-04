"""Demo: AdaGraph (innovation #1) only.

Activates adaptive chunking + layered extraction + dual redundancy
pruning on the ingestion side. Query side uses vanilla defaults so
differences are fully attributable to AdaGraph.

Run::

    uv run python examples/demo_adagraph.py
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

    raw = (REPO / "configs" / "mock_adagraph_only.yaml").read_text()
    raw = raw.replace("./storage/adagraph_only", str(workspace / "storage"))
    cfg_path = workspace / "config.yaml"
    cfg_path.write_text(raw)

    crag = ChimeraRAG.from_config(cfg_path)
    print("=== ADAGRAPH ONLY ===")
    print(f"active implementations: {crag.active_implementations()}")

    # A denser, longer document so the adaptive chunk size is visible.
    doc = Document(
        doc_id="nobel-long",
        content=(
            "Alfred Nobel was a Swedish chemist, engineer, inventor and businessman. "
            "He invented dynamite in 1867 and registered the patent in Sweden. "
            "Nobel later founded the Nobel Prize in 1895 via his last will. "
            "The prize is awarded in six categories and is administered in Stockholm and Oslo. "
            "Nobel also held hundreds of other patents throughout his lifetime."
        ),
    )
    crag.ingest([doc])
    print(f"ingested: {crag.stats()}")

    # Inspect AdaGraph metadata on produced chunks.
    for cid, ch in list(crag.state.chunk_lookup.items())[:3]:
        print(f"\nchunk {cid}:")
        print(f"  adaptive_chunk_size={ch.metadata.get('adaptive_chunk_size')}")
        print(f"  complexity={ch.metadata.get('complexity')}")
        print(f"  density={ch.metadata.get('density')}")

    ans = crag.query(Query(text="Who invented dynamite?"))
    print(f"\nQ: Who invented dynamite?")
    print(f"  answer={ans.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
