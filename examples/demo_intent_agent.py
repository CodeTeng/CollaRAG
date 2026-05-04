"""Demo: IntentAgent (innovation #2) only.

Shows how intent classification routes different queries to different
strategies. Ingestion side uses vanilla defaults.

Run::

    uv run python examples/demo_intent_agent.py
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

    raw = (REPO / "configs" / "mock_intent_only.yaml").read_text()
    raw = raw.replace("./storage/intent_agent_only", str(workspace / "storage"))
    cfg_path = workspace / "config.yaml"
    cfg_path.write_text(raw)

    crag = ChimeraRAG.from_config(cfg_path)
    print("=== INTENTAGENT ONLY ===")
    print(f"active implementations: {crag.active_implementations()}")

    crag.ingest(
        [
            Document(
                doc_id="history",
                content=(
                    "Alfred Nobel invented dynamite. "
                    "Albert Einstein formulated relativity. "
                    "Marie Curie discovered radium."
                ),
            )
        ]
    )

    queries = [
        "Who invented dynamite?",                   # factual -> direct
        "Compare Nobel and Einstein",               # comparative -> parallel
        "Why was dynamite important?",              # analytical -> iterative
        "What is the relation between Nobel and Einstein?",  # multi_hop
        "Tell me about Curie",                      # exploratory -> expansion
    ]
    for q in queries:
        ans = crag.query(Query(text=q))
        strategy = ans.trace.get("retrieval_metadata", {}).get("strategy")
        print(f"\nQ: {q}")
        print(f"  intent={ans.intent_label}  strategy={strategy}")
        print(f"  answer={ans.text[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
