"""End-to-end integration test for the IntentAgent plugin."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def intent_agent_config(tmp_path: Path) -> Path:
    raw = (REPO / "configs" / "intent_agent_only.yaml").read_text()
    raw = raw.replace(
        "./storage/intent_agent_only",
        str(tmp_path / "storage").replace("\\", "/"),
    )
    p = tmp_path / "config.yaml"
    p.write_text(raw)
    return p


def test_intent_agent_plugin_activates_when_enabled(intent_agent_config: Path):
    from chimera_rag import ChimeraRAG

    crag = ChimeraRAG.from_config(intent_agent_config)
    active = crag.active_implementations()
    # Ingestion side stays on defaults in intent_agent_only.
    assert active["chunker"].startswith("defaults.")
    assert active["extractor"].startswith("defaults.")
    assert active["pruner"].startswith("defaults.")
    # Query side is taken over by intent_agent.
    assert active["intent_classifier"] == "intent_agent.llm"
    assert active["retriever"] == "intent_agent.strategy_selector"


def test_intent_agent_end_to_end_ingest_then_query(intent_agent_config: Path):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document, Query

    crag = ChimeraRAG.from_config(intent_agent_config)
    crag.ingest(
        [
            Document(
                doc_id="nobel",
                content=(
                    "Alfred Nobel was a Swedish chemist. "
                    "Nobel invented dynamite in 1867. "
                    "Einstein formulated the theory of relativity."
                ),
            )
        ]
    )
    ans = crag.query(Query(text="Compare Nobel and Einstein"))
    assert isinstance(ans.text, str)
    # intent_label must have been populated by the agent.
    assert ans.intent_label in {
        "factual", "analytical", "comparative",
        "multi_hop", "exploratory", "follow_up",
    }
