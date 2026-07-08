"""End-to-end integration test for the CollaRAG plugin."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def colla_rag_config(mock_colla_rag_only_yaml: str) -> str:
    return mock_colla_rag_only_yaml
def test_colla_rag_plugin_activates_when_enabled(colla_rag_config: Path):
    from chimera_rag import ChimeraRAG

    crag = ChimeraRAG.from_config(colla_rag_config)
    active = crag.active_implementations()
    # Ingestion side stays on defaults in colla_rag_only.
    assert active["chunker"].startswith("defaults.")
    assert active["extractor"].startswith("defaults.")
    assert active["pruner"].startswith("defaults.")
    # Query side is taken over by colla_rag.
    assert active["intent_classifier"] == "colla_rag.tree"
    assert active["retriever"] == "colla_rag.multi_agent"


def test_colla_rag_end_to_end_ingest_then_query(colla_rag_config: Path):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document, Query

    crag = ChimeraRAG.from_config(colla_rag_config)
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
        "greeting", "single_hop", "multi_hop", "summarization", "other",
    }
