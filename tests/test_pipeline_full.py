"""End-to-end integration test with both AdaGraph and IntentAgent enabled."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def full_config(mock_full_yaml: str) -> str:
    return mock_full_yaml
def test_full_config_activates_both_plugins_across_all_slots(full_config: Path):
    from chimera_rag import ChimeraRAG

    crag = ChimeraRAG.from_config(full_config)
    active = crag.active_implementations()
    # ingestion side -> AdaGraph
    assert active["chunker"].startswith("adagraph.")
    assert active["extractor"].startswith("adagraph.")
    assert active["pruner"].startswith("adagraph.")
    # query side -> IntentAgent
    assert active["intent_classifier"] == "intent_agent.llm"
    assert active["retriever"] == "intent_agent.strategy_selector"


def test_full_pipeline_ingest_then_query_with_both_plugins(full_config: Path):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document, Query

    crag = ChimeraRAG.from_config(full_config)
    crag.ingest(
        [
            Document(
                doc_id="nobel",
                content=(
                    "Alfred Nobel was a Swedish chemist. "
                    "Nobel invented dynamite in 1867. "
                    "Nobel founded the Nobel Prize in 1901."
                ),
            )
        ]
    )
    ans = crag.query(Query(text="Who invented dynamite?"))
    assert isinstance(ans.text, str)
    # intent is populated by IntentAgent
    assert ans.intent_label in {
        "factual", "analytical", "comparative",
        "multi_hop", "exploratory", "follow_up",
    }


def test_full_pipeline_reports_adaptive_chunk_size_via_adagraph(full_config: Path):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document

    crag = ChimeraRAG.from_config(full_config)
    crag.ingest([Document(doc_id="d", content="Alfred Nobel invented dynamite.")])
    chunk = next(iter(crag.state.chunk_lookup.values()))
    # AdaGraph's DynamicChunker stamps metadata on every chunk it emits.
    assert "adaptive_chunk_size" in chunk.metadata
