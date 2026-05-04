"""End-to-end integration test for the AdaGraph plugin.

Toggles ``plugins.adagraph.enabled: true`` and routes all three ingestion
slots to ``adagraph.*`` implementations. The test verifies that:

* ChimeraRAG wires up AdaGraph components when the plugin is enabled,
* ingest + query complete without errors,
* the active implementations reported include "adagraph." prefixes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def adagraph_config(mock_adagraph_only_yaml: str) -> str:
    return mock_adagraph_only_yaml
def test_adagraph_plugin_activates_when_enabled(adagraph_config: Path):
    from chimera_rag import ChimeraRAG

    crag = ChimeraRAG.from_config(adagraph_config)
    active = crag.active_implementations()
    assert active["chunker"] == "adagraph.dynamic"
    assert active["extractor"] == "adagraph.layered"
    assert active["pruner"] == "adagraph.dual_redundancy"
    # query-side slots still come from defaults (adagraph is ingest-only).
    assert active["retriever"].startswith("defaults.")
    assert active["generator"].startswith("defaults.")


def test_adagraph_end_to_end_ingest_then_query(adagraph_config: Path):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document, Query

    crag = ChimeraRAG.from_config(adagraph_config)
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
    stats = crag.stats()
    assert stats["chunks"] >= 1

    ans = crag.query(Query(text="Who invented dynamite?"))
    assert isinstance(ans.text, str)


def test_adagraph_chunker_produces_metadata_visible_in_chunks(adagraph_config: Path):
    """DynamicChunker should tag chunks with adaptive_chunk_size et al."""
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document

    crag = ChimeraRAG.from_config(adagraph_config)
    crag.ingest([Document(doc_id="d", content="Alfred Nobel invented dynamite in 1867.")])
    any_chunk = next(iter(crag.state.chunk_lookup.values()))
    assert "adaptive_chunk_size" in any_chunk.metadata
