"""End-to-end integration test for the PrismRAG plugin.

Toggles ``plugins.prism.enabled: true`` and routes the chunker slot to
``prism.chunker``. PrismRAG owns **only** the chunker slot — extractor and
pruner stay on ``defaults.*`` so that prism_only vs vanilla isolates the
chunking method as the single variable.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def prism_config(mock_prism_only_yaml: str) -> str:
    return mock_prism_only_yaml


def test_prism_plugin_activates_when_enabled(prism_config: Path):
    from chimera_rag import ChimeraRAG

    crag = ChimeraRAG.from_config(prism_config)
    active = crag.active_implementations()
    assert active["chunker"] == "prism.chunker"
    # PrismRAG is chunker-only; the other ingest slots stay on defaults.
    assert active["extractor"] == "defaults.simple_llm"
    assert active["pruner"] == "defaults.noop"
    # Query-side slots too (prism is ingest-only).
    assert active["retriever"].startswith("defaults.")
    assert active["generator"].startswith("defaults.")


def test_prism_end_to_end_ingest_then_query(prism_config: Path):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document, Query

    crag = ChimeraRAG.from_config(prism_config)
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


def test_prism_chunker_produces_metadata_visible_in_chunks(prism_config: Path):
    """PrismChunker should tag chunks with adaptive_chunk_size et al."""
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document

    crag = ChimeraRAG.from_config(prism_config)
    crag.ingest([Document(doc_id="d", content="Alfred Nobel invented dynamite in 1867.")])
    any_chunk = next(iter(crag.state.chunk_lookup.values()))
    assert "adaptive_chunk_size" in any_chunk.metadata
