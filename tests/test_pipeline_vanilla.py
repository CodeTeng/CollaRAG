"""End-to-end smoke test for the vanilla pipeline.

Assembles IngestionPipeline + QueryPipeline (or the top-level ChimeraRAG
facade) using only defaults and Mock providers, then ingests a tiny
document and asks a question. If this stays green we know:

* @register decorators fired for all 9 defaults
* Registry lookup by (slot, active) works end-to-end
* Chunker -> Extractor -> Pruner -> Graph/Vector stores pipeline is wired
* Retriever -> Generator pipeline produces a traceable Answer

All LLM/Embedding calls go through Mock, so no API key is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def vanilla_config(mock_vanilla_yaml: str, tmp_path: Path) -> Path:
    """Return a vanilla config whose workspace_dir is under tmp_path."""
    from chimera_rag.core.config import load_config  # noqa: F401

    repo = Path(__file__).resolve().parent.parent
    raw = Path(mock_vanilla_yaml).read_text()
    raw = raw.replace(
        "./storage/vanilla",
        str(tmp_path / "storage").replace("\\", "/"),
    )
    p = tmp_path / "config.yaml"
    p.write_text(raw)
    return p


def test_chimera_rag_vanilla_ingest_then_query(vanilla_config: Path):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document, Query

    crag = ChimeraRAG.from_config(vanilla_config)
    crag.ingest(
        [
            Document(
                doc_id="nobel",
                content=(
                    "Alfred Nobel was a Swedish chemist. "
                    "Nobel invented dynamite in 1867. "
                    "Nobel founded the Nobel Prize."
                ),
            )
        ]
    )
    # Graph should now contain some triples (MockLLMProvider returns default
    # JSON; extractor will parse whatever the mock emits).
    ans = crag.query(Query(text="Who invented dynamite?"))

    # Answer must be a non-empty string and carry evidence ids.
    assert isinstance(ans.text, str) and len(ans.text) > 0
    assert isinstance(ans.evidence_chunk_ids, list)


def test_chimera_rag_state_persisted_between_ingest_and_query(vanilla_config: Path):
    """The graph/vector stores must survive across separate method calls."""
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document, Query

    crag = ChimeraRAG.from_config(vanilla_config)
    crag.ingest([Document(doc_id="x", content="Einstein formulated relativity.")])

    stats = crag.stats()
    assert stats["chunks"] >= 1
    # Querying should return a RetrievalResult with at least one chunk.
    ans = crag.query(Query(text="relativity"))
    assert len(ans.evidence_chunk_ids) >= 1


def test_plugin_disabled_means_defaults_are_used(vanilla_config: Path):
    """With mock_vanilla.yaml, plugins.adagraph.enabled and plugins.colla_rag.enabled
    are both False; the active implementations must come from defaults.*."""
    from chimera_rag import ChimeraRAG

    crag = ChimeraRAG.from_config(vanilla_config)
    # The pipeline exposes its active slot implementations under .active.
    active = crag.active_implementations()
    assert active["chunker"].startswith("defaults.")
    assert active["extractor"].startswith("defaults.")
    assert active["retriever"].startswith("defaults.")
    assert active["generator"].startswith("defaults.")
