"""W1 — granularity routing in ``IngestionPipeline``.

Guards the contract in ``docs/innovation-1/design.md`` §4.1::

    granularity | extractor | vector_store | chunk_lookup
    ------------+-----------+--------------+-------------
    unified     |    yes    |     yes      |     yes
    fine        |    yes    |     no       |     yes
    coarse      |    no     |     yes      |     yes

``unified`` is the default value, so every pre-existing chunker keeps its
exact behaviour and the ablation baselines stay fair (§4.2). The tests
below pin both halves: the new routing, and the unchanged legacy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from chimera_rag.core.pipeline import IngestionPipeline, PipelineState
from chimera_rag.core.types import Chunk, Document, Triple
from chimera_rag.defaults.pruner import NoOpPruner
from chimera_rag.interfaces.chunker import BaseChunker
from chimera_rag.interfaces.extractor import BaseTripleExtractor
from chimera_rag.interfaces.graph_store import BaseGraphStore
from chimera_rag.interfaces.vector_store import BaseVectorStore


# ---------------------------------------------------------------------------
# Recording doubles
# ---------------------------------------------------------------------------
class ScriptedChunker(BaseChunker):
    """Returns a caller-supplied chunk list verbatim."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def chunk(self, document: Document) -> list[Chunk]:
        return list(self._chunks)


class RecordingExtractor(BaseTripleExtractor):
    """Records which chunks it was asked to extract, one triple per chunk."""

    def __init__(self) -> None:
        self.seen_chunk_ids: list[str] = []

    def extract(self, chunk: Chunk) -> list[Triple]:
        self.seen_chunk_ids.append(chunk.chunk_id)
        return [
            Triple(
                subject=f"s::{chunk.chunk_id}",
                predicate="mentions",
                object=chunk.text[:12] or "empty",
                source_chunk_id=chunk.chunk_id,
            )
        ]


class RecordingVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.added_ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        self.added_ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        return []

    def persist(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


class RecordingGraphStore(BaseGraphStore):
    def __init__(self) -> None:
        self.triples: list[Triple] = []

    def add_triple(self, triple: Triple) -> None:
        self.triples.append(triple)

    def get_neighbors(self, entity: str, hops: int = 1) -> list[Triple]:
        return []

    def all_triples(self) -> list[Triple]:
        return list(self.triples)

    def persist(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


class FakeEmbedder:
    """Minimal stand-in: dimension only matters for shape, not content."""

    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.ones((len(texts), self.dim), dtype=np.float32)


def _run(chunks: list[Chunk]) -> tuple[IngestionPipeline, RecordingExtractor, RecordingVectorStore]:
    extractor = RecordingExtractor()
    vectors = RecordingVectorStore()
    graph = RecordingGraphStore()
    state = PipelineState(graph_store=graph, vector_store=vectors)
    pipe = IngestionPipeline(
        chunker=ScriptedChunker(chunks),
        extractor=extractor,
        pruner=NoOpPruner(),
        embedder=FakeEmbedder(),
        state=state,
        show_progress=False,
    )
    pipe.run([Document(doc_id="d1", content="irrelevant; chunks are scripted")])
    return pipe, extractor, vectors


def _mk(chunk_id: str, granularity: str, *, parent: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        index=0,
        text=f"text of {chunk_id}",
        granularity=granularity,
        parent_chunk_id=parent,
    )


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------
def test_chunk_defaults_to_unified_granularity():
    """Backward compatibility: existing chunkers build Chunk without the new
    fields and must get ``unified`` / ``None``."""
    ch = Chunk(chunk_id="d1::0", doc_id="d1", index=0, text="hello")
    assert ch.granularity == "unified"
    assert ch.parent_chunk_id is None


def test_fine_chunk_accepts_a_parent():
    ch = _mk("d1::fine::0", "fine", parent="d1::coarse::0")
    assert ch.parent_chunk_id == "d1::coarse::0"


@pytest.mark.parametrize("granularity", ["unified", "coarse"])
def test_only_fine_chunks_may_carry_a_parent(granularity: str):
    """``coarse`` is the top of the provenance chain and ``unified`` has no
    chain at all; a parent on either would silently corrupt contribution 1."""
    with pytest.raises(ValueError, match="parent_chunk_id"):
        _mk("d1::x::0", granularity, parent="d1::coarse::0")


def test_unknown_granularity_is_rejected():
    with pytest.raises(ValueError):
        _mk("d1::x::0", "medium")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_unified_chunk_is_both_extracted_and_indexed():
    _, extractor, vectors = _run([_mk("d1::0", "unified")])
    assert extractor.seen_chunk_ids == ["d1::0"]
    assert vectors.added_ids == ["d1::0"]


def test_fine_chunk_is_extracted_but_not_indexed():
    _, extractor, vectors = _run([_mk("d1::fine::0", "fine")])
    assert extractor.seen_chunk_ids == ["d1::fine::0"]
    assert vectors.added_ids == []


def test_coarse_chunk_is_indexed_but_not_extracted():
    _, extractor, vectors = _run([_mk("d1::coarse::0", "coarse")])
    assert extractor.seen_chunk_ids == []
    assert vectors.added_ids == ["d1::coarse::0"]


def test_every_granularity_lands_in_chunk_lookup():
    chunks = [
        _mk("d1::coarse::0", "coarse"),
        _mk("d1::fine::0", "fine", parent="d1::coarse::0"),
        _mk("d1::fine::1", "fine", parent="d1::coarse::0"),
    ]
    pipe, _, _ = _run(chunks)
    assert set(pipe.state.chunk_lookup) == {c.chunk_id for c in chunks}


def test_triples_come_only_from_extracted_granularities():
    """The coarse chunk must contribute no triple, so the graph holds exactly
    the fine + unified output."""
    chunks = [
        _mk("d1::coarse::0", "coarse"),
        _mk("d1::fine::0", "fine", parent="d1::coarse::0"),
        _mk("d1::unified::0", "unified"),
    ]
    pipe, _, _ = _run(chunks)
    assert {t.source_chunk_id for t in pipe.state.graph_store.all_triples()} == {
        "d1::fine::0",
        "d1::unified::0",
    }


def test_coarse_only_document_never_calls_the_extractor():
    _, extractor, vectors = _run([_mk("d1::coarse::0", "coarse")])
    assert extractor.seen_chunk_ids == []
    assert vectors.added_ids == ["d1::coarse::0"]


def test_extraction_order_follows_chunk_order():
    """Legacy guarantee: triples are concatenated in chunk order, not in
    completion order."""
    chunks = [_mk(f"d1::{i}", "unified") for i in range(5)]
    pipe, extractor, _ = _run(chunks)
    assert extractor.seen_chunk_ids == [f"d1::{i}" for i in range(5)]
    assert [t.source_chunk_id for t in pipe.state.graph_store.all_triples()] == [
        f"d1::{i}" for i in range(5)
    ]


# ---------------------------------------------------------------------------
# Legacy regression: defaults.fixed must be untouched
# ---------------------------------------------------------------------------
def test_legacy_fixed_chunker_routes_every_chunk_to_both_sides():
    """§4.2: because ``granularity`` defaults to ``unified``, the vanilla
    baseline's extractor and vector-store inputs are the same set as before
    the routing change — ablation fairness is enforced by the type system."""
    from chimera_rag.defaults.chunker import FixedSizeChunker

    doc = Document(doc_id="d1", content="A" * 700 + " " + "B" * 700)
    chunks = FixedSizeChunker(chunk_size=300, overlap=50).chunk(doc)
    assert chunks and all(c.granularity == "unified" for c in chunks)

    extractor = RecordingExtractor()
    vectors = RecordingVectorStore()
    pipe = IngestionPipeline(
        chunker=ScriptedChunker(chunks),
        extractor=extractor,
        pruner=NoOpPruner(),
        embedder=FakeEmbedder(),
        state=PipelineState(graph_store=RecordingGraphStore(), vector_store=vectors),
        show_progress=False,
    )
    pipe.run([doc])

    assert extractor.seen_chunk_ids == [c.chunk_id for c in chunks]
    assert vectors.added_ids == [c.chunk_id for c in chunks]
    assert len(pipe.state.chunk_lookup) == len(chunks)
