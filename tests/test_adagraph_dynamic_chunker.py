"""Tests for :mod:`chimera_rag.plugins.adagraph.dynamic_chunker`.

AdaGraph's dynamic chunker adapts chunk size per-document based on two
signals:

* **complexity** in [0, 1] — syntactic + lexical diversity + length variance
* **entity density** in [0, 1] — estimated entities / tokens

The adaptive size formula (from the design doc) is::

    size = base - (score - 0.5) * range
    where  score = (complexity + density) / 2
    and    range = max_chunk_size - min_chunk_size

so a more complex / entity-dense document yields *smaller* chunks
(more context per retrieval), while simple prose gets larger chunks
(fewer API calls).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# compute_adaptive_chunk_size: the core formula
# ---------------------------------------------------------------------------
def test_adaptive_size_score_half_returns_base_size():
    """score=0.5 is the neutral point: output equals the base."""
    from chimera_rag.plugins.adagraph.dynamic_chunker import compute_adaptive_chunk_size

    out = compute_adaptive_chunk_size(score=0.5, min_size=100, max_size=500, base_size=300)
    assert out == 300


def test_adaptive_size_high_score_yields_smaller_chunks():
    """Complex / dense text -> smaller chunks than base."""
    from chimera_rag.plugins.adagraph.dynamic_chunker import compute_adaptive_chunk_size

    out = compute_adaptive_chunk_size(score=1.0, min_size=100, max_size=500, base_size=300)
    # base - (1.0 - 0.5) * (500 - 100) = 300 - 200 = 100
    assert out == 100


def test_adaptive_size_low_score_yields_larger_chunks():
    """Simple prose -> larger chunks than base."""
    from chimera_rag.plugins.adagraph.dynamic_chunker import compute_adaptive_chunk_size

    out = compute_adaptive_chunk_size(score=0.0, min_size=100, max_size=500, base_size=300)
    # base - (0.0 - 0.5) * (500 - 100) = 300 + 200 = 500
    assert out == 500


def test_adaptive_size_clamped_within_min_max_bounds():
    """Even with out-of-range scores, output must be clamped."""
    from chimera_rag.plugins.adagraph.dynamic_chunker import compute_adaptive_chunk_size

    assert (
        compute_adaptive_chunk_size(score=2.0, min_size=100, max_size=500, base_size=300)
        == 100
    )
    assert (
        compute_adaptive_chunk_size(score=-1.0, min_size=100, max_size=500, base_size=300)
        == 500
    )


# ---------------------------------------------------------------------------
# DynamicChunker.chunk end-to-end
# ---------------------------------------------------------------------------
def test_dynamic_chunker_implements_base_chunker_abc():
    from chimera_rag.interfaces.chunker import BaseChunker
    from chimera_rag.plugins.adagraph.dynamic_chunker import DynamicChunker

    assert isinstance(DynamicChunker(), BaseChunker)


def test_dynamic_chunker_returns_nonempty_chunks_for_nonempty_document():
    from chimera_rag.core.types import Document
    from chimera_rag.plugins.adagraph.dynamic_chunker import DynamicChunker

    text = (
        "Alfred Nobel was a Swedish chemist. He invented dynamite in 1867. "
        "Nobel later founded the Nobel Prize in 1901. "
    ) * 10
    chunker = DynamicChunker(min_chunk_size=100, max_chunk_size=500, base_chunk_size=300)

    chunks = chunker.chunk(Document(doc_id="nobel", content=text))
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "nobel"
    assert chunks[0].chunk_id.startswith("nobel::")
    # Adaptive size must fall within [min, max].
    for ch in chunks:
        assert len(ch.text) <= 500 + 50   # allow small slack for sentence-ending
        assert len(ch.text) > 0


def test_dynamic_chunker_records_adaptive_size_in_metadata():
    """Traceability: downstream debugging should see which size was picked."""
    from chimera_rag.core.types import Document
    from chimera_rag.plugins.adagraph.dynamic_chunker import DynamicChunker

    chunker = DynamicChunker()
    chunks = chunker.chunk(Document(doc_id="d", content="hello. world."))
    assert chunks
    assert "adaptive_chunk_size" in chunks[0].metadata


def test_dynamic_chunker_empty_document_returns_empty_list():
    from chimera_rag.core.types import Document
    from chimera_rag.plugins.adagraph.dynamic_chunker import DynamicChunker

    assert DynamicChunker().chunk(Document(doc_id="d", content="")) == []


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_dynamic_chunker_registers_under_adagraph_slot():
    import chimera_rag.plugins.adagraph  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.adagraph.dynamic_chunker import DynamicChunker

    assert get_registry().get("chunker", "adagraph.dynamic") is DynamicChunker
