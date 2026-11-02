"""Tests for :mod:`chimera_rag.plugins.prism.chunker` and ``prism.features``.

The chunker adapts its target size per document from two zero-dependency
signals:

* **complexity** in [0, 1] — lexical diversity + sentence-length variance
  + average word length
* **entity density** in [0, 1] — capitalised tokens / all tokens, with
  sentence-initial capitals discounted

The adaptive size formula is::

    size = base - (score - 0.5) * (max_chunk_size - min_chunk_size)
    where  score = complexity_weight * complexity + density_weight * density

so a complex / entity-dense document yields *smaller* chunks while simple
prose gets larger ones.

⚠️ This is PrismRAG's **transitional single-granularity** implementation.
Dual-granularity output (fine / coarse + ``parent_chunk_id``) and the
typed-defect audit loop are not covered here — see
``docs/innovation-1/design.md``.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# compute_adaptive_chunk_size: the core formula
# ---------------------------------------------------------------------------
def test_adaptive_size_score_half_returns_base_size():
    """score=0.5 is the neutral point: output equals the base."""
    from chimera_rag.plugins.prism.chunker import compute_adaptive_chunk_size

    out = compute_adaptive_chunk_size(score=0.5, min_size=100, max_size=500, base_size=300)
    assert out == 300


def test_adaptive_size_high_score_yields_smaller_chunks():
    """Complex / dense text -> smaller chunks than base."""
    from chimera_rag.plugins.prism.chunker import compute_adaptive_chunk_size

    out = compute_adaptive_chunk_size(score=1.0, min_size=100, max_size=500, base_size=300)
    # base - (1.0 - 0.5) * (500 - 100) = 300 - 200 = 100
    assert out == 100


def test_adaptive_size_low_score_yields_larger_chunks():
    """Simple prose -> larger chunks than base."""
    from chimera_rag.plugins.prism.chunker import compute_adaptive_chunk_size

    out = compute_adaptive_chunk_size(score=0.0, min_size=100, max_size=500, base_size=300)
    # base - (0.0 - 0.5) * (500 - 100) = 300 + 200 = 500
    assert out == 500


def test_adaptive_size_clamped_within_min_max_bounds():
    """Even with out-of-range scores, output must be clamped."""
    from chimera_rag.plugins.prism.chunker import compute_adaptive_chunk_size

    assert compute_adaptive_chunk_size(score=2.0, min_size=100, max_size=500, base_size=300) == 100
    assert compute_adaptive_chunk_size(score=-1.0, min_size=100, max_size=500, base_size=300) == 500


def test_adaptive_size_rejects_inverted_bounds():
    from chimera_rag.plugins.prism.chunker import compute_adaptive_chunk_size

    with pytest.raises(ValueError, match="min_size must be <= max_size"):
        compute_adaptive_chunk_size(score=0.5, min_size=500, max_size=100, base_size=300)


# ---------------------------------------------------------------------------
# PrismChunker.chunk end-to-end
# ---------------------------------------------------------------------------
def test_prism_chunker_implements_base_chunker_abc():
    from chimera_rag.interfaces.chunker import BaseChunker
    from chimera_rag.plugins.prism.chunker import PrismChunker

    assert isinstance(PrismChunker(), BaseChunker)


def test_prism_chunker_returns_nonempty_chunks_for_nonempty_document():
    from chimera_rag.core.types import Document
    from chimera_rag.plugins.prism.chunker import PrismChunker

    text = (
        "Alfred Nobel was a Swedish chemist. He invented dynamite in 1867. "
        "Nobel later founded the Nobel Prize in 1901. "
    ) * 10
    chunker = PrismChunker(min_chunk_size=100, max_chunk_size=500, base_chunk_size=300)

    chunks = chunker.chunk(Document(doc_id="nobel", content=text))
    assert len(chunks) >= 1
    assert chunks[0].doc_id == "nobel"
    assert chunks[0].chunk_id.startswith("nobel::")
    for ch in chunks:
        assert len(ch.text) <= 500 + 50  # slack for sentence-ending overflow
        assert len(ch.text) > 0


def test_prism_chunker_records_adaptive_size_in_metadata():
    """Traceability: downstream debugging should see which size was picked."""
    from chimera_rag.core.types import Document
    from chimera_rag.plugins.prism.chunker import PrismChunker

    chunks = PrismChunker().chunk(Document(doc_id="d", content="hello. world."))
    assert chunks
    assert "adaptive_chunk_size" in chunks[0].metadata


def test_prism_chunker_empty_document_returns_empty_list():
    from chimera_rag.core.types import Document
    from chimera_rag.plugins.prism.chunker import PrismChunker

    assert PrismChunker().chunk(Document(doc_id="d", content="")) == []


def test_prism_chunker_accepts_defaults_style_chunk_size_param():
    """``chunk_size`` (the defaults.fixed knob) maps onto ``base_chunk_size``
    so a slot toggle never raises on a foreign param name."""
    from chimera_rag.plugins.prism.chunker import PrismChunker

    assert PrismChunker(chunk_size=420).base_chunk_size == 420


def test_prism_chunker_rejects_inconsistent_size_bounds():
    from chimera_rag.plugins.prism.chunker import PrismChunker

    with pytest.raises(ValueError, match="min_chunk_size <= base_chunk_size <= max_chunk_size"):
        PrismChunker(min_chunk_size=400, base_chunk_size=100, max_chunk_size=500)


def test_prism_chunker_rejects_weights_that_do_not_sum_to_one():
    from chimera_rag.plugins.prism.chunker import PrismChunker

    with pytest.raises(ValueError, match="must sum to 1.0"):
        PrismChunker(complexity_weight=0.7, density_weight=0.7)


def test_prism_chunker_never_splits_a_sentence():
    """Sentence-level packing must not cut mid-sentence the way a character
    window would."""
    from chimera_rag.core.types import Document
    from chimera_rag.plugins.prism.chunker import PrismChunker

    text = "Alfred Nobel invented dynamite. " * 20
    chunker = PrismChunker(min_chunk_size=60, max_chunk_size=60, base_chunk_size=60)

    for ch in chunker.chunk(Document(doc_id="d", content=text)):
        assert ch.text.endswith(".")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_prism_chunker_registers_under_prism_slot():
    import chimera_rag.plugins.prism  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.prism.chunker import PrismChunker

    assert get_registry().get("chunker", "prism.chunker") is PrismChunker


# ---------------------------------------------------------------------------
# features: the two internal signals
# ---------------------------------------------------------------------------
def test_estimate_complexity_is_bounded():
    from chimera_rag.plugins.prism.features import estimate_complexity

    for text in ["", "a", "The quick brown fox jumps over the lazy dog. Again!"]:
        assert 0.0 <= estimate_complexity(text) <= 1.0


def test_estimate_complexity_returns_neutral_for_degenerate_input():
    from chimera_rag.plugins.prism.features import estimate_complexity

    assert estimate_complexity("x") == 0.5


def test_estimate_entity_density_discounts_sentence_initial_capitals():
    from chimera_rag.plugins.prism.features import estimate_entity_density

    # The only capital here is sentence-initial, so nothing counts as an entity.
    assert estimate_entity_density("The the the the") == 0.0
    # Mid-sentence capitals do count.
    assert estimate_entity_density("The Nobel Curie Einstein prize") > 0.0


def test_estimate_entity_density_ranks_entity_rich_text_higher():
    from chimera_rag.plugins.prism.features import estimate_entity_density

    dense = "Alfred Nobel founded the Nobel Prize. Marie Curie discovered radium."
    sparse = "some plain prose about nothing in particular at all here today"
    assert estimate_entity_density(dense) > estimate_entity_density(sparse)


def test_estimate_entity_density_zero_for_empty_text():
    from chimera_rag.plugins.prism.features import estimate_entity_density

    assert estimate_entity_density("") == 0.0


def test_split_sentences_falls_back_to_whole_text():
    from chimera_rag.plugins.prism.features import split_sentences

    assert split_sentences("no terminal punctuation here") == ["no terminal punctuation here"]
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
