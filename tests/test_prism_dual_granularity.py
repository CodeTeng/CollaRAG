"""W4 — PrismRAG dual-granularity construction (Phase 1 + 2).

Guards contribution 1 of ``docs/innovation-1/design.md`` §6: extraction
granularity and retrieval granularity are decoupled into two layers of the
same document, bound by ``parent_chunk_id`` provenance.

The payoff is the routing table in §4.1, so the tests pin both halves — the
layer shape, and the fact that each layer really lands on the side it claims::

    granularity | extractor | vector_store
    ------------+-----------+-------------
    fine        |    yes    |     no
    coarse      |    no     |     yes

Every test forces ``min_size == max_size`` on both layers. The adaptive size
formula is score-driven, so without that the expected group boundaries would
depend on the lexical statistics of the fixture text — the tests would be
asserting an accident rather than the contract.
"""

from __future__ import annotations

import pytest

from chimera_rag.core.types import (
    EXTRACTED_GRANULARITIES,
    INDEXED_GRANULARITIES,
    Document,
)
from chimera_rag.plugins.prism.chunker import (
    PrismChunker,
    assign_parents,
    pack_sentences,
)
from chimera_rag.plugins.prism.features import split_sentences

# 19 chars each, so pack_sentences sees a uniform sent_len of 20 (+1 separator).
SENTENCE = "One two three four."


def _doc(n: int, *, doc_id: str = "d") -> Document:
    return Document(doc_id=doc_id, content=" ".join([SENTENCE] * n))


def _chunker(**kw) -> PrismChunker:
    """A chunker whose two layer targets are pinned regardless of text score."""
    params = {
        "fine_min_size": 60,
        "fine_max_size": 60,
        "coarse_min_size": 100,
        "coarse_max_size": 100,
    }
    params.update(kw)
    return PrismChunker(**params)


def _layer(chunks, granularity):
    return [c for c in chunks if c.granularity == granularity]


# ---------------------------------------------------------------------------
# pack_sentences
# ---------------------------------------------------------------------------
def test_pack_sentences_partitions_every_index_exactly_once_in_order():
    groups = pack_sentences([SENTENCE] * 10, 60)
    assert [i for g in groups for i in g] == list(range(10))


def test_pack_sentences_respects_the_target():
    """3 sentences of 20 packed units fit in 60; the 4th would overflow."""
    assert pack_sentences([SENTENCE] * 10, 60) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_pack_sentences_never_splits_an_oversized_sentence():
    """A sentence longer than the target still gets its own group — cutting it
    would reintroduce exactly the mid-sentence artefact the packer exists to
    avoid."""
    long = "X" * 500 + "."
    assert pack_sentences([long, SENTENCE], 60) == [[0], [1]]


def test_pack_sentences_empty_input_gives_no_groups():
    assert pack_sentences([], 60) == []


@pytest.mark.parametrize("target", [0, -1])
def test_pack_sentences_rejects_a_nonpositive_target(target: int):
    with pytest.raises(ValueError, match="target must be >= 1"):
        pack_sentences([SENTENCE], target)


# ---------------------------------------------------------------------------
# assign_parents
# ---------------------------------------------------------------------------
def test_assign_parents_maps_wholly_inside_groups():
    fine = [[0, 1, 2], [3, 4, 5]]
    coarse = [[0, 1, 2], [3, 4, 5]]
    assert assign_parents(fine, coarse) == [0, 1]


def test_assign_parents_binds_a_straddling_group_to_its_majority():
    """fine [3, 4, 5] straddles the coarse boundary between 4 and 5; two of its
    three sentences live in coarse 0, so that is its parent."""
    fine = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    coarse = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert assign_parents(fine, coarse) == [0, 0, 1, 1]


def test_assign_parents_breaks_ties_toward_the_earlier_coarse_chunk():
    """fine [4, 5] is split 1-1 across the boundary. Ties must be deterministic
    — the chunker re-runs on every ingest and a fluctuating parent would make
    the provenance chain unreproducible."""
    fine = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    coarse = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    assert assign_parents(fine, coarse) == [0, 0, 0, 1, 1]


def test_assign_parents_falls_back_to_zero_for_uncovered_sentences():
    assert assign_parents([[7]], [[0, 1]]) == [0]


# ---------------------------------------------------------------------------
# Dual-granularity output
# ---------------------------------------------------------------------------
def test_dual_granularity_emits_both_layers():
    chunks = _chunker().chunk(_doc(10))
    assert len(_layer(chunks, "coarse")) == 2
    assert len(_layer(chunks, "fine")) == 4


def test_every_fine_chunk_points_at_an_existing_coarse_parent():
    """The provenance chain is what makes contribution 1 more than "extract
    twice": a retrieval hit on a coarse chunk must be resolvable to the fine
    children that produced the triples."""
    chunks = _chunker().chunk(_doc(10))
    coarse_ids = {c.chunk_id for c in _layer(chunks, "coarse")}
    for ch in _layer(chunks, "fine"):
        assert ch.parent_chunk_id in coarse_ids


def test_coarse_chunks_never_carry_a_parent():
    chunks = _chunker().chunk(_doc(10))
    assert all(c.parent_chunk_id is None for c in _layer(chunks, "coarse"))


def test_the_fine_layer_is_finer_than_the_coarse_layer():
    chunks = _chunker().chunk(_doc(10))
    assert len(_layer(chunks, "fine")) > len(_layer(chunks, "coarse"))
    assert max(len(c.text) for c in _layer(chunks, "fine")) < min(
        len(c.text) for c in _layer(chunks, "coarse")
    )


def test_each_layer_covers_the_document_exactly_once():
    """Both layers are segmentations of the *same* text. If a layer dropped or
    duplicated a sentence, the fine/coarse comparison in E1 would be measuring
    two different corpora."""
    doc = _doc(10)
    expected = " ".join(split_sentences(doc.content))
    for granularity in ("fine", "coarse"):
        joined = " ".join(c.text for c in _layer(_chunker().chunk(doc), granularity))
        assert joined == expected


def test_ids_are_namespaced_so_the_two_layers_cannot_collide():
    """Both layers number ``index`` from 0, so the id — not the index — is the
    only unique handle. This is the invariant the D2/D3 sibling lookups depend
    on (they must additionally filter by granularity)."""
    chunks = _chunker().chunk(_doc(10))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[0].chunk_id == "d::coarse::0"
    assert any(c.chunk_id == "d::fine::0" for c in chunks)
    assert ("coarse", 0) in {(c.granularity, c.index) for c in chunks}
    assert ("fine", 0) in {(c.granularity, c.index) for c in chunks}


def test_indices_are_per_layer_and_contiguous():
    chunks = _chunker().chunk(_doc(10))
    for granularity, expected in (("coarse", 2), ("fine", 4)):
        layer = _layer(chunks, granularity)
        assert [c.index for c in layer] == list(range(expected))


def test_coarse_metadata_counts_its_actual_fine_children():
    chunks = _chunker().chunk(_doc(10))
    for ch in _layer(chunks, "coarse"):
        actual = sum(1 for c in _layer(chunks, "fine") if c.parent_chunk_id == ch.chunk_id)
        assert ch.metadata["fine_children"] == actual == 2


def test_metadata_records_the_layer_and_the_size_that_produced_it():
    chunks = _chunker().chunk(_doc(10))
    for ch in chunks:
        assert ch.metadata["layer"] == ch.granularity
        assert ch.metadata["adaptive_chunk_size"] == (100 if ch.granularity == "coarse" else 60)
        assert {"complexity", "density", "score"} <= ch.metadata.keys()


def test_coarse_chunks_are_emitted_before_fine_chunks():
    chunks = _chunker().chunk(_doc(10))
    assert [c.granularity for c in chunks] == ["coarse"] * 2 + ["fine"] * 4


def test_each_layer_lands_on_the_routing_side_it_claims():
    """Contribution 1 is only worth anything if the pipeline actually routes the
    two layers differently, so tie the chunker's output back to the routing
    sets rather than restating them."""
    chunks = _chunker().chunk(_doc(10))
    for ch in _layer(chunks, "fine"):
        assert ch.granularity in EXTRACTED_GRANULARITIES
        assert ch.granularity not in INDEXED_GRANULARITIES
    for ch in _layer(chunks, "coarse"):
        assert ch.granularity in INDEXED_GRANULARITIES
        assert ch.granularity not in EXTRACTED_GRANULARITIES


def test_a_single_short_document_still_yields_one_chunk_per_layer():
    chunks = _chunker().chunk(_doc(1))
    assert len(chunks) == 2
    assert {c.granularity for c in chunks} == {"fine", "coarse"}
    assert chunks[1].parent_chunk_id == chunks[0].chunk_id


def test_empty_document_returns_no_chunks_in_either_layer():
    assert _chunker().chunk(Document(doc_id="d", content="")) == []


def test_the_two_layers_are_independent_segmentations_not_a_split_of_one():
    """A coarse chunk is not the concatenation of *its own* fine children in the
    general case: fine [3, 4, 5] straddles the coarse boundary, so coarse 0
    holds sentences 0-4 while its children hold 0-2 and 3-5. Asserting the
    naive nesting would freeze an accident of the fixture."""
    chunks = _chunker().chunk(_doc(10))
    first_coarse = _layer(chunks, "coarse")[0]
    children = [c for c in _layer(chunks, "fine") if c.parent_chunk_id == first_coarse.chunk_id]
    assert [c.text for c in children] == [
        " ".join([SENTENCE] * 3),
        " ".join([SENTENCE] * 3),
    ]
    assert first_coarse.text == " ".join([SENTENCE] * 5)


# ---------------------------------------------------------------------------
# The transitional single-granularity path (E2 row 2)
# ---------------------------------------------------------------------------
def test_disabling_dual_granularity_reproduces_the_transitional_output():
    """Ablation row 2 of E2 needs this path to stay reachable, and it must be
    byte-identical to what the pre-W4 chunker produced."""
    chunks = _chunker(dual_granularity=False).chunk(_doc(10))
    assert [c.chunk_id for c in chunks] == [f"d::{i}" for i in range(len(chunks))]
    assert all(c.granularity == "unified" for c in chunks)
    assert all(c.parent_chunk_id is None for c in chunks)
    assert all(c.metadata["layer"] == "unified" for c in chunks)


def test_single_granularity_uses_the_legacy_size_knobs():
    chunks = PrismChunker(
        min_chunk_size=60, max_chunk_size=60, base_chunk_size=60, dual_granularity=False
    ).chunk(_doc(10))
    assert all(c.metadata["adaptive_chunk_size"] == 60 for c in chunks)
    assert len(chunks) == 4


def test_the_legacy_size_knobs_do_not_leak_into_the_dual_layers():
    """min_/max_/base_chunk_size are ignored when dual is on; the four fine_*/
    coarse_* keys own the two layers. Pinning this stops a future refactor from
    silently letting a config preset change only one of the two."""
    chunks = _chunker(min_chunk_size=10, max_chunk_size=10, base_chunk_size=10).chunk(_doc(10))
    assert {c.metadata["adaptive_chunk_size"] for c in chunks} == {60, 100}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_rejects_a_coarse_layer_finer_than_the_fine_layer():
    """Inverted bounds would make "coarse" the extraction layer, silently
    swapping contribution 1's two consumers."""
    with pytest.raises(ValueError, match="must not be finer than the fine layer"):
        PrismChunker(fine_min_size=200, fine_max_size=400, coarse_min_size=100, coarse_max_size=300)


def test_rejects_inverted_fine_bounds():
    with pytest.raises(ValueError, match="fine_min_size <= fine_max_size"):
        PrismChunker(fine_min_size=300, fine_max_size=100)


def test_rejects_inverted_coarse_bounds():
    with pytest.raises(ValueError, match="coarse_min_size <= coarse_max_size"):
        PrismChunker(coarse_min_size=900, coarse_max_size=400)


@pytest.mark.parametrize("budget", [0, -5])
def test_rejects_a_nonpositive_triple_budget(budget: int):
    """D5 reads this cap as its signal, so a nonsensical budget would either
    fire on every chunk or never."""
    with pytest.raises(ValueError, match="triple_budget must be >= 1"):
        PrismChunker(triple_budget=budget)


# ---------------------------------------------------------------------------
# Capability wiring
# ---------------------------------------------------------------------------
def test_prism_chunker_implements_both_the_slot_and_the_audit_capability():
    from chimera_rag.interfaces.chunker import BaseChunkAuditor, BaseChunker

    chunker = PrismChunker()
    assert isinstance(chunker, BaseChunker)
    assert isinstance(chunker, BaseChunkAuditor)


def test_audit_disabled_records_no_defects_but_still_chunks():
    """Ablation row 4 of E2: dual granularity without the audit loop."""
    chunker = _chunker(audit_enabled=False)
    assert chunker.chunk(_doc(10))
    assert chunker.audit(_chunker().chunk(_doc(10)), {}) == []


def test_repair_disabled_returns_the_input_chunks_untouched():
    """Ablation row 5 of E2: defects are recorded, but nothing is re-cut, so
    ``defects_applied`` stays 0 and the pipeline loop terminates at once."""
    from chimera_rag.core.types import ChunkDefect

    chunker = _chunker(repair_enabled=False)
    chunks = chunker.chunk(_doc(10))
    defect = ChunkDefect(
        code="D5", chunk_id=chunks[0].chunk_id, action="split", detail="over budget"
    )
    outcome = chunker.repair([defect], chunks)
    assert outcome.defects_applied == 0
    assert outcome.defects_deferred == 1
    assert outcome.chunks == list(chunks)
