"""Tests for the two context-starvation detectors, D2 and D4.

Both fire when a chunk asks the reader (or the extractor) for context the
chunk does not contain, and both are repaired by *enlarging* rather than
splitting — the direction that makes them irreconcilable with D1/D5/D6 in a
single segmentation layer (``docs/innovation-1/design.md`` §1.2).

* **D2 antecedent loss** — the opening sentence contains an anaphor whose
  antecedent is not in the chunk. Action ``extend_left``, on the layer that
  has to read standalone.
* **D4 context starvation** — the chunk visibly contains a named entity yet
  yields zero triples. Action ``merge_up``: give up extracting it alone and
  let the coarse parent cover it.

D4 carries a corpus-level guard: when *most* audited chunks yield nothing, the
extractor is broken rather than the chunking, so the defect drops to
``confidence="low"`` (record-only, per risk R3) instead of triggering repairs
that cannot help.
"""

from __future__ import annotations

import pytest

from chimera_rag.core.types import Chunk, Triple


def _chunk(text: str, index: int = 0, doc_id: str = "d", **kw) -> Chunk:
    return Chunk(
        chunk_id=kw.pop("chunk_id", f"{doc_id}::{index}"),
        doc_id=doc_id,
        index=index,
        text=text,
        **kw,
    )


def _t(subject: str, predicate: str, obj: str, chunk_id: str = "d::0") -> Triple:
    return Triple(subject=subject, predicate=predicate, object=obj, source_chunk_id=chunk_id)


def _ctx(*, chunks=(), triples_by_chunk=None):
    from chimera_rag.plugins.prism.defects import AuditContext

    return AuditContext(chunks=tuple(chunks), triples_by_chunk=triples_by_chunk or {})


# ===========================================================================
# D2 — antecedent loss
# ===========================================================================
def _d2(**kw):
    from chimera_rag.plugins.prism.defects import D2AntecedentDetector

    return D2AntecedentDetector(**kw)


def test_d2_flags_an_unresolvable_opening_pronoun():
    ch = _chunk("He invented dynamite in 1867. The blast destroyed the shed.")
    (defect,) = _d2().detect(ch, [])
    assert defect.code == "D2"
    assert defect.action == "extend_left"
    assert defect.metadata["confidence"] == "high"
    assert defect.metadata["anaphor"] == "He"


def test_d2_flags_a_pronoun_midway_through_the_opening_sentence():
    ch = _chunk("After the accident, he left Sweden for good. Nothing followed.")
    (defect,) = _d2().detect(ch, [])
    assert defect.metadata["anaphor"] == "he"


def test_d2_flags_a_generic_definite_np():
    ch = _chunk("The company was founded in 1976. Its products sold well.")
    (defect,) = _d2().detect(ch, [])
    assert defect.metadata["anaphor"] == "The company"


def test_d2_flags_a_sentence_initial_demonstrative():
    ch = _chunk("It was patented in 1867. Nothing else is said here.")
    (defect,) = _d2().detect(ch, [])
    assert defect.metadata["anaphor"] == "It"


def test_d2_ignores_a_mid_sentence_complementiser_that():
    """"that" is only anaphoric at the head of the sentence; elsewhere it is a
    complementiser and would fire on almost every English paragraph."""
    ch = _chunk("The report shows that dynamite was patented. More text here.")
    assert _d2().detect(ch, []) == []


def test_d2_is_silent_when_the_opening_sentence_has_no_anaphor():
    ch = _chunk("Alfred Nobel was a Swedish chemist. He invented dynamite.")
    assert _d2().detect(ch, []) == []


def test_d2_is_silent_when_an_entity_precedes_the_anaphor():
    ch = _chunk("In Stockholm, Nobel founded the company he later expanded.")
    assert _d2().detect(ch, []) == []


def test_d2_drops_to_low_confidence_when_cataphora_is_possible():
    """The chunk does name the entity, just *after* the pronoun — a reader can
    still resolve it, so this is recorded rather than repaired."""
    ch = _chunk("He invented dynamite in 1867. Alfred Nobel was born in Sweden.")
    (defect,) = _d2().detect(ch, [])
    assert defect.metadata["confidence"] == "low"


def test_d2_points_at_the_preceding_chunk_as_the_extension_source():
    prev = _chunk("Alfred Nobel was a Swedish chemist.", index=0)
    cur = _chunk("He invented dynamite in 1867.", index=1, chunk_id="d::1")
    (defect,) = _d2().detect(cur, [], _ctx(chunks=[prev, cur]))
    assert defect.related_chunk_ids == ["d::0"]


def test_d2_leaves_related_chunks_empty_at_the_start_of_a_document():
    cur = _chunk("He invented dynamite in 1867.", index=0)
    (defect,) = _d2().detect(cur, [], _ctx(chunks=[cur]))
    assert defect.related_chunk_ids == []


def test_d2_does_not_cross_documents():
    other = _chunk("Some other text.", index=0, doc_id="other", chunk_id="other::0")
    cur = _chunk("He invented dynamite in 1867.", index=1, doc_id="d")
    (defect,) = _d2().detect(cur, [], _ctx(chunks=[other, cur]))
    assert defect.related_chunk_ids == []


def test_d2_picks_the_same_layer_predecessor_when_indices_collide():
    """``unified`` and ``coarse`` chunks really can coexist in one document (a
    D3 ``bind`` converts two ``unified`` chunks into ``fine`` children of a new
    ``coarse`` cover, leaving the rest of the document ``unified``), and both
    layers number ``index`` from 0. A layer-blind ``index - 1`` lookup would
    hand ``extend_left`` a chunk from the other granularity."""
    unified0 = _chunk("Unified opening text.", index=0, chunk_id="d::0")
    coarse0 = _chunk("Coarse opening text.", index=0, granularity="coarse", chunk_id="d::coarse::0")
    coarse1 = _chunk(
        "He invented dynamite in 1867.", index=1, granularity="coarse", chunk_id="d::coarse::1"
    )
    (defect,) = _d2().detect(coarse1, [], _ctx(chunks=[unified0, coarse0, coarse1]))
    assert defect.related_chunk_ids == ["d::coarse::0"]


# ---------------------------------------------------------------------------
# D2 is a property of the retrieval layer, not the extraction layer
# ---------------------------------------------------------------------------
def test_d2_skips_fine_chunks():
    """A fine chunk is *meant* to be a fragment — its coarse parent carries the
    context, so flagging it would be pure noise."""
    ch = _chunk("He invented dynamite in 1867.", granularity="fine", parent_chunk_id="d::c0")
    assert _d2().detect(ch, []) == []


@pytest.mark.parametrize("granularity", ["unified", "coarse"])
def test_d2_audits_the_self_sufficient_layers(granularity: str):
    ch = _chunk("He invented dynamite in 1867.", granularity=granularity)
    assert len(_d2().detect(ch, [])) == 1


def test_d2_detector_exposes_its_code_as_a_class_attribute():
    from chimera_rag.plugins.prism.defects import D2AntecedentDetector

    assert D2AntecedentDetector.code == "D2"


# ===========================================================================
# D4 — context starvation
# ===========================================================================
def _d4(**kw):
    from chimera_rag.plugins.prism.defects import D4StarvationDetector

    return D4StarvationDetector(**kw)


def test_d4_flags_an_entity_bearing_chunk_that_yielded_nothing():
    ch = _chunk("Alfred Nobel founded the Nobel Prize in 1901.")
    (defect,) = _d4().detect(ch, [], _ctx(triples_by_chunk={"d::0": []}))
    assert defect.code == "D4"
    assert defect.action == "merge_up"
    assert defect.metadata["confidence"] == "high"
    assert defect.metadata["entity_count"] == 2


def test_d4_is_silent_when_the_chunk_yielded_triples():
    ch = _chunk("Alfred Nobel founded the Nobel Prize in 1901.")
    triples = [_t("Alfred Nobel", "founded", "Nobel Prize")]
    assert _d4().detect(ch, triples, _ctx(triples_by_chunk={"d::0": triples})) == []


def test_d4_is_silent_when_the_chunk_has_no_entity_to_starve_on():
    """Zero triples from pure prose is not a chunking defect."""
    ch = _chunk("some plain prose about nothing in particular at all here today")
    assert _d4().detect(ch, [], _ctx(triples_by_chunk={"d::0": []})) == []


def test_d4_drops_to_low_confidence_when_the_whole_batch_is_empty():
    """If most chunks yield nothing the extractor is broken, not the chunking —
    repairing boundaries cannot help, so record only (risk R3)."""
    batch = {f"d::{i}": [] for i in range(4)}
    ch = _chunk("Alfred Nobel founded the Nobel Prize in 1901.")
    (defect,) = _d4().detect(ch, [], _ctx(triples_by_chunk=batch))
    assert defect.metadata["confidence"] == "low"
    assert defect.metadata["batch_zero_rate"] == 1.0


def test_d4_stays_high_confidence_when_only_this_chunk_is_empty():
    batch = {
        "d::0": [],
        "d::1": [_t("a", "p", "b", "d::1")],
        "d::2": [_t("c", "p", "d", "d::2")],
    }
    ch = _chunk("Alfred Nobel founded the Nobel Prize in 1901.")
    (defect,) = _d4().detect(ch, [], _ctx(triples_by_chunk=batch))
    assert defect.metadata["confidence"] == "high"


def test_d4_zero_rate_guard_is_configurable():
    batch = {"d::0": [], "d::1": [_t("a", "p", "b", "d::1")]}
    ch = _chunk("Alfred Nobel founded the Nobel Prize in 1901.")
    # 0.5 zero rate: strict guard flags it, relaxed guard does not.
    (strict,) = _d4(max_zero_rate=0.25).detect(ch, [], _ctx(triples_by_chunk=batch))
    assert strict.metadata["confidence"] == "low"
    (relaxed,) = _d4(max_zero_rate=0.9).detect(ch, [], _ctx(triples_by_chunk=batch))
    assert relaxed.metadata["confidence"] == "high"


def test_d4_without_a_batch_assumes_the_chunk_is_the_only_evidence():
    ch = _chunk("Alfred Nobel founded the Nobel Prize in 1901.")
    (defect,) = _d4().detect(ch, [])
    assert defect.metadata["confidence"] == "high"


def test_d4_zero_rate_guard_needs_a_batch_to_judge():
    """A rate computed from one chunk is not evidence, so the guard stays
    inert below ``min_batch``."""
    ch = _chunk("Alfred Nobel founded the Nobel Prize in 1901.")
    lone = _ctx(triples_by_chunk={"d::0": []})
    (defect,) = _d4(min_batch=2).detect(ch, [], lone)
    assert defect.metadata["confidence"] == "high"
    pair = _ctx(triples_by_chunk={"d::0": [], "d::1": []})
    (defect,) = _d4(min_batch=2).detect(ch, [], pair)
    assert defect.metadata["confidence"] == "low"


@pytest.mark.parametrize("granularity", ["unified", "fine"])
def test_d4_audits_the_extraction_layers(granularity: str):
    parent = "d::c0" if granularity == "fine" else None
    ch = _chunk(
        "Alfred Nobel founded the Nobel Prize in 1901.",
        granularity=granularity,
        parent_chunk_id=parent,
    )
    assert len(_d4().detect(ch, [])) == 1


def test_d4_skips_coarse_chunks():
    """Coarse chunks are never extracted, so zero triples is expected."""
    ch = _chunk("Alfred Nobel founded the Nobel Prize in 1901.", granularity="coarse")
    assert _d4().detect(ch, []) == []


def test_d4_rejects_a_max_zero_rate_outside_zero_to_one():
    with pytest.raises(ValueError, match="max_zero_rate must be in"):
        _d4(max_zero_rate=1.5)


def test_d4_detector_exposes_its_code_as_a_class_attribute():
    from chimera_rag.plugins.prism.defects import D4StarvationDetector

    assert D4StarvationDetector.code == "D4"
