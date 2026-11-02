"""Tests for the two triple-structure detectors, D3 and D6.

Both are read off the extractor's output rather than the text, and both
damage the **path** and **attribute** projections respectively.

* **D3 relation straddling** — two adjacent extraction units share a bridge
  entity, so a 2-hop path runs across the boundary and exists in neither
  chunk. Action ``bind``: pull both under a common coarse parent. This is the
  ingestion-side twin of CollaRAG's ``missing_path`` gap, which is what makes
  experiment E5 measurable.
* **D6 intra-chunk contradiction** — one subject carrying two different values
  for a functional (single-valued) predicate inside the same chunk, i.e. two
  topics got packed together. Action ``split_by_topic``.

D6 cannot know which predicates are functional without an ontology, so it has
two tiers: an explicit ``exclusive_predicates`` set reports ``high``, and a
name-based heuristic reports ``low`` (record-only, per risk R3).
"""

from __future__ import annotations

import pytest

from chimera_rag.core.types import Chunk, Triple


def _chunk(index: int = 0, doc_id: str = "d", text: str = "x", **kw) -> Chunk:
    return Chunk(
        chunk_id=kw.pop("chunk_id", f"{doc_id}::{index}"),
        doc_id=doc_id,
        index=index,
        text=text,
        **kw,
    )


def _t(subject: str, predicate: str, obj: str, chunk_id: str | None = None) -> Triple:
    return Triple(subject=subject, predicate=predicate, object=obj, source_chunk_id=chunk_id)


def _ctx(chunks_with_triples: list[tuple[Chunk, list[Triple]]]):
    from chimera_rag.plugins.prism.defects import AuditContext

    return AuditContext(
        chunks=tuple(c for c, _ in chunks_with_triples),
        triples_by_chunk={c.chunk_id: ts for c, ts in chunks_with_triples},
    )


# ===========================================================================
# D3 — relation straddling a chunk boundary
# ===========================================================================
def _d3(**kw):
    from chimera_rag.plugins.prism.defects import D3StraddleDetector

    return D3StraddleDetector(**kw)


def _bridge_pair(**kw):
    """A -> Apple -> B split across an adjacent boundary: the textbook D3."""
    granularity = kw.pop("granularity", "unified")
    parent = kw.pop("parent", None)
    a = _chunk(index=0, granularity=granularity, parent_chunk_id=parent)
    b = _chunk(index=1, granularity=granularity, parent_chunk_id=parent)
    ta = [_t("Steve Jobs", "founded", "Apple", a.chunk_id)]
    tb = [_t("Apple", "headquartered_in", "Cupertino", b.chunk_id)]
    return a, b, ta, tb


def test_d3_flags_a_path_that_straddles_an_adjacent_boundary():
    a, b, ta, tb = _bridge_pair()
    (defect,) = _d3().detect(a, ta, _ctx([(a, ta), (b, tb)]))
    assert defect.code == "D3"
    assert defect.action == "bind"


def test_d3_anchors_on_the_earlier_chunk_and_names_the_sibling():
    a, b, ta, tb = _bridge_pair()
    (defect,) = _d3().detect(a, ta, _ctx([(a, ta), (b, tb)]))
    assert defect.chunk_id == "d::0"
    assert defect.related_chunk_ids == ["d::1"]


def test_d3_records_the_bridge_entity_and_the_severed_path():
    a, b, ta, tb = _bridge_pair()
    (defect,) = _d3().detect(a, ta, _ctx([(a, ta), (b, tb)]))
    assert defect.metadata["bridges"] == ["Apple"]
    assert defect.metadata["confidence"] == "high"
    assert "Apple" in defect.detail


def test_d3_reports_one_defect_per_pair_even_with_several_bridges():
    a = _chunk(index=0)
    b = _chunk(index=1)
    ta = [
        _t("Steve Jobs", "founded", "Apple", a.chunk_id),
        _t("Apple", "acquired", "NeXT", a.chunk_id),
    ]
    tb = [
        _t("NeXT", "developed", "Interface Builder", b.chunk_id),
        _t("Apple", "product", "iPhone", b.chunk_id),
    ]
    (defect,) = _d3().detect(a, ta, _ctx([(a, ta), (b, tb)]))
    assert defect.metadata["bridges"] == ["Apple", "NeXT"]


# ---------------------------------------------------------------------------
# Suppression
# ---------------------------------------------------------------------------
def test_d3_is_suppressed_when_the_pair_already_shares_a_coarse_parent():
    """The connective tissue already exists — ``bind`` would be a no-op, so
    this is the D3 analogue of D1's graph-based falsification."""
    a, b, ta, tb = _bridge_pair(granularity="fine", parent="d::coarse::0")
    assert _d3().detect(a, ta, _ctx([(a, ta), (b, tb)])) == []


def test_d3_still_fires_when_the_siblings_have_different_parents():
    a = _chunk(index=0, granularity="fine", parent_chunk_id="d::coarse::0")
    b = _chunk(index=1, granularity="fine", parent_chunk_id="d::coarse::1")
    ta = [_t("Steve Jobs", "founded", "Apple", a.chunk_id)]
    tb = [_t("Apple", "headquartered_in", "Cupertino", b.chunk_id)]
    assert len(_d3().detect(a, ta, _ctx([(a, ta), (b, tb)]))) == 1


def test_d3_is_silent_without_a_shared_entity():
    a = _chunk(index=0)
    b = _chunk(index=1)
    ta = [_t("Steve Jobs", "founded", "Apple", a.chunk_id)]
    tb = [_t("Alfred Nobel", "invented", "dynamite", b.chunk_id)]
    assert _d3().detect(a, ta, _ctx([(a, ta), (b, tb)])) == []


def test_d3_is_silent_when_the_two_halves_are_already_joined():
    """Jobs and Cupertino are directly linked inside B, so the path through
    Apple has a shortcut and binding would add nothing."""
    a = _chunk(index=0)
    b = _chunk(index=1)
    ta = [_t("Steve Jobs", "founded", "Apple", a.chunk_id)]
    tb = [
        _t("Apple", "headquartered_in", "Cupertino", b.chunk_id),
        _t("Steve Jobs", "born_in", "Cupertino", b.chunk_id),
    ]
    assert _d3().detect(a, ta, _ctx([(a, ta), (b, tb)])) == []


def test_d3_only_looks_at_immediately_adjacent_chunks():
    a = _chunk(index=0)
    mid = _chunk(index=1)
    far = _chunk(index=2)
    ta = [_t("Steve Jobs", "founded", "Apple", a.chunk_id)]
    tm = [_t("Something", "else", "entirely", mid.chunk_id)]
    tf = [_t("Apple", "headquartered_in", "Cupertino", far.chunk_id)]
    ctx = _ctx([(a, ta), (mid, tm), (far, tf)])
    assert _d3().detect(a, ta, ctx) == []


def test_d3_never_bridges_across_documents():
    a = _chunk(index=0, doc_id="d1", chunk_id="d1::0")
    b = _chunk(index=1, doc_id="d2", chunk_id="d2::1")
    ta = [_t("Steve Jobs", "founded", "Apple", a.chunk_id)]
    tb = [_t("Apple", "headquartered_in", "Cupertino", b.chunk_id)]
    assert _d3().detect(a, ta, _ctx([(a, ta), (b, tb)])) == []


def test_d3_picks_the_same_layer_sibling_when_indices_collide():
    """Under dual granularity the fine and coarse layers each number from 0, so
    ``index == 1`` exists twice in one document."""
    fine0 = _chunk(index=0, granularity="fine", parent_chunk_id="d::c::0", chunk_id="d::fine::0")
    fine1 = _chunk(index=1, granularity="fine", parent_chunk_id="d::c::1", chunk_id="d::fine::1")
    coarse1 = _chunk(index=1, granularity="coarse", chunk_id="d::coarse::1")
    t0 = [_t("Steve Jobs", "founded", "Apple", fine0.chunk_id)]
    t1 = [_t("Apple", "headquartered_in", "Cupertino", fine1.chunk_id)]
    # coarse1 is listed ahead of fine1, so a layer-blind lookup finds it first.
    ctx = _ctx([(fine0, t0), (coarse1, []), (fine1, t1)])
    (defect,) = _d3().detect(fine0, t0, ctx)
    assert defect.related_chunk_ids == ["d::fine::1"]


def test_d3_is_silent_without_sibling_context():
    a, _b, ta, _tb = _bridge_pair()
    assert _d3().detect(a, ta) == []


def test_d3_is_silent_for_an_empty_chunk():
    a = _chunk(index=0)
    b = _chunk(index=1)
    tb = [_t("Apple", "headquartered_in", "Cupertino", b.chunk_id)]
    assert _d3().detect(a, [], _ctx([(a, []), (b, tb)])) == []


# ---------------------------------------------------------------------------
# Directionality decides confidence
# ---------------------------------------------------------------------------
def test_d3_drops_to_low_confidence_without_a_directed_flow():
    """Both halves use the bridge as a subject, so there is no in->out path to
    reconnect — weaker evidence, record only."""
    a = _chunk(index=0)
    b = _chunk(index=1)
    ta = [_t("Marie Curie", "born_in", "Warsaw", a.chunk_id)]
    tb = [_t("Marie Curie", "won", "Nobel Prize in Physics", b.chunk_id)]
    (defect,) = _d3().detect(a, ta, _ctx([(a, ta), (b, tb)]))
    assert defect.metadata["confidence"] == "low"


# ---------------------------------------------------------------------------
# Granularity gating
# ---------------------------------------------------------------------------
def test_d3_skips_chunks_that_are_never_extracted():
    a, b, _ta, tb = _bridge_pair(granularity="coarse")
    assert _d3().detect(a, [], _ctx([(a, []), (b, tb)])) == []


def test_d3_detector_exposes_its_code_as_a_class_attribute():
    from chimera_rag.plugins.prism.defects import D3StraddleDetector

    assert D3StraddleDetector.code == "D3"


# ===========================================================================
# D6 — intra-chunk contradiction
# ===========================================================================
def _d6(**kw):
    from chimera_rag.plugins.prism.defects import D6ContradictionDetector

    return D6ContradictionDetector(**kw)


CONTRADICTED = [
    _t("Marie Curie", "born_in", "Warsaw"),
    _t("Marie Curie", "born_in", "Krakow"),
]


def test_d6_flags_two_values_for_a_functional_predicate():
    defects = _d6().detect(_chunk(), CONTRADICTED)
    assert len(defects) == 1
    assert defects[0].code == "D6"
    assert defects[0].action == "split_by_topic"


def test_d6_an_explicit_exclusive_predicate_reports_high_confidence():
    (defect,) = _d6().detect(_chunk(), CONTRADICTED)
    assert defect.metadata["confidence"] == "high"
    assert defect.metadata["source"] == "explicit"
    assert defect.metadata["subject"] == "Marie Curie"
    assert defect.metadata["predicate"] == "born_in"
    assert defect.metadata["objects"] == ["Krakow", "Warsaw"]


def test_d6_a_name_heuristic_reports_low_confidence():
    """``born_at`` is not in the curated set, but its name says functional —
    plausible enough to record, not enough to drive a repair."""
    triples = [_t("Marie Curie", "born_at", "Warsaw"), _t("Marie Curie", "born_at", "Krakow")]
    (defect,) = _d6().detect(_chunk(), triples)
    assert defect.metadata["confidence"] == "low"
    assert defect.metadata["source"] == "heuristic"


def test_d6_heuristic_can_be_switched_off():
    triples = [_t("Marie Curie", "born_at", "Warsaw"), _t("Marie Curie", "born_at", "Krakow")]
    assert _d6(enable_heuristic=False).detect(_chunk(), triples) == []


def test_d6_is_silent_for_a_legitimately_multi_valued_predicate():
    triples = [
        _t("Marie Curie", "won", "Nobel Prize in Physics"),
        _t("Marie Curie", "won", "Nobel Prize in Chemistry"),
    ]
    assert _d6().detect(_chunk(), triples) == []


def test_d6_is_silent_for_duplicate_objects():
    """A repeat is redundancy for the pruner, not a contradiction."""
    triples = [_t("Marie Curie", "born_in", "Warsaw"), _t("Marie Curie", "born_in", "Warsaw")]
    assert _d6().detect(_chunk(), triples) == []


def test_d6_is_silent_when_objects_differ_only_by_surface_form():
    """"Warsaw" and "Warsaw, Poland" are one value written two ways."""
    triples = [
        _t("Marie Curie", "born_in", "Warsaw"),
        _t("Marie Curie", "born_in", "Warsaw, Poland"),
    ]
    assert _d6().detect(_chunk(), triples) == []


def test_d6_is_silent_for_a_single_triple():
    assert _d6().detect(_chunk(), CONTRADICTED[:1]) == []


def test_d6_reports_one_defect_per_contradicted_subject():
    triples = [
        *CONTRADICTED,
        _t("Alfred Nobel", "born_in", "Stockholm"),
        _t("Alfred Nobel", "born_in", "Gothenburg"),
    ]
    defects = _d6().detect(_chunk(), triples)
    assert sorted(d.metadata["subject"] for d in defects) == ["Alfred Nobel", "Marie Curie"]


def test_d6_exclusive_predicates_are_configurable():
    d = _d6(exclusive_predicates={"capital_of"})
    configured = [_t("Poland", "capital_of", "Europe"), _t("Poland", "capital_of", "Warsaw")]
    (defect,) = d.detect(_chunk(), configured)
    assert defect.metadata["source"] == "explicit"
    # The curated default is gone, so born_in now only reaches the heuristic.
    (demoted,) = d.detect(_chunk(), CONTRADICTED)
    assert demoted.metadata["source"] == "heuristic"


def test_d6_skips_chunks_that_are_never_extracted():
    ch = _chunk(granularity="coarse")
    assert _d6().detect(ch, CONTRADICTED) == []


def test_d6_rejects_an_empty_exclusive_predicate_set():
    with pytest.raises(ValueError, match="exclusive_predicates must not be empty"):
        _d6(exclusive_predicates=set())


def test_d6_detector_exposes_its_code_as_a_class_attribute():
    from chimera_rag.plugins.prism.defects import D6ContradictionDetector

    assert D6ContradictionDetector.code == "D6"
