"""Tests for the D1 (entity-collision) chunk defect detector.

D1 fires when one surface form inside a single chunk maps to two or more
mutually unreachable neighbourhoods — the structural fingerprint of two
different real-world entities sharing a name ("Apple" the company and "apple"
the fruit), which poisons the attribute projection.

Two decisions from ``docs/innovation-1/design.md`` are load-bearing here:

* **A local-only check false-positives on every hub entity.** Marie Curie's
  neighbours {Warsaw, Nobel Prize in Physics, Physics} have no edges among
  them inside one chunk, yet she is not a collision. So the detector must be
  able to read the existing graph to *falsify* a candidate.
* **Risk R3 — detector noise is amplified by the repair loop.** When no graph
  is available the detector cannot falsify, so it reports the candidate at
  ``confidence="low"`` and the loop must treat it as record-only. Only a
  candidate that survives global confirmation is reported at
  ``confidence="high"`` and allowed to drive a repair.
"""

from __future__ import annotations

import pytest

from chimera_rag.core.types import Chunk, Triple


def _chunk(text: str = "x", **kw) -> Chunk:
    return Chunk(
        chunk_id=kw.pop("chunk_id", "d::0"),
        doc_id=kw.pop("doc_id", "d"),
        index=0,
        text=text,
        **kw,
    )


def _t(subject: str, predicate: str, obj: str) -> Triple:
    return Triple(subject=subject, predicate=predicate, object=obj, source_chunk_id="d::0")


def _detector(**kw):
    from chimera_rag.plugins.prism.defects import D1CollisionDetector

    return D1CollisionDetector(**kw)


def _graph(triples: list[Triple]):
    from chimera_rag.stores.graph import NetworkXGraphStore

    g = NetworkXGraphStore()
    for t in triples:
        g.add_triple(t)
    return g


# The canonical collision: one surface form, four unrelated neighbourhoods.
APPLE_MIXED = [
    _t("Apple", "product", "iPhone"),
    _t("Apple", "founded_by", "Steve Jobs"),
    _t("Apple", "is_a", "fruit"),
    _t("Apple", "tastes", "sweet"),
]

# The canonical hub: three neighbours with no edges among them *in this chunk*,
# yet a single real entity. Must not be reported when the graph can falsify it.
CURIE_HUB = [
    _t("Marie Curie", "born_in", "Warsaw"),
    _t("Marie Curie", "won", "Nobel Prize in Physics"),
    _t("Marie Curie", "field", "Physics"),
]


# ---------------------------------------------------------------------------
# Firing
# ---------------------------------------------------------------------------
def test_d1_flags_one_surface_form_with_unreachable_neighbourhoods():
    defects = _detector().detect(_chunk(), APPLE_MIXED)
    assert len(defects) == 1
    assert defects[0].code == "D1"
    assert defects[0].action == "split"
    assert defects[0].chunk_id == "d::0"


def test_d1_names_the_colliding_entity_in_metadata():
    (defect,) = _detector().detect(_chunk(), APPLE_MIXED)
    assert defect.metadata["entity"] == "Apple"
    assert defect.metadata["components"] == 4
    assert defect.metadata["confidence"] == "low"
    assert "Apple" in defect.detail


def test_d1_without_a_graph_reports_low_confidence():
    """No graph means no way to falsify — record-only per risk R3."""
    (defect,) = _detector().detect(_chunk(), APPLE_MIXED)
    assert defect.metadata["confidence"] == "low"


def test_d1_reports_high_confidence_when_the_graph_cannot_falsify_it():
    from chimera_rag.plugins.prism.defects import AuditContext

    # The global graph knows "Apple" but its neighbourhood is still split.
    ctx = AuditContext(graph=_graph([_t("Apple", "product", "iPad")]))
    (defect,) = _detector().detect(_chunk(), APPLE_MIXED, ctx)
    assert defect.metadata["confidence"] == "high"


# ---------------------------------------------------------------------------
# Falsification: the hub-entity case that motivated reading the graph
# ---------------------------------------------------------------------------
def test_d1_suppresses_a_hub_entity_whose_neighbours_connect_globally():
    from chimera_rag.plugins.prism.defects import AuditContext

    # Globally, Warsaw hosted the ceremony and the prize belongs to Physics,
    # so the three "disconnected" local components merge without going
    # through Marie Curie herself.
    ctx = AuditContext(
        graph=_graph(
            [
                *CURIE_HUB,
                _t("Warsaw", "hosted", "Nobel Prize in Physics"),
                _t("Nobel Prize in Physics", "in_field", "Physics"),
            ]
        )
    )
    assert _detector().detect(_chunk(), CURIE_HUB, ctx) == []


def test_d1_keeps_a_candidate_whose_neighbours_stay_split_globally():
    from chimera_rag.plugins.prism.defects import AuditContext

    # The global graph merges the company neighbourhood internally, but the
    # fruit neighbourhood is still unreachable from it.
    ctx = AuditContext(
        graph=_graph([*APPLE_MIXED, _t("Steve Jobs", "designed", "iPhone")])
    )
    (defect,) = _detector().detect(_chunk(), APPLE_MIXED, ctx)
    assert defect.metadata["confidence"] == "high"
    assert defect.metadata["components"] == 4
    assert defect.metadata["global_components"] == 3


def test_d1_does_not_falsify_through_the_ambiguous_entity_itself():
    """The confirmation path must not run *through* the surface form, or every
    candidate would trivially merge at hop 1."""
    from chimera_rag.plugins.prism.defects import AuditContext

    # Every neighbour hangs off "Apple" and nothing else. Routing through the
    # ambiguous surface form would merge them all; removing it must not.
    ctx = AuditContext(graph=_graph(APPLE_MIXED))
    (defect,) = _detector().detect(_chunk(), APPLE_MIXED, ctx)
    assert defect.metadata["confidence"] == "high"
    assert defect.metadata["global_components"] == 4


# ---------------------------------------------------------------------------
# Silence: what must not be a collision
# ---------------------------------------------------------------------------
def test_d1_is_silent_below_the_neighbour_floor():
    two = [_t("Nobel", "born_in", "Sweden"), _t("Nobel", "invented", "dynamite")]
    assert _detector(min_neighbors=3).detect(_chunk(), two) == []


def test_d1_is_silent_when_neighbours_are_connected_inside_the_chunk():
    triples = [
        _t("Marie Curie", "won", "Nobel Prize in Physics"),
        _t("Marie Curie", "field", "Physics"),
        _t("Marie Curie", "born_in", "Warsaw"),
        # The connective tissue is right here in the same chunk.
        _t("Nobel Prize in Physics", "in_field", "Physics"),
        _t("Warsaw", "hosts", "Nobel Prize in Physics"),
    ]
    assert _detector().detect(_chunk(), triples) == []


def test_d1_is_silent_for_a_single_neighbour_chain():
    triples = [_t("a", "p", "b"), _t("b", "p", "c"), _t("c", "p", "d")]
    assert _detector().detect(_chunk(), triples) == []


def test_d1_is_silent_for_an_empty_triple_list():
    assert _detector().detect(_chunk(), []) == []


def test_d1_ignores_self_loops():
    """``(X, same_as, X)`` must not inflate the neighbour count."""
    triples = [
        _t("Nobel", "same_as", "Nobel"),
        _t("Nobel", "born_in", "Sweden"),
        _t("Nobel", "invented", "dynamite"),
    ]
    assert _detector(min_neighbors=3).detect(_chunk(), triples) == []


def test_d1_counts_neighbours_reached_in_either_direction():
    """A triple where the entity is the object still contributes a neighbour."""
    triples = [
        _t("iPhone", "made_by", "Apple"),
        _t("Steve Jobs", "co-founded", "Apple"),
        _t("fruit", "example_of", "Apple"),
    ]
    (defect,) = _detector().detect(_chunk(), triples)
    assert defect.metadata["entity"] == "Apple"


def test_d1_reports_one_defect_per_colliding_entity():
    triples = [
        *APPLE_MIXED,
        _t("Mercury", "is_a", "planet"),
        _t("Mercury", "orbits", "Sun"),
        _t("Mercury", "is_a", "element"),
        _t("Mercury", "symbol", "Hg"),
    ]
    defects = _detector().detect(_chunk(), triples)
    assert sorted(d.metadata["entity"] for d in defects) == ["Apple", "Mercury"]


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("min_neighbors", [0, -1])
def test_d1_rejects_a_non_positive_neighbour_floor(min_neighbors: int):
    with pytest.raises(ValueError, match="min_neighbors must be >= 1"):
        _detector(min_neighbors=min_neighbors)


@pytest.mark.parametrize("min_components", [1, 0])
def test_d1_rejects_a_component_floor_below_two(min_components: int):
    with pytest.raises(ValueError, match="min_components must be >= 2"):
        _detector(min_components=min_components)


def test_d1_component_floor_is_configurable():
    d = _detector(min_neighbors=2, min_components=3)
    two_components = [_t("X", "p", "a"), _t("X", "p", "b")]
    assert d.detect(_chunk(), two_components) == []
    three_components = [_t("X", "p", "a"), _t("X", "p", "b"), _t("X", "p", "c")]
    assert len(d.detect(_chunk(), three_components)) == 1


def test_d1_detector_exposes_its_code_as_a_class_attribute():
    from chimera_rag.plugins.prism.defects import D1CollisionDetector

    assert D1CollisionDetector.code == "D1"
