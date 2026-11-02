"""Tests for the D5 (budget-truncation) chunk defect detector.

D5 is PrismRAG's zero-cost signal: ``SimpleLLMExtractor`` caps extraction at
``max_triples_per_chunk`` (default 30) and silently discards the overflow at
``defaults/extractor.py:75``. Hitting that cap is direct evidence that the
chunk is too dense to be a single extraction unit, and no method currently
uses it. Repair action: ``split`` on the fine layer.
"""

from __future__ import annotations

import pytest

from chimera_rag.core.types import Chunk, Triple


def _chunk(text: str = "x", **kw) -> Chunk:
    return Chunk(chunk_id=kw.pop("chunk_id", "d::0"), doc_id=kw.pop("doc_id", "d"), index=0, text=text, **kw)


def _triples(n: int, chunk_id: str = "d::0") -> list[Triple]:
    return [
        Triple(subject=f"s{i}", predicate="p", object=f"o{i}", source_chunk_id=chunk_id)
        for i in range(n)
    ]


def _detector(**kw):
    from chimera_rag.plugins.prism.defects import D5BudgetDetector

    return D5BudgetDetector(**kw)


# ---------------------------------------------------------------------------
# The core signal: hitting the cap
# ---------------------------------------------------------------------------
def test_d5_flags_a_chunk_that_exactly_hits_the_budget():
    defects = _detector(budget=30).detect(_chunk(), _triples(30))
    assert len(defects) == 1
    assert defects[0].code == "D5"
    assert defects[0].action == "split"
    assert defects[0].chunk_id == "d::0"


def test_d5_flags_a_chunk_that_overshoots_the_budget():
    """A hand-fed triple list longer than the cap is still D5."""
    assert len(_detector(budget=10).detect(_chunk(), _triples(25))) == 1


def test_d5_stays_silent_just_below_the_budget():
    assert _detector(budget=30).detect(_chunk(), _triples(29)) == []


def test_d5_stays_silent_for_a_sparse_chunk():
    assert _detector(budget=30).detect(_chunk(), _triples(3)) == []


def test_d5_stays_silent_for_an_empty_chunk():
    assert _detector(budget=30).detect(_chunk(), []) == []


def test_d5_emits_at_most_one_defect_per_chunk():
    """One chunk is one extraction unit, so one split instruction."""
    assert len(_detector(budget=5).detect(_chunk(), _triples(40))) == 1


# ---------------------------------------------------------------------------
# ratio: an early-warning threshold below the hard cap
# ---------------------------------------------------------------------------
def test_d5_ratio_lowers_the_threshold():
    d = _detector(budget=30, ratio=0.5)
    assert d.detect(_chunk(), _triples(15)) != []
    assert d.detect(_chunk(), _triples(14)) == []


def test_d5_threshold_rounds_up_so_a_fractional_ratio_never_means_zero():
    # ceil(9 * 0.5) = 5
    d = _detector(budget=9, ratio=0.5)
    assert d.detect(_chunk(), _triples(5)) != []
    assert d.detect(_chunk(), _triples(4)) == []


def test_d5_ratio_one_is_the_default_hard_cap():
    assert _detector(budget=12).ratio == 1.0


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("budget", [0, -1])
def test_d5_rejects_a_non_positive_budget(budget: int):
    with pytest.raises(ValueError, match="budget must be >= 1"):
        _detector(budget=budget)


@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5])
def test_d5_rejects_a_ratio_outside_zero_to_one(ratio: float):
    with pytest.raises(ValueError, match="ratio must be in"):
        _detector(budget=30, ratio=ratio)


# ---------------------------------------------------------------------------
# Traceability: the defect must say why it fired
# ---------------------------------------------------------------------------
def test_d5_defect_metadata_carries_the_counts():
    (defect,) = _detector(budget=30).detect(_chunk(), _triples(30))
    assert defect.metadata["triple_count"] == 30
    assert defect.metadata["budget"] == 30
    assert defect.metadata["threshold"] == 30


def test_d5_detail_mentions_the_cap():
    (defect,) = _detector(budget=7).detect(_chunk(), _triples(7))
    assert "7" in defect.detail


# ---------------------------------------------------------------------------
# Independence: D5 must not need the graph, the sibling chunks or any LLM call
# ---------------------------------------------------------------------------
def test_d5_needs_no_audit_context_at_all():
    assert len(_detector(budget=3).detect(_chunk(), _triples(3))) == 1


def test_d5_ignores_a_graph_bearing_context():
    from chimera_rag.plugins.prism.defects import AuditContext
    from chimera_rag.stores.graph import NetworkXGraphStore

    g = NetworkXGraphStore()
    for t in _triples(3):
        g.add_triple(t)
    ctx = AuditContext(graph=g)

    assert len(_detector(budget=3).detect(_chunk(), _triples(3), ctx)) == 1


def test_d5_detector_exposes_its_code_as_a_class_attribute():
    from chimera_rag.plugins.prism.defects import D5BudgetDetector

    assert D5BudgetDetector.code == "D5"
