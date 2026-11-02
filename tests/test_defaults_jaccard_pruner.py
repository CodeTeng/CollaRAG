"""Tests for :class:`chimera_rag.defaults.pruner.JaccardPruner`.

A generic near-duplicate filter over extracted triples: two triples whose
(s, p, o) surfaces have a token-Jaccard similarity at or above the
threshold collapse to the earlier one. Token Jaccard keeps it free of
optional dependencies.
"""

from __future__ import annotations

import pytest


def _t(subject: str, predicate: str, obj: str):
    from chimera_rag.core.types import Triple

    return Triple(subject=subject, predicate=predicate, object=obj)


def test_jaccard_pruner_implements_base_pruner_abc():
    from chimera_rag.defaults.pruner import JaccardPruner
    from chimera_rag.interfaces.pruner import BasePruner

    assert isinstance(JaccardPruner(), BasePruner)


def test_jaccard_pruner_removes_exact_duplicates():
    from chimera_rag.defaults.pruner import JaccardPruner

    triples = [
        _t("Nobel", "invented", "dynamite"),
        _t("Nobel", "invented", "dynamite"),
        _t("Nobel", "founded", "NobelPrize"),
    ]
    assert len(JaccardPruner().prune(triples)) == 2


def test_jaccard_pruner_removes_near_duplicates_above_threshold():
    """Surface-string similarity above threshold -> drop the later one."""
    from chimera_rag.defaults.pruner import JaccardPruner

    triples = [
        _t("Alfred Nobel", "invented", "dynamite"),
        _t("Alfred Nobel", "invented", "Dynamite"),  # only case differs
    ]
    assert len(JaccardPruner(similarity_threshold=0.85).prune(triples)) == 1


def test_jaccard_pruner_keeps_distinct_triples():
    from chimera_rag.defaults.pruner import JaccardPruner

    triples = [
        _t("Nobel", "invented", "dynamite"),
        _t("Curie", "discovered", "radium"),
        _t("Einstein", "formulated", "relativity"),
    ]
    assert len(JaccardPruner().prune(triples)) == 3


def test_jaccard_pruner_preserves_order_of_first_occurrence():
    """For deterministic ablation-study output, earlier triples win."""
    from chimera_rag.defaults.pruner import JaccardPruner

    triples = [
        _t("Nobel", "invented", "dynamite"),
        _t("Curie", "discovered", "radium"),
        _t("Nobel", "invented", "dynamite"),  # dup of index 0
    ]
    out = JaccardPruner().prune(triples)
    assert out[0].subject == "Nobel"
    assert out[1].subject == "Curie"


def test_jaccard_pruner_rejects_out_of_range_threshold():
    from chimera_rag.defaults.pruner import JaccardPruner

    with pytest.raises(ValueError, match=r"similarity_threshold must be in \(0, 1\]"):
        JaccardPruner(similarity_threshold=0.0)
    with pytest.raises(ValueError, match=r"similarity_threshold must be in \(0, 1\]"):
        JaccardPruner(similarity_threshold=1.5)


def test_jaccard_pruner_threshold_one_only_collapses_identical_token_sets():
    from chimera_rag.defaults.pruner import JaccardPruner

    triples = [
        _t("Nobel", "invented", "dynamite"),
        _t("Nobel", "invented", "Dynamite"),  # identical once lowercased
        _t("Nobel", "invented", "the dynamite"),  # one extra token -> not identical
    ]
    assert len(JaccardPruner(similarity_threshold=1.0).prune(triples)) == 2


def test_jaccard_pruner_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.pruner import JaccardPruner

    assert get_registry().get("pruner", "defaults.jaccard") is JaccardPruner
