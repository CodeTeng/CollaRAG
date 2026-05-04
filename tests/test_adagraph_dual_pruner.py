"""Tests for :mod:`chimera_rag.plugins.adagraph.dual_pruner`.

AdaGraph's pruner removes two kinds of noise:

* **similarity redundancy** — triples whose (s, p, o) surfaces are near
  duplicates (string similarity above a threshold, or embedding cosine
  above a threshold). This round implements the string-based variant
  since it keeps the zero-dep promise.
* **transitive redundancy** — if A->B and B->C imply A->C in some
  closure (e.g. 'invented_in' is transitive), the derivable A->C is
  removed. Reserved for a follow-up TDD cycle.
"""

from __future__ import annotations


def _t(subject: str, predicate: str, obj: str):
    from chimera_rag.core.types import Triple

    return Triple(subject=subject, predicate=predicate, object=obj)


def test_dual_pruner_implements_base_pruner_abc():
    from chimera_rag.interfaces.pruner import BasePruner
    from chimera_rag.plugins.adagraph.dual_pruner import DualRedundancyPruner

    assert isinstance(DualRedundancyPruner(), BasePruner)


def test_dual_pruner_removes_exact_duplicates():
    from chimera_rag.plugins.adagraph.dual_pruner import DualRedundancyPruner

    triples = [
        _t("Nobel", "invented", "dynamite"),
        _t("Nobel", "invented", "dynamite"),   # exact duplicate
        _t("Nobel", "founded", "NobelPrize"),
    ]
    out = DualRedundancyPruner().prune(triples)
    assert len(out) == 2


def test_dual_pruner_removes_near_duplicates_above_threshold():
    """Surface-string similarity above threshold -> drop the later one."""
    from chimera_rag.plugins.adagraph.dual_pruner import DualRedundancyPruner

    triples = [
        _t("Alfred Nobel", "invented", "dynamite"),
        _t("Alfred Nobel", "invented", "Dynamite"),  # only case differs
    ]
    out = DualRedundancyPruner(similarity_threshold=0.85).prune(triples)
    assert len(out) == 1


def test_dual_pruner_keeps_distinct_triples():
    from chimera_rag.plugins.adagraph.dual_pruner import DualRedundancyPruner

    triples = [
        _t("Nobel", "invented", "dynamite"),
        _t("Curie", "discovered", "radium"),
        _t("Einstein", "formulated", "relativity"),
    ]
    out = DualRedundancyPruner().prune(triples)
    assert len(out) == 3


def test_dual_pruner_transitive_flag_off_by_default():
    """Transitive pruning is reserved; flag must default False."""
    from chimera_rag.plugins.adagraph.dual_pruner import DualRedundancyPruner

    p = DualRedundancyPruner()
    assert p.enable_transitive is False


def test_dual_pruner_transitive_flag_on_does_not_crash():
    """Enabling transitive pruning must not alter results until implemented."""
    from chimera_rag.plugins.adagraph.dual_pruner import DualRedundancyPruner

    triples = [
        _t("A", "related_to", "B"),
        _t("B", "related_to", "C"),
    ]
    p = DualRedundancyPruner(enable_transitive=True)
    out = p.prune(triples)
    assert len(out) == 2  # reserved: no change


def test_dual_pruner_registers_under_slot():
    import chimera_rag.plugins.adagraph  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.adagraph.dual_pruner import DualRedundancyPruner

    assert (
        get_registry().get("pruner", "adagraph.dual_redundancy")
        is DualRedundancyPruner
    )


def test_dual_pruner_preserves_order_of_first_occurrence():
    """For deterministic ablation-study output, earlier triples win."""
    from chimera_rag.plugins.adagraph.dual_pruner import DualRedundancyPruner

    triples = [
        _t("Nobel", "invented", "dynamite"),
        _t("Curie", "discovered", "radium"),
        _t("Nobel", "invented", "dynamite"),  # dup of index 0
    ]
    out = DualRedundancyPruner().prune(triples)
    assert out[0].subject == "Nobel"
    assert out[1].subject == "Curie"
