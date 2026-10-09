"""Tests for the simple "identity" / "no-op" default implementations.

These are the implementations used when a feature is turned off at the
config level. They must be absolute zero-risk: no side effects, no
coercion, no surprises.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# NoOpPruner
# ---------------------------------------------------------------------------
def test_noop_pruner_returns_input_unchanged():
    from chimera_rag.core.types import Triple
    from chimera_rag.defaults.pruner import NoOpPruner

    triples = [Triple(subject="a", predicate="p", object="b")]
    out = NoOpPruner().prune(triples)
    assert out == triples


def test_noop_pruner_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.pruner import NoOpPruner

    assert get_registry().get("pruner", "defaults.noop") is NoOpPruner


# ---------------------------------------------------------------------------
# RuleBasedIntentClassifier
# ---------------------------------------------------------------------------
def test_rule_based_intent_classifier_always_returns_factual():
    from chimera_rag.core.types import Query
    from chimera_rag.defaults.classifier import RuleBasedIntentClassifier

    c = RuleBasedIntentClassifier()
    for q in ["who invented dynamite?", "compare A and B", "walk me through ..."]:
        intent = c.classify(Query(text=q))
        assert intent.label == "factual"
        assert 0.0 <= intent.confidence <= 1.0

def test_rule_intent_classifier_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.classifier import RuleBasedIntentClassifier

    assert (
        get_registry().get("intent_classifier", "defaults.rule")
        is RuleBasedIntentClassifier
    )
