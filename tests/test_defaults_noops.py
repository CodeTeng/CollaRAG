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
    from chimera_rag.defaults.noop_pruner import NoOpPruner

    triples = [Triple(subject="a", predicate="p", object="b")]
    out = NoOpPruner().prune(triples)
    assert out == triples


def test_noop_pruner_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401

    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.noop_pruner import NoOpPruner

    assert get_registry().get("pruner", "defaults.noop") is NoOpPruner


# ---------------------------------------------------------------------------
# RuleBasedIntentClassifier
# ---------------------------------------------------------------------------
def test_rule_based_intent_classifier_always_returns_factual():
    from chimera_rag.core.types import Query
    from chimera_rag.defaults.rule_intent_classifier import RuleBasedIntentClassifier

    c = RuleBasedIntentClassifier()
    for q in ["who invented dynamite?", "compare A and B", "walk me through ..."]:
        intent = c.classify(Query(text=q))
        assert intent.label == "factual"
        assert 0.0 <= intent.confidence <= 1.0

def test_rule_intent_classifier_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401

    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.rule_intent_classifier import RuleBasedIntentClassifier

    assert (
        get_registry().get("intent_classifier", "defaults.rule")
        is RuleBasedIntentClassifier
    )


# ---------------------------------------------------------------------------
# NoMemory
# ---------------------------------------------------------------------------
def test_no_memory_is_stateless():
    from chimera_rag.defaults.no_memory import NoMemory

    m = NoMemory()
    m.append("s", "hi")
    assert m.get_history("s") == []
    m.clear("s")
    assert m.get_history("s") == []


def test_no_memory_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401

    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.no_memory import NoMemory

    assert get_registry().get("memory", "defaults.none") is NoMemory


# ---------------------------------------------------------------------------
# NoOpReflector
# ---------------------------------------------------------------------------
async def test_noop_reflector_always_reports_perfect_quality():
    from chimera_rag.core.types import Answer, Query
    from chimera_rag.defaults.no_reflection import NoOpReflector

    fb = await NoOpReflector().evaluate(Query(text="x"), Answer(text="y"))
    assert fb.quality == 1.0


def test_noop_reflector_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401

    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.no_reflection import NoOpReflector

    assert get_registry().get("reflector", "defaults.noop") is NoOpReflector


# ---------------------------------------------------------------------------
# NoOpReranker
# ---------------------------------------------------------------------------
async def test_noop_reranker_preserves_order():
    from chimera_rag.core.types import Chunk, Query, RetrievalResult
    from chimera_rag.defaults.no_reranker import NoOpReranker

    ch = [Chunk(chunk_id=f"c{i}", doc_id="d", index=i, text=str(i)) for i in range(3)]
    res = RetrievalResult(chunks=ch, scores=[0.9, 0.8, 0.7])
    out = await NoOpReranker().rerank(Query(text="q"), res)
    assert [c.chunk_id for c in out.chunks] == ["c0", "c1", "c2"]
    assert out.scores == [0.9, 0.8, 0.7]


def test_noop_reranker_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401

    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.no_reranker import NoOpReranker

    assert get_registry().get("reranker", "defaults.noop") is NoOpReranker
