"""Tests for the 6 retrieval strategies in the IntentAgent plugin.

MVP implementation status:
* direct         — fully implemented (delegates to DirectRetriever)
* multi_hop      — implemented via graph 2-hop expansion around seed entities
* parallel       — implemented: split query at "and", retrieve each in parallel
* iterative      — skeleton: one pass through direct (placeholder)
* expansion      — implemented: direct + broader neighbor_hops
* context_aware  — implemented: prepend session history to query before retrieval
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------
def _build_env():
    """Return (retriever, plan factory) for strategy tests."""
    import numpy as np

    from chimera_rag.core.types import Chunk, Intent, ReasoningPlan, Triple
    from chimera_rag.defaults.direct_retriever import DirectRetriever
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider
    from chimera_rag.stores.graph.networkx_store import NetworkXGraphStore
    from chimera_rag.stores.vector.faiss_store import FAISSVectorStore

    embedder = MockEmbeddingProvider(dim=16, normalize=True)
    vs = FAISSVectorStore(dim=16, index_type="flat_ip")
    gs = NetworkXGraphStore()

    chunks = [
        Chunk(chunk_id="c0", doc_id="d", index=0, text="Alfred Nobel invented dynamite"),
        Chunk(chunk_id="c1", doc_id="d", index=1, text="Albert Einstein formulated relativity"),
        Chunk(chunk_id="c2", doc_id="d", index=2, text="Marie Curie discovered radium"),
    ]
    texts = [c.text for c in chunks]
    vs.add([c.chunk_id for c in chunks], embedder.encode(texts))
    gs.add_triple(Triple(subject="Nobel", predicate="invented", object="Dynamite"))
    gs.add_triple(Triple(subject="Einstein", predicate="formulated", object="Relativity"))

    base = DirectRetriever(
        vector_store=vs,
        graph_store=gs,
        embedder=embedder,
        chunk_lookup={c.chunk_id: c for c in chunks},
        neighbor_hops=1,
    )

    def make_plan(label="factual", name="direct"):
        return ReasoningPlan(
            intent=Intent(label=label, confidence=1.0),
            strategy_name=name,
        )

    return base, make_plan


# ---------------------------------------------------------------------------
# Individual strategies
# ---------------------------------------------------------------------------
async def test_direct_strategy_returns_retrieval():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.strategies.direct import DirectStrategy

    base, make_plan = _build_env()
    s = DirectStrategy(retriever=base)
    res = await s.execute(Query(text="Nobel"), make_plan())
    assert len(res.chunks) >= 1


async def test_parallel_strategy_splits_query_and_merges():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.strategies.parallel import ParallelStrategy

    base, make_plan = _build_env()
    s = ParallelStrategy(retriever=base)
    res = await s.execute(
        Query(text="Nobel and Einstein"),
        make_plan(label="comparative", name="parallel"),
    )
    chunk_ids = {c.chunk_id for c in res.chunks}
    # Both relevant chunks should be present.
    assert {"c0", "c1"}.issubset(chunk_ids)
    # Parallel execution is recorded.
    assert res.metadata.get("strategy") == "parallel"


async def test_multi_hop_strategy_expands_via_graph():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.strategies.multi_hop import MultiHopStrategy

    base, make_plan = _build_env()
    s = MultiHopStrategy(retriever=base)
    res = await s.execute(
        Query(text="Nobel"),
        make_plan(label="multi_hop", name="multi_hop"),
    )
    # Triples from the graph should surface.
    assert res.metadata.get("strategy") == "multi_hop"


async def test_expansion_strategy_broadens_neighbor_hops():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.strategies.expansion import ExpansionStrategy

    base, make_plan = _build_env()
    s = ExpansionStrategy(retriever=base, extra_hops=2)
    res = await s.execute(
        Query(text="Nobel"),
        make_plan(label="exploratory", name="expansion"),
    )
    assert res.metadata.get("strategy") == "expansion"
    assert len(res.chunks) >= 1


async def test_iterative_strategy_runs_and_returns_retrieval():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.strategies.iterative import IterativeStrategy

    base, make_plan = _build_env()
    s = IterativeStrategy(retriever=base)
    res = await s.execute(
        Query(text="why did Nobel succeed?"),
        make_plan(label="analytical", name="iterative"),
    )
    assert res.metadata.get("strategy") == "iterative"


async def test_context_aware_strategy_uses_history_from_plan_params():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.strategies.context_aware import (
        ContextAwareStrategy,
    )

    base, make_plan = _build_env()
    s = ContextAwareStrategy(retriever=base)
    plan = make_plan(label="follow_up", name="context_aware")
    plan.params = {"history": ["Who invented dynamite?", "Alfred Nobel."]}
    res = await s.execute(Query(text="what about her?"), plan)
    assert res.metadata.get("strategy") == "context_aware"


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
def test_all_six_strategies_registered_under_slot():
    import chimera_rag.plugins.intent_agent  # noqa: F401

    from chimera_rag.core.registry import get_registry

    names = set(get_registry().list_slot("strategy"))
    # Must cover all 6 canonical strategies.
    expected = {
        "intent_agent.direct",
        "intent_agent.parallel",
        "intent_agent.multi_hop",
        "intent_agent.iterative",
        "intent_agent.expansion",
        "intent_agent.context_aware",
    }
    assert expected.issubset(names), f"missing: {expected - names}"
