"""Tests for :mod:`chimera_rag.plugins.intent_agent.agent`.

IntentAgentRetriever is the public face of innovation #2: registered
under the ``retriever`` slot as ``intent_agent.strategy_selector``. It
wires together the classifier, reasoner, strategies, and hooks and
returns a :class:`RetrievalResult` so it is a drop-in replacement for
the vanilla ``DirectRetriever``.
"""

from __future__ import annotations


def _build_env():
    from chimera_rag.core.types import Chunk, Triple
    from chimera_rag.defaults.direct_retriever import DirectRetriever
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider
    from chimera_rag.stores.graph.networkx_store import NetworkXGraphStore
    from chimera_rag.stores.vector.faiss_store import FAISSVectorStore

    embedder = MockEmbeddingProvider(dim=16, normalize=True)
    vs = FAISSVectorStore(dim=16, index_type="flat_ip")
    gs = NetworkXGraphStore()

    chunks = [
        Chunk(chunk_id="c0", doc_id="d", index=0, text="Alfred Nobel invented dynamite"),
        Chunk(chunk_id="c1", doc_id="d", index=1, text="Albert Einstein formulated relativity"),
    ]
    vs.add([c.chunk_id for c in chunks], embedder.encode([c.text for c in chunks]))
    gs.add_triple(Triple(subject="Nobel", predicate="invented", object="Dynamite"))

    base = DirectRetriever(
        vector_store=vs,
        graph_store=gs,
        embedder=embedder,
        chunk_lookup={c.chunk_id: c for c in chunks},
        neighbor_hops=1,
    )
    classifier = LLMIntentClassifier(llm=MockLLMProvider())  # falls back to rules
    return base, classifier


def test_intent_agent_retriever_implements_base_retriever_abc():
    from chimera_rag.interfaces.retriever import BaseRetriever
    from chimera_rag.plugins.intent_agent.agent import IntentAgentRetriever

    base, classifier = _build_env()
    r = IntentAgentRetriever(base_retriever=base, classifier=classifier)
    assert isinstance(r, BaseRetriever)


def test_intent_agent_retriever_classifies_factual_and_uses_direct_strategy():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.agent import IntentAgentRetriever

    base, classifier = _build_env()
    r = IntentAgentRetriever(base_retriever=base, classifier=classifier)

    res = r.retrieve(Query(text="Who invented dynamite?"), top_k=2)
    assert len(res.chunks) >= 1
    assert res.metadata.get("strategy") == "direct"
    assert res.metadata.get("intent") == "factual"


def test_intent_agent_retriever_uses_parallel_strategy_for_comparative_query():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.agent import IntentAgentRetriever

    base, classifier = _build_env()
    r = IntentAgentRetriever(base_retriever=base, classifier=classifier)

    res = r.retrieve(Query(text="Compare Nobel and Einstein"), top_k=5)
    assert res.metadata.get("intent") == "comparative"
    assert res.metadata.get("strategy") == "parallel"


def test_intent_agent_retriever_applies_reranker_when_provided():
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.agent import IntentAgentRetriever
    from chimera_rag.plugins.intent_agent.hooks.reranking_hook import (
        TokenOverlapRerankerHook,
    )

    base, classifier = _build_env()
    r = IntentAgentRetriever(
        base_retriever=base,
        classifier=classifier,
        reranker=TokenOverlapRerankerHook(),
    )
    res = r.retrieve(Query(text="dynamite nobel"), top_k=2)
    assert res.metadata.get("reranked_by") == "token_overlap"


def test_intent_agent_retriever_registers_under_slot():
    import chimera_rag.plugins.intent_agent  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.intent_agent.agent import IntentAgentRetriever

    assert (
        get_registry().get("retriever", "intent_agent.strategy_selector")
        is IntentAgentRetriever
    )
