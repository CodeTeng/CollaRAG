"""Tests for LangGraphAgentRetriever (:mod:`...langgraph.retriever`).

Verifies the BaseRetriever contract: registration under the slot, top_k
honouring, metadata provenance, and the empty-evidence fallback.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")


def _build_retriever(default_react: str, **kwargs):
    from chimera_rag.core.types import Chunk, Triple
    from chimera_rag.defaults.direct_retriever import DirectRetriever
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier
    from chimera_rag.plugins.intent_agent.langgraph.retriever import LangGraphAgentRetriever
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider
    from chimera_rag.stores.graph.networkx_store import NetworkXGraphStore
    from chimera_rag.stores.vector.faiss_store import FAISSVectorStore

    embedder = MockEmbeddingProvider(dim=16, normalize=True)
    vs = FAISSVectorStore(dim=16, index_type="flat_ip")
    gs = NetworkXGraphStore()
    chunks = [
        Chunk(chunk_id="c0", doc_id="d", index=0, text="Alfred Nobel invented dynamite"),
        Chunk(chunk_id="c1", doc_id="d", index=1, text="Einstein formulated relativity"),
        Chunk(chunk_id="c2", doc_id="d", index=2, text="Curie discovered radium"),
    ]
    vs.add([c.chunk_id for c in chunks], embedder.encode([c.text for c in chunks]))
    gs.add_triple(Triple(subject="Nobel", predicate="invented", object="Dynamite"))
    base = DirectRetriever(vector_store=vs, graph_store=gs, embedder=embedder,
                           chunk_lookup={c.chunk_id: c for c in chunks}, neighbor_hops=1)
    llm = MockLLMProvider(default_response=default_react)
    classifier = LLMIntentClassifier(llm=MockLLMProvider())
    return LangGraphAgentRetriever(
        base_retriever=base, classifier=classifier, llm=llm, graph_store=gs, **kwargs
    )


def test_implements_base_retriever_abc():
    from chimera_rag.interfaces.retriever import BaseRetriever

    r = _build_retriever('Final Answer: ok')
    assert isinstance(r, BaseRetriever)


def test_registers_under_langgraph_slot():
    import chimera_rag.plugins.intent_agent.langgraph  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.intent_agent.langgraph.retriever import LangGraphAgentRetriever

    assert get_registry().get("retriever", "intent_agent.langgraph") is LangGraphAgentRetriever


def test_retrieve_returns_chunks_and_langgraph_metadata():
    from chimera_rag.core.types import Query

    react = 'Action: vector_search\nAction Input: {"query": "who invented dynamite", "top_k": 3}'
    r = _build_retriever(react)
    res = r.retrieve(Query(text="Who invented dynamite?"), top_k=2)
    assert len(res.chunks) >= 1
    assert res.metadata["retriever"] == "langgraph"
    assert res.metadata["strategy"] == "langgraph_agent"
    assert res.metadata["intent"] == "factual"
    assert "vector_search" in res.metadata["tool_calls"]


def test_retrieve_honours_top_k():
    from chimera_rag.core.types import Query

    react = 'Action: vector_search\nAction Input: {"query": "dynamite relativity radium", "top_k": 3}'
    r = _build_retriever(react)
    res = r.retrieve(Query(text="tell me everything"), top_k=1)
    assert len(res.chunks) <= 1


def test_empty_evidence_falls_back_to_direct():
    from chimera_rag.core.types import Query

    # Final Answer immediately -> no tools called -> no evidence accumulated.
    r = _build_retriever("Final Answer: I already know.")
    res = r.retrieve(Query(text="Who invented dynamite?"), top_k=2)
    # Fallback to base retriever guarantees non-empty result.
    assert len(res.chunks) >= 1
    assert res.metadata["retriever"] == "langgraph"


def test_unknown_tool_names_are_filtered():
    r = _build_retriever("Final Answer: ok", tools=["vector_search", "no_such_tool"])
    assert r.enabled_tools == ["vector_search"]
