"""Degradation tests for innovation #2c.

Ensures the retriever behaves identically to pure 2b when no experiential
memory is injected, and that the store handles a non-existent directory
gracefully (auto-creates it).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")


def _build_without_memory(default_react: str):
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
    chunks = [Chunk(chunk_id="c0", doc_id="d", index=0, text="Alfred Nobel invented dynamite")]
    vs.add([c.chunk_id for c in chunks], embedder.encode([c.text for c in chunks]))
    gs.add_triple(Triple(subject="Nobel", predicate="invented", object="Dynamite"))
    base = DirectRetriever(vector_store=vs, graph_store=gs, embedder=embedder,
                           chunk_lookup={c.chunk_id: c for c in chunks}, neighbor_hops=1)
    llm = MockLLMProvider(default_response=default_react)
    classifier = LLMIntentClassifier(llm=MockLLMProvider())
    return LangGraphAgentRetriever(
        base_retriever=base, classifier=classifier, llm=llm, graph_store=gs,
    )


def test_no_memory_behaves_like_pure_2b():
    from chimera_rag.core.types import Query

    react = 'Action: vector_search\nAction Input: {"query": "dynamite", "top_k": 2}'
    r = _build_without_memory(react)
    assert r.experiential_memory is None
    res = r.retrieve(Query(text="Who invented dynamite?"), top_k=2)
    # No experience metadata written when memory is absent.
    assert "experience_recorded" not in res.metadata
    assert res.metadata["retriever"] == "langgraph"
    assert len(res.chunks) >= 1


def test_store_autocreates_missing_directory(tmp_path: Path):
    from chimera_rag.plugins.intent_agent.experiential_memory import FileExperientialMemory

    target = tmp_path / "does" / "not" / "exist"
    assert not target.exists()
    mem = FileExperientialMemory(path=str(target))
    # Constructor must create the nested experience dir without error.
    assert (target / "experience").exists()
    assert mem.stats()["total"] == 0


def test_invalid_max_per_bucket_rejected(tmp_path: Path):
    from chimera_rag.plugins.intent_agent.experiential_memory import FileExperientialMemory

    with pytest.raises(ValueError):
        FileExperientialMemory(path=str(tmp_path / "m"), max_per_bucket=0)
