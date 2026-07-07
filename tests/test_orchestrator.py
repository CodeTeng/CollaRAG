"""Tests for MultiAgentRetriever orchestrator."""
import numpy as np
from chimera_rag.core.types import Chunk, Query, RetrievalResult


class MockLLM:
    async def generate(self, prompt: str, **kw) -> str:
        if "category" in prompt.lower() or "classify" in prompt.lower():
            return "single_hop"
        return "Mock answer based on evidence."


class MockVectorStore:
    def search(self, qvec, top_k=5):
        return [("c1", 0.9)]

    def add(self, ids, vecs):
        pass


class MockGraphStore:
    def get_neighbors(self, entity, hops=1):
        return []

    def all_triples(self):
        return []


class MockEmbedder:
    def encode(self, texts):
        return np.random.randn(len(texts), 64).astype(np.float32)


def _make_retriever(tmp_path):
    from chimera_rag.plugins.intent_agent.orchestrator import MultiAgentRetriever
    chunks = {"c1": Chunk(chunk_id="c1", doc_id="d1", index=0, text="Test chunk about Einstein")}
    return MultiAgentRetriever(
        llm=MockLLM(),
        vector_store=MockVectorStore(),
        graph_store=MockGraphStore(),
        embedder=MockEmbedder(),
        chunk_lookup=chunks,
        memory_path=str(tmp_path / "memory"),
    )


def test_orchestrator_creation(tmp_path):
    r = _make_retriever(tmp_path)
    assert r is not None


def test_orchestrator_greeting(tmp_path):
    r = _make_retriever(tmp_path)
    result = r.retrieve(Query(text="你好"), top_k=5)
    assert isinstance(result, RetrievalResult)
    assert result.metadata.get("agent") == "greeting"


def test_orchestrator_single_hop(tmp_path):
    r = _make_retriever(tmp_path)
    result = r.retrieve(Query(text="Where was Einstein born?"), top_k=5)
    assert isinstance(result, RetrievalResult)
    assert result.metadata.get("intent") == "single_hop"
    assert "answer_text" in result.metadata
