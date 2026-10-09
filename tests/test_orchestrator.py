"""Tests for MultiAgentRetriever orchestrator."""
import numpy as np

from chimera_rag.core.types import Chunk, Query, RetrievalResult


class MockLLM:
    async def complete(self, prompt: str, **kw) -> str:
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

    # Graph-footprint probes (C4). The mock graph is empty, so these return
    # graph-poor values and the toolset falls back to the static matrix.
    def entity_exists(self, entity):
        return False

    def shortest_path(self, a, b):
        return None

    def schema_predicates(self):
        return []


class MockEmbedder:
    def encode(self, texts):
        return np.random.randn(len(texts), 64).astype(np.float32)


def _make_retriever(tmp_path):
    from chimera_rag.plugins.colla_rag.orchestrator import MultiAgentRetriever
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


def test_derive_lesson_multihop_includes_gaps():
    from chimera_rag.plugins.colla_rag.orchestrator import MultiAgentRetriever

    class FakeAnswer:
        text = "draft"
        trace = {
            "quality": 0.2, "reflect_count": 1,
            "structural_gaps": [
                {"kind": "missing_path", "detail": "no Einstein-Bohr edge", "repair_query": "x"},
            ],
        }

    q = Query(text="Compare Einstein and Bohr")
    lesson = MultiAgentRetriever._derive_lesson("multi_hop", q, FakeAnswer(), 0.2)
    assert lesson.startswith("low_quality q=0.20")
    assert "reflect=1" in lesson
    assert "missing_path" in lesson
    assert "no Einstein-Bohr edge" in lesson


def test_derive_lesson_non_multihop_is_generic():
    from chimera_rag.plugins.colla_rag.orchestrator import MultiAgentRetriever

    class FakeAnswer:
        trace = {"quality": 0.3}

    q = Query(text="What is the capital of France?")
    lesson = MultiAgentRetriever._derive_lesson("single_hop", q, FakeAnswer(), 0.3)
    assert lesson.startswith("low_quality q=0.30")
    assert "capital of France" in lesson


def test_experience_entry_tags_lesson_and_carries_text():
    from chimera_rag.plugins.colla_rag.orchestrator import MultiAgentRetriever

    class FakeAnswer:
        text = "bad answer"
        trace = {"quality": 0.2, "tool_call_log": ["vector_search"], "reflect_count": 0,
                 "structural_gaps": []}

    q = Query(text="Q")
    entry = MultiAgentRetriever._experience_entry(
        MultiAgentRetriever.__new__(MultiAgentRetriever), "multi_hop", q, FakeAnswer(),
    )
    assert entry["outcome"] == "lesson"
    assert entry["final_quality"] == 0.2
    assert "lesson" in entry and entry["lesson"].startswith("low_quality")
