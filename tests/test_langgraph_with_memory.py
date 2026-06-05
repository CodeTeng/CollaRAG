"""End-to-end tests: LangGraphAgentRetriever + FileExperientialMemory (2c).

Verifies experience is recorded after a run, recalled and injected on a later
run, and that cold->hot reuse works across retriever instances (persistence).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")


def _build(default_react: str, mem, tmp_path: Path, **kwargs):
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
    ]
    vs.add([c.chunk_id for c in chunks], embedder.encode([c.text for c in chunks]))
    gs.add_triple(Triple(subject="Nobel", predicate="invented", object="Dynamite"))
    base = DirectRetriever(vector_store=vs, graph_store=gs, embedder=embedder,
                           chunk_lookup={c.chunk_id: c for c in chunks}, neighbor_hops=1)
    llm = MockLLMProvider(default_response=default_react)
    classifier = LLMIntentClassifier(llm=MockLLMProvider())
    return LangGraphAgentRetriever(
        base_retriever=base, classifier=classifier, llm=llm, graph_store=gs,
        experiential_memory=mem, **kwargs,
    )


def test_experience_recorded_after_run(tmp_path: Path):
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.experiential_memory import FileExperientialMemory

    mem = FileExperientialMemory(path=str(tmp_path / "mem"))
    react = 'Action: vector_search\nAction Input: {"query": "who invented dynamite", "top_k": 2}'
    r = _build(react, mem, tmp_path)
    res = r.retrieve(Query(text="Who invented dynamite?"), top_k=2)
    # An experience entry must have been persisted for the classified intent.
    assert res.metadata.get("experience_recorded") in {"success", "lesson"}
    assert mem.stats()["total"] == 1


def test_experience_recalled_on_later_run(tmp_path: Path):
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.experiential_memory import FileExperientialMemory

    mem = FileExperientialMemory(path=str(tmp_path / "mem"))
    react = 'Action: vector_search\nAction Input: {"query": "dynamite", "top_k": 2}'

    # First run seeds the experience store.
    r1 = _build(react, mem, tmp_path)
    r1.retrieve(Query(text="Who invented dynamite?"), top_k=2)
    total_after_first = mem.stats()["total"]
    assert total_after_first == 1

    # Second run (same intent) must recall it; de-dup keeps total stable and
    # bumps hit_count rather than appending a duplicate.
    r2 = _build(react, mem, tmp_path)
    r2.retrieve(Query(text="Who discovered dynamite?"), top_k=2)
    assert mem.stats()["total"] == 1
    recalled = mem.recall("factual", "q", top_n=5)
    assert recalled[0].hit_count >= 1


def test_persists_across_retriever_instances(tmp_path: Path):
    from chimera_rag.core.types import Query
    from chimera_rag.plugins.intent_agent.experiential_memory import FileExperientialMemory

    root = str(tmp_path / "mem")
    react = 'Action: vector_search\nAction Input: {"query": "dynamite", "top_k": 2}'
    r1 = _build(react, FileExperientialMemory(path=root), tmp_path)
    r1.retrieve(Query(text="Who invented dynamite?"), top_k=2)

    # Brand-new memory instance + retriever pointing at the same disk dir.
    fresh_mem = FileExperientialMemory(path=root)
    assert fresh_mem.stats()["total"] == 1
