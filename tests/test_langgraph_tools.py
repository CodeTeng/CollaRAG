"""Tests for innovation 2b tool layer (:mod:`...langgraph.tools`).

These verify that each LangChain Tool is a *thin, correct* wrapper over the
existing retrieval primitives: argument validation, delegation to the right
underlying component, and accumulation into the shared :class:`ToolContext`.

All offline — MockProvider only, zero API keys.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")


def _build_env():
    from chimera_rag.core.types import Chunk, Triple
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
    vs.add([c.chunk_id for c in chunks], embedder.encode([c.text for c in chunks]))
    gs.add_triple(Triple(subject="Nobel", predicate="invented", object="Dynamite"))
    gs.add_triple(Triple(subject="Dynamite", predicate="used_in", object="Mining"))

    base = DirectRetriever(
        vector_store=vs,
        graph_store=gs,
        embedder=embedder,
        chunk_lookup={c.chunk_id: c for c in chunks},
        neighbor_hops=1,
    )
    return base, gs


def _build_tools():
    from chimera_rag.plugins.intent_agent.langgraph.tools import ToolContext, build_tools

    base, gs = _build_env()
    ctx = ToolContext()
    tools = build_tools(base_retriever=base, graph_store=gs, context=ctx, default_top_k=3)
    by_name = {t.name: t for t in tools}
    return by_name, ctx


# ---------------------------------------------------------------------------
# Factory / structure
# ---------------------------------------------------------------------------
def test_build_tools_returns_all_five_named_tools():
    from chimera_rag.plugins.intent_agent.langgraph.tools import ALL_TOOL_NAMES

    by_name, _ = _build_tools()
    assert set(by_name) == set(ALL_TOOL_NAMES)


def test_tools_are_structured_langchain_tools_with_schema():
    from langchain_core.tools import StructuredTool

    by_name, _ = _build_tools()
    for tool in by_name.values():
        assert isinstance(tool, StructuredTool)
        assert tool.description  # non-empty
        assert tool.args_schema is not None


# ---------------------------------------------------------------------------
# vector_search
# ---------------------------------------------------------------------------
def test_vector_search_returns_chunks_and_accumulates_context():
    by_name, ctx = _build_tools()
    out = by_name["vector_search"].invoke({"query": "Who invented dynamite?", "top_k": 2})
    assert out["tool"] == "vector_search"
    assert out["num_chunks"] >= 1
    assert all("chunk_id" in c and "preview" in c for c in out["chunks"])
    # Accumulated into shared context.
    assert len(ctx.collected) >= 1


def test_vector_search_validates_top_k_bounds():
    import pydantic

    by_name, _ = _build_tools()
    with pytest.raises((pydantic.ValidationError, ValueError, Exception)):
        by_name["vector_search"].invoke({"query": "x", "top_k": 0})


# ---------------------------------------------------------------------------
# graph_neighbors
# ---------------------------------------------------------------------------
def test_graph_neighbors_returns_connecting_triples():
    by_name, ctx = _build_tools()
    out = by_name["graph_neighbors"].invoke({"entity": "Nobel", "hops": 2})
    assert out["tool"] == "graph_neighbors"
    assert out["num_triples"] >= 1
    subjects = {t["subject"] for t in out["triples"]}
    assert "Nobel" in subjects or any("Dynamite" in (t["subject"], t["object"]) for t in out["triples"])
    assert len(ctx.triples) >= 1


def test_graph_neighbors_unknown_entity_is_empty():
    by_name, _ = _build_tools()
    out = by_name["graph_neighbors"].invoke({"entity": "NoSuchEntity", "hops": 1})
    assert out["num_triples"] == 0


# ---------------------------------------------------------------------------
# multi_hop / parallel delegate to existing strategies
# ---------------------------------------------------------------------------
def test_multi_hop_retrieve_delegates_to_strategy():
    by_name, ctx = _build_tools()
    out = by_name["multi_hop_retrieve"].invoke({"query": "impact of dynamite"})
    assert out["tool"] == "multi_hop_retrieve"
    assert out["num_chunks"] >= 1


def test_parallel_retrieve_handles_comparative_query():
    by_name, _ = _build_tools()
    out = by_name["parallel_retrieve"].invoke({"query": "Compare Nobel and Einstein"})
    assert out["tool"] == "parallel_retrieve"
    assert out["num_chunks"] >= 1


# ---------------------------------------------------------------------------
# assess_evidence uses the reflector hook
# ---------------------------------------------------------------------------
def test_assess_evidence_scores_accumulated_context():
    by_name, _ = _build_tools()
    # First gather some evidence so assessment is meaningful.
    by_name["vector_search"].invoke({"query": "Who invented dynamite?", "top_k": 2})
    out = by_name["assess_evidence"].invoke({"query": "Who invented dynamite?"})
    assert out["tool"] == "assess_evidence"
    assert 0.0 <= out["quality"] <= 1.0
    assert 0.0 <= out["relevance"] <= 1.0
    assert out["num_evidence_chunks"] >= 1


# ---------------------------------------------------------------------------
# ToolContext merge semantics
# ---------------------------------------------------------------------------
def test_tool_context_merges_and_dedupes_by_chunk_id():
    by_name, ctx = _build_tools()
    by_name["vector_search"].invoke({"query": "dynamite", "top_k": 3})
    by_name["vector_search"].invoke({"query": "dynamite nobel", "top_k": 3})
    merged = ctx.merged_result(top_k=3)
    ids = [c.chunk_id for c in merged.chunks]
    assert len(ids) == len(set(ids))  # no duplicate chunk ids
    # scores are sorted descending
    assert merged.scores == sorted(merged.scores, reverse=True)
