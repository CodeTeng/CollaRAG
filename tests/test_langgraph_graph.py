"""Tests for the LangGraph StateGraph (:mod:`...langgraph.graph`).

Verifies graph wiring and — crucially — the **reflection loop**:
* a tool-call path executes a tool then reflects,
* high quality terminates immediately,
* low quality re-plans until ``max_iterations`` caps it,
* an immediate Final Answer skips tools.

Deterministic via MockLLMProvider returning canned ReAct text.
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")


def _env(default_react: str, reflector=None, max_iterations=2, quality_threshold=0.6):
    from chimera_rag.core.types import Chunk, Triple
    from chimera_rag.defaults.direct_retriever import DirectRetriever
    from chimera_rag.plugins.intent_agent.hooks.reflection_hook import HeuristicReflectionHook
    from chimera_rag.plugins.intent_agent.intent_classifier import LLMIntentClassifier
    from chimera_rag.plugins.intent_agent.langgraph.graph import build_agent_graph
    from chimera_rag.plugins.intent_agent.langgraph.llm_adapter import ChimeraReActChatModel
    from chimera_rag.plugins.intent_agent.langgraph.tools import ToolContext, build_tools
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

    ctx = ToolContext()
    tools = build_tools(base_retriever=base, graph_store=gs, context=ctx, default_top_k=3)
    chat = ChimeraReActChatModel(provider=MockLLMProvider(default_response=default_react))
    classifier = LLMIntentClassifier(llm=MockLLMProvider())
    refl = reflector or HeuristicReflectionHook()
    graph = build_agent_graph(
        chat_model=chat,
        tools=tools,
        context=ctx,
        classifier=classifier,
        reflector=refl,
        memory=None,
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
    )
    return graph, ctx


def _initial(query: str) -> dict:
    return {"query": query, "session_id": None}


def test_graph_executes_a_tool_then_reflects():
    react = 'Action: vector_search\nAction Input: {"query": "who invented dynamite", "top_k": 3}'
    graph, ctx = _env(default_react=react)
    final = graph.invoke(_initial("Who invented dynamite?"))
    assert final["intent"] == "factual"
    assert "vector_search" in final["tool_call_log"]
    assert "quality" in final
    assert len(ctx.collected) >= 1


def test_immediate_final_answer_skips_tools():
    graph, ctx = _env(default_react="Final Answer: Alfred Nobel.")
    final = graph.invoke(_initial("Who invented dynamite?"))
    assert final.get("tool_call_log") == []
    assert final["quality"] >= 0.0


def test_high_quality_reflector_terminates_after_one_pass():
    class AlwaysHigh:
        async def evaluate(self, query, answer):
            from chimera_rag.core.types import QualityFeedback
            return QualityFeedback(quality=0.95, relevance=1.0, completeness=1.0)

    react = 'Action: vector_search\nAction Input: {"query": "x", "top_k": 2}'
    graph, _ = _env(default_react=react, reflector=AlwaysHigh(), quality_threshold=0.6)
    final = graph.invoke(_initial("q"))
    assert final["iteration_count"] == 1
    assert final["quality"] >= 0.6


def test_low_quality_reflector_caps_at_max_iterations():
    class AlwaysLow:
        async def evaluate(self, query, answer):
            from chimera_rag.core.types import QualityFeedback
            return QualityFeedback(quality=0.0, relevance=0.0, completeness=0.0)

    react = 'Action: vector_search\nAction Input: {"query": "x", "top_k": 2}'
    graph, _ = _env(default_react=react, reflector=AlwaysLow(),
                    quality_threshold=0.9, max_iterations=2)
    final = graph.invoke(_initial("q"), {"recursion_limit": 50})
    # Should re-plan until the iteration cap, then stop.
    assert final["iteration_count"] == 2


def test_graph_compiles_with_all_five_nodes():
    react = "Final Answer: ok"
    graph, _ = _env(default_react=react)
    node_names = set(graph.get_graph().nodes)
    for expected in {"memory", "reasoning", "decision", "tools", "reflect"}:
        assert expected in node_names
