"""Tests for the ReAct ChatModel adapter (:mod:`...langgraph.llm_adapter`).

Verifies that the adapter:
* is a valid LangChain ChatModel that wraps a Chimera-RAG provider,
* parses ``Action: / Action Input:`` text into AIMessage.tool_calls,
* parses ``Final Answer:`` text into plain content,
* degrades gracefully on malformed output.

All deterministic via MockLLMProvider (canned ReAct strings).
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")


def _model(rules=None, default="Final Answer: nothing"):
    from chimera_rag.plugins.intent_agent.langgraph.llm_adapter import ChimeraReActChatModel
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    provider = MockLLMProvider(rules=rules or [], default_response=default)
    return ChimeraReActChatModel(provider=provider)


def _tools():
    from chimera_rag.plugins.intent_agent.langgraph.tools import ToolContext, build_tools
    from chimera_rag.core.types import Chunk, Triple
    from chimera_rag.defaults.direct_retriever import DirectRetriever
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider
    from chimera_rag.stores.graph.networkx_store import NetworkXGraphStore
    from chimera_rag.stores.vector.faiss_store import FAISSVectorStore

    embedder = MockEmbeddingProvider(dim=16, normalize=True)
    vs = FAISSVectorStore(dim=16, index_type="flat_ip")
    gs = NetworkXGraphStore()
    chunks = [Chunk(chunk_id="c0", doc_id="d", index=0, text="Nobel invented dynamite")]
    vs.add(["c0"], embedder.encode(["Nobel invented dynamite"]))
    base = DirectRetriever(vector_store=vs, graph_store=gs, embedder=embedder,
                           chunk_lookup={c.chunk_id: c for c in chunks})
    return build_tools(base_retriever=base, graph_store=gs, context=ToolContext())


def test_adapter_is_a_chat_model():
    from langchain_core.language_models.chat_models import BaseChatModel

    assert isinstance(_model(), BaseChatModel)
    assert _model()._llm_type == "chimera_react"


def test_bind_tools_returns_model_with_tools():
    m = _model().bind_tools(_tools())
    assert len(m.bound_tools) == 5


def test_parses_action_into_tool_call():
    from langchain_core.messages import HumanMessage

    react = 'Action: vector_search\nAction Input: {"query": "who invented dynamite", "top_k": 3}'
    m = _model(default=react).bind_tools(_tools())
    result = m.invoke([HumanMessage(content="who invented dynamite?")])
    assert result.tool_calls, "expected a tool_call to be produced"
    call = result.tool_calls[0]
    assert call["name"] == "vector_search"
    assert call["args"]["query"] == "who invented dynamite"
    assert call["args"]["top_k"] == 3


def test_parses_final_answer_into_content():
    from langchain_core.messages import HumanMessage

    m = _model(default="Final Answer: Alfred Nobel invented dynamite.")
    result = m.invoke([HumanMessage(content="who?")])
    assert not result.tool_calls
    assert "Nobel" in result.content


def test_final_answer_preferred_when_before_action():
    from langchain_core.messages import HumanMessage

    raw = "Final Answer: done\nAction: vector_search\nAction Input: {}"
    m = _model(default=raw)
    result = m.invoke([HumanMessage(content="q")])
    assert not result.tool_calls
    assert result.content == "done"


def test_malformed_output_degrades_to_plain_content():
    from langchain_core.messages import HumanMessage

    m = _model(default="I have no idea what format to use")
    result = m.invoke([HumanMessage(content="q")])
    assert not result.tool_calls
    assert result.content


def test_action_input_fallback_to_query_when_not_json():
    from langchain_core.messages import HumanMessage

    raw = "Action: vector_search\nAction Input: who invented dynamite"
    m = _model(default=raw).bind_tools(_tools())
    result = m.invoke([HumanMessage(content="q")])
    assert result.tool_calls[0]["args"]["query"] == "who invented dynamite"
