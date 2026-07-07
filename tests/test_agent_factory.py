"""Tests for AgentFactory and ToolSetBuilder."""
from langchain_core.tools import StructuredTool

from chimera_rag.plugins.intent_agent.agent_factory import (
    TOOL_MATRIX,
    AgentFactory,
    ToolSetBuilder,
)


def _dummy_tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(lambda: "ok", name=name, description=f"dummy {name}")


def test_tool_matrix_has_all_agents():
    assert "preprocessor" in TOOL_MATRIX
    assert "single_hop" in TOOL_MATRIX
    assert "multi_hop" in TOOL_MATRIX
    assert "summarization" in TOOL_MATRIX
    assert "other" in TOOL_MATRIX


def test_tool_set_builder_filters():
    all_tools = [_dummy_tool("vector_search"), _dummy_tool("web_search"), _dummy_tool("bm25_search")]
    filtered = ToolSetBuilder.for_agent("single_hop", all_tools)
    names = {t.name for t in filtered}
    assert "vector_search" in names
    assert "bm25_search" in names
    assert "web_search" not in names


def test_tool_set_builder_unknown_agent():
    all_tools = [_dummy_tool("vector_search")]
    filtered = ToolSetBuilder.for_agent("nonexistent_agent", all_tools)
    assert filtered == []


def test_agent_factory_register_and_create():
    from chimera_rag.core.types import Answer, Query
    from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent

    @AgentFactory.register("_test_intent")
    class TestAgent(BaseAgent):
        def do_execute(self, query: Query, context: AgentContext) -> Answer:
            return Answer(text="test")

    all_tools = [_dummy_tool("vector_search"), _dummy_tool("hybrid_search")]
    agent = AgentFactory.create("_test_intent", all_tools=all_tools, memory_manager=None, config={})
    assert agent.agent_type == "_test_intent"
