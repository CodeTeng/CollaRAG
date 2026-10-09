"""Tests for AgentFactory and ToolSetBuilder."""
from langchain_core.tools import StructuredTool

from chimera_rag.plugins.colla_rag.agent_factory import (
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


# ---------------------------------------------------------------------------
# C4: Graph-Conditioned Capability Specialization
# ---------------------------------------------------------------------------
def _fake_tool(name: str):
    from langchain_core.tools import StructuredTool
    return StructuredTool.from_function(lambda **kw: {}, name=name,
                                        description=f"fake {name}")


def test_c4_static_matrix_when_no_footprint():
    """Without a footprint, ToolSetBuilder falls back to the static matrix."""
    from chimera_rag.plugins.colla_rag.agent_factory import ToolSetBuilder
    all_tools = [_fake_tool(n) for n in
                 ("graph_path_search", "graph_neighbors", "hybrid_search",
                  "web_search", "vector_search")]
    sel = ToolSetBuilder.for_agent("multi_hop", all_tools, footprint=None)
    names = {t.name for t in sel}
    assert "graph_path_search" in names
    assert "hybrid_search" in names
    # web_search is NOT in the static multi_hop matrix
    assert "web_search" not in names


def test_c4_graph_poor_multihop_drops_traversal_adds_web():
    """A graph-poor footprint withdraws traversal tools and adds web fallback."""
    from chimera_rag.plugins.colla_rag.agent_factory import GraphFootprint, ToolSetBuilder
    all_tools = [_fake_tool(n) for n in
                 ("graph_path_search", "graph_subgraph_extract", "graph_neighbors",
                  "graph_community_search", "graph_statistics",
                  "hybrid_search", "vector_search", "web_search", "web_fetch")]
    fp = GraphFootprint(entity_hit_rate=0.0, has_path=False, density="sparse", n_entities=2)
    assert fp.graph_poor is True
    sel = ToolSetBuilder.for_agent("multi_hop", all_tools, footprint=fp)
    names = {t.name for t in sel}
    # traversal tools withdrawn
    assert "graph_path_search" not in names
    assert "graph_subgraph_extract" not in names
    assert "graph_neighbors" not in names
    # web fallback added (even though not in static multi_hop matrix)
    assert "web_search" in names
    assert "web_fetch" in names
    # dense retrieval retained
    assert "hybrid_search" in names


def test_c4_dense_multihop_keeps_traversal():
    """A dense footprint with a path keeps the traversal tools."""
    from chimera_rag.plugins.colla_rag.agent_factory import GraphFootprint, ToolSetBuilder
    all_tools = [_fake_tool(n) for n in
                 ("graph_path_search", "graph_neighbors", "hybrid_search", "web_search")]
    fp = GraphFootprint(entity_hit_rate=1.0, has_path=True, density="dense", n_entities=2)
    assert fp.graph_poor is False
    sel = ToolSetBuilder.for_agent("multi_hop", all_tools, footprint=fp)
    names = {t.name for t in sel}
    assert "graph_path_search" in names
    assert "web_search" not in names  # not added in dense setting


def test_c4_single_hop_drops_graph_neighbors_when_entity_absent():
    from chimera_rag.plugins.colla_rag.agent_factory import GraphFootprint, ToolSetBuilder
    all_tools = [_fake_tool(n) for n in ("graph_neighbors", "hybrid_search", "vector_search")]
    fp = GraphFootprint(entity_hit_rate=0.0, has_path=False, density="sparse", n_entities=1)
    sel = ToolSetBuilder.for_agent("single_hop", all_tools, footprint=fp)
    names = {t.name for t in sel}
    assert "graph_neighbors" not in names
    assert "hybrid_search" in names


def test_c4_graph_views_formalized():
    from chimera_rag.plugins.colla_rag.agent_factory import GRAPH_VIEWS, ToolSetBuilder
    assert GRAPH_VIEWS["multi_hop"] == "path"
    assert GRAPH_VIEWS["summarization"] == "community"
    assert ToolSetBuilder.graph_view("single_hop") == "attribute"
