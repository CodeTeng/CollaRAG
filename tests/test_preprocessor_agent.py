"""Tests for PreprocessorAgent."""
from langchain_core.tools import StructuredTool

from chimera_rag.core.types import Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.preprocessor import PreprocessorAgent


def _mock_rewrite_tool():
    return StructuredTool.from_function(
        lambda query, context="": f"rewritten: {query}",
        name="query_rewrite", description="mock rewrite",
    )


def test_preprocessor_rewrites_query():
    agent = PreprocessorAgent(
        agent_type="preprocessor",
        tools=[_mock_rewrite_tool()],
        config={},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="Where was he born?"), ctx)
    assert "rewritten" in answer.text
    assert ctx.trace["original_query"] == "Where was he born?"
