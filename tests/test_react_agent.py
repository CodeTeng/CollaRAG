"""Tests for ReActAgent."""
from chimera_rag.core.types import Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.intent_agent.agents.react_agent import ReActAgent
from langchain_core.tools import StructuredTool


def test_react_agent_creates():
    class FakeLLM:
        async def complete(self, prompt, **kw):
            return "Final Answer: The answer is 42."

    agent = ReActAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_iterations": 2, "quality_threshold": 0.6},
    )
    assert agent.agent_type == "multi_hop"


def test_react_agent_executes():
    class FakeLLM:
        async def complete(self, prompt, **kw):
            return "Final Answer: The answer is 42."

    agent = ReActAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="What is 6*7?"), ctx)
    assert "42" in answer.text
