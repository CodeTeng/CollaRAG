"""Tests for BaseAgent ABC and AgentContext."""
import pytest

from chimera_rag.core.types import Answer, Query
from chimera_rag.interfaces.base_agent import AgentContext, BaseAgent


class DummyAgent(BaseAgent):
    """Minimal concrete agent for testing the template method."""

    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        return Answer(text=f"answer for: {query.text}")


def test_agent_context_creation():
    ctx = AgentContext(session_id="s1")
    assert ctx.session_id == "s1"
    assert ctx.trace == {}


def test_base_agent_template_method():
    agent = DummyAgent(agent_type="test", tools=[], config={})
    ctx = AgentContext(session_id="s1")
    answer = agent.execute(Query(text="hello"), ctx)
    assert answer.text == "answer for: hello"


def test_base_agent_trace_populated():
    agent = DummyAgent(agent_type="test", tools=[], config={})
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="test"), ctx)
    assert "agent_type" in answer.trace
    assert answer.trace["agent_type"] == "test"


def test_base_agent_is_abstract():
    with pytest.raises(TypeError):
        BaseAgent(agent_type="x", tools=[], config={})  # type: ignore
