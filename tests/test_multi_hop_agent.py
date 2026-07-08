"""Tests for MultiHopAgent (Plan-Execute-Reflect)."""

from chimera_rag.core.types import Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.multi_hop import MultiHopAgent


def test_multi_hop_agent_creates():
    class FakeLLM:
        async def complete(self, prompt, **kw):
            return "Step Result: The answer is 42."

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_plan_steps": 3, "quality_threshold": 0.6},
    )
    assert agent.agent_type == "multi_hop"


def test_multi_hop_agent_executes():
    call_count = 0

    class FakeLLM:
        async def complete(self, prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '[{"step": "Find the answer", "tools_hint": []}]'
            if "Step Result" in prompt or "step" in prompt.lower():
                return "Step Result: The answer is 42."
            return "The answer is 42."

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="What is 6*7?"), ctx)
    assert "42" in answer.text
    assert answer.strategy_name == "plan_execute"
    assert "plan" in ctx.trace


def test_multi_hop_plan_fallback():
    """When LLM returns unparseable plan, should fall back to single-step."""

    class FakeLLM:
        async def complete(self, prompt, **kw):
            if "planning" in prompt.lower() or "plan" in prompt.lower():
                return "I cannot generate a plan."
            return "Step Result: Fallback answer."

    agent = MultiHopAgent(
        agent_type="multi_hop",
        tools=[],
        config={"llm": FakeLLM(), "max_step_iterations": 1, "quality_threshold": 0.6},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="complex query"), ctx)
    assert len(answer.text) > 0
    assert len(ctx.trace["plan"]) == 1
