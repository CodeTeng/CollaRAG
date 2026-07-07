"""Tests for WebSearchAgent."""
from chimera_rag.core.types import Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.intent_agent.agents.web_search import WebSearchAgent
from langchain_core.tools import StructuredTool


def _mock_web_search():
    return StructuredTool.from_function(
        lambda query, max_results=5: [{"title": "Result", "href": "http://example.com", "body": "Some info"}],
        name="web_search", description="mock",
    )


def test_web_search_agent():
    class FakeLLM:
        async def complete(self, prompt, **kw):
            return "Based on web results, the answer is X."

    agent = WebSearchAgent(
        agent_type="other",
        tools=[_mock_web_search()],
        config={"llm": FakeLLM()},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="What is the weather in Tokyo?"), ctx)
    assert len(answer.text) > 0
    assert answer.strategy_name == "web_search"
