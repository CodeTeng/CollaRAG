"""Tests for SummarizationAgent."""
from langchain_core.tools import StructuredTool

from chimera_rag.core.types import Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.summarization import SummarizationAgent


def _mock_hybrid():
    return StructuredTool.from_function(
        lambda query, top_k=5, vector_weight=0.5: {"tool": "hybrid_search", "num_chunks": 2, "chunks": [
            {"chunk_id": "c1", "score": 0.9, "preview": "Point A about topic"},
            {"chunk_id": "c2", "score": 0.7, "preview": "Point B about topic"},
        ]},
        name="hybrid_search", description="mock",
    )


def test_summarization_produces_summary():
    class FakeLLM:
        async def complete(self, prompt, **kw):
            if "extract" in prompt.lower() or "key information" in prompt.lower():
                return "Key point from this chunk."
            return "Summary: The topic has two main points."

    agent = SummarizationAgent(
        agent_type="summarization",
        tools=[_mock_hybrid()],
        config={"llm": FakeLLM(), "retrieve_top_k": 10},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="Summarize the topic"), ctx)
    assert len(answer.text) > 0
    assert answer.strategy_name == "map_reduce"
