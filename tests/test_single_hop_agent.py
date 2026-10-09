"""Tests for SingleHopAgent."""
from langchain_core.tools import StructuredTool

from chimera_rag.core.types import Query
from chimera_rag.interfaces.base_agent import AgentContext
from chimera_rag.plugins.colla_rag.agents.single_hop import SingleHopAgent


def _mock_hybrid():
    return StructuredTool.from_function(
        lambda query, top_k=5, vector_weight=0.5: {"tool": "hybrid_search", "num_chunks": 2, "chunks": [
            {"chunk_id": "c1", "score": 0.9, "preview": "Einstein was born in Ulm"},
            {"chunk_id": "c2", "score": 0.7, "preview": "He won the Nobel Prize"},
        ]},
        name="hybrid_search", description="mock hybrid",
    )


def _mock_assess():
    return StructuredTool.from_function(
        lambda query: {"tool": "assess_evidence", "quality": 0.8, "relevance": 0.9,
                       "completeness": 0.7, "num_evidence_chunks": 2, "suggestion": None},
        name="assess_evidence", description="mock assess",
    )


def test_single_hop_produces_answer():
    class FakeLLM:
        async def complete(self, prompt, **kw):
            return "Einstein was born in Ulm, Germany."

    agent = SingleHopAgent(
        agent_type="single_hop",
        tools=[_mock_hybrid(), _mock_assess()],
        config={"llm": FakeLLM(), "top_k": 5},
    )
    ctx = AgentContext(session_id=None)
    answer = agent.execute(Query(text="Where was Einstein born?"), ctx)
    assert len(answer.text) > 0
    assert answer.trace.get("agent_type") == "single_hop"
