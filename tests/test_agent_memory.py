"""Tests for AgentPrivateMemory."""
import tempfile

from chimera_rag.plugins.intent_agent.memory.agent_memory import AgentPrivateMemory


def test_remember_and_recall():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        entry = {
            "intent": "multi_hop",
            "query_text": "What is X?",
            "tool_sequence": ["hybrid_search", "graph_neighbors"],
            "final_quality": 0.8,
            "outcome": "success",
        }
        am.remember("react", entry)
        recalled = am.recall("react", "multi_hop", top_n=3)
        assert len(recalled) == 1
        assert recalled[0]["tool_sequence"] == ["hybrid_search", "graph_neighbors"]


def test_recall_empty():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        assert am.recall("react", "single_hop") == []


def test_dedup_on_same_tool_sequence():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        entry = {"intent": "multi_hop", "query_text": "Q", "tool_sequence": ["a", "b"],
                 "final_quality": 0.5, "outcome": "success"}
        am.remember("react", entry)
        entry2 = {**entry, "final_quality": 0.9}
        am.remember("react", entry2)
        recalled = am.recall("react", "multi_hop")
        assert len(recalled) == 1
        assert recalled[0]["final_quality"] == 0.9


def test_cross_agent_isolation():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        am.remember("react", {"intent": "multi_hop", "query_text": "Q", "tool_sequence": ["a"],
                               "final_quality": 0.8, "outcome": "success"})
        am.remember("native_rag", {"intent": "single_hop", "query_text": "Q2", "tool_sequence": ["b"],
                                    "final_quality": 0.7, "outcome": "success"})
        assert len(am.recall("react", "multi_hop")) == 1
        assert len(am.recall("native_rag", "multi_hop")) == 0
        assert len(am.recall("native_rag", "single_hop")) == 1


def test_stats():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        am.remember("react", {"intent": "multi_hop", "query_text": "Q", "tool_sequence": [],
                               "final_quality": 0.8, "outcome": "success"})
        s = am.stats()
        assert s["total"] >= 1
