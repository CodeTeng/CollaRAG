"""Tests for AgentPrivateMemory."""
import tempfile

from chimera_rag.plugins.colla_rag.memory.agent_memory import AgentPrivateMemory


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


def test_recall_surfaces_lessons_after_successes():
    """recall() returns successes first, then lessons, so failures resurface."""
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        am.remember("multi_hop", {
            "intent": "multi_hop", "query_text": "Q1", "tool_sequence": ["hybrid_search"],
            "final_quality": 0.9, "outcome": "success",
        })
        am.remember("multi_hop", {
            "intent": "multi_hop", "query_text": "Q2", "tool_sequence": ["vector_search"],
            "final_quality": 0.2, "outcome": "lesson",
            "lesson": "vector_search alone insufficient for multi-hop",
        })
        recalled = am.recall("multi_hop", "multi_hop", top_n=3)
        # Success first, lesson appended — both surface.
        assert len(recalled) == 2
        assert recalled[0]["outcome"] == "success"
        assert recalled[1]["outcome"] == "lesson"
        assert recalled[1]["lesson"].startswith("vector_search alone")


def test_recall_lessons_only_and_recurrence_ranked():
    """recall_lessons() returns only failures, ranked by hit count."""
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        am.remember("multi_hop", {
            "intent": "multi_hop", "query_text": "Q1", "tool_sequence": ["a"],
            "final_quality": 0.9, "outcome": "success",
        })
        # Two different failing tool sequences.
        am.remember("multi_hop", {
            "intent": "multi_hop", "query_text": "Q2", "tool_sequence": ["b"],
            "final_quality": 0.1, "outcome": "lesson", "lesson": "fail-b",
        })
        am.remember("multi_hop", {
            "intent": "multi_hop", "query_text": "Q3", "tool_sequence": ["c"],
            "final_quality": 0.1, "outcome": "lesson", "lesson": "fail-c",
        })
        # Re-fail on the same tool_sequence as "b" → hit_count grows.
        am.remember("multi_hop", {
            "intent": "multi_hop", "query_text": "Q4", "tool_sequence": ["b"],
            "final_quality": 0.1, "outcome": "lesson", "lesson": "fail-b-updated",
        })
        lessons = am.recall_lessons("multi_hop", "multi_hop", top_n=3)
        assert len(lessons) == 2
        # The recurring failure ("b", hit_count=1) ranks ahead of the one-off ("c", 0).
        assert lessons[0]["tool_sequence"] == ["b"]
        assert lessons[0]["hit_count"] == 1
        # Dedup refreshed the lesson text with the latest note.
        assert lessons[0]["lesson"] == "fail-b-updated"


def test_dedup_merges_lesson_text():
    with tempfile.TemporaryDirectory() as d:
        am = AgentPrivateMemory(path=d)
        am.remember("multi_hop", {
            "intent": "multi_hop", "query_text": "Q", "tool_sequence": ["x"],
            "final_quality": 0.2, "outcome": "lesson", "lesson": "old note",
        })
        am.remember("multi_hop", {
            "intent": "multi_hop", "query_text": "Q", "tool_sequence": ["x"],
            "final_quality": 0.2, "outcome": "lesson", "lesson": "new note",
        })
        recalled = am.recall_lessons("multi_hop", "multi_hop")
        assert len(recalled) == 1
        assert recalled[0]["lesson"] == "new note"
        assert recalled[0]["hit_count"] == 1
