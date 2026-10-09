"""Tests for MemoryManager (dual-layer facade)."""
import tempfile

from chimera_rag.plugins.colla_rag.memory.memory_manager import MemoryManager


def test_memory_manager_creation():
    with tempfile.TemporaryDirectory() as d:
        mm = MemoryManager(base_path=d)
        assert mm.shared is not None
        assert mm.agent_private is not None
        assert mm.session is not None


def test_memory_manager_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        mm = MemoryManager(base_path=d)
        mm.shared.write("qa_cache", "q1", {"answer": "test"})
        assert mm.shared.read("qa_cache", "q1")["answer"] == "test"

        mm.agent_private.remember("react", {
            "intent": "multi_hop", "query_text": "Q",
            "tool_sequence": ["a"], "final_quality": 0.9, "outcome": "success",
        })
        assert len(mm.agent_private.recall("react", "multi_hop")) == 1
