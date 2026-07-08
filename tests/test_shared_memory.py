"""Tests for SharedMemory."""
import tempfile

from chimera_rag.plugins.colla_rag.memory.shared_memory import SharedMemory


def test_shared_memory_read_write():
    with tempfile.TemporaryDirectory() as d:
        sm = SharedMemory(path=d)
        sm.write("qa_cache", "q1_hash", {"answer": "42", "quality": 0.9})
        result = sm.read("qa_cache", "q1_hash")
        assert result["answer"] == "42"


def test_shared_memory_read_missing():
    with tempfile.TemporaryDirectory() as d:
        sm = SharedMemory(path=d)
        assert sm.read("qa_cache", "nonexistent") is None


def test_shared_memory_routing_stats():
    with tempfile.TemporaryDirectory() as d:
        sm = SharedMemory(path=d)
        sm.write("routing_stats", "q1", {"intent": "multi_hop", "agent": "react", "quality": 0.8})
        result = sm.read("routing_stats", "q1")
        assert result["intent"] == "multi_hop"


def test_shared_memory_persistence():
    with tempfile.TemporaryDirectory() as d:
        sm1 = SharedMemory(path=d)
        sm1.write("entity_aliases", "Einstein", "Albert Einstein")
        sm1.persist()
        sm2 = SharedMemory(path=d)
        sm2.load()
        assert sm2.read("entity_aliases", "Einstein") == "Albert Einstein"


def test_shared_memory_stats():
    with tempfile.TemporaryDirectory() as d:
        sm = SharedMemory(path=d)
        sm.write("qa_cache", "q1", {"answer": "x"})
        sm.write("qa_cache", "q2", {"answer": "y"})
        s = sm.stats()
        assert s["qa_cache"] == 2
