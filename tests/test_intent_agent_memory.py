"""Tests for :mod:`chimera_rag.plugins.intent_agent.memory`."""

from __future__ import annotations


def test_session_memory_implements_base_memory_abc():
    from chimera_rag.interfaces.memory import BaseMemory
    from chimera_rag.plugins.intent_agent.memory import SessionMemory

    assert isinstance(SessionMemory(), BaseMemory)


def test_session_memory_appends_and_retrieves_in_order():
    from chimera_rag.plugins.intent_agent.memory import SessionMemory

    m = SessionMemory()
    m.append("s1", "hello")
    m.append("s1", "world")
    assert m.get_history("s1") == ["hello", "world"]


def test_session_memory_sessions_are_isolated():
    from chimera_rag.plugins.intent_agent.memory import SessionMemory

    m = SessionMemory()
    m.append("a", "x")
    m.append("b", "y")
    assert m.get_history("a") == ["x"]
    assert m.get_history("b") == ["y"]


def test_session_memory_clear_resets_only_that_session():
    from chimera_rag.plugins.intent_agent.memory import SessionMemory

    m = SessionMemory()
    m.append("a", "x")
    m.append("b", "y")
    m.clear("a")
    assert m.get_history("a") == []
    assert m.get_history("b") == ["y"]


def test_session_memory_respects_max_history():
    """Ring-buffer style: oldest entries drop once max_history is exceeded."""
    from chimera_rag.plugins.intent_agent.memory import SessionMemory

    m = SessionMemory(max_history=3)
    for i in range(5):
        m.append("s", f"turn{i}")
    hist = m.get_history("s")
    assert len(hist) == 3
    # Oldest kept should be turn2 (0 and 1 evicted).
    assert hist == ["turn2", "turn3", "turn4"]


def test_session_memory_registers_under_intent_agent_slot():
    import chimera_rag.plugins.intent_agent  # noqa: F401

    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.intent_agent.memory import SessionMemory

    assert get_registry().get("memory", "intent_agent.session") is SessionMemory
