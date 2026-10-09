"""Tests for memory tools."""
from chimera_rag.plugins.colla_rag.tools.memory_tools import build_memory_tools


class FakeSharedMemory:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def read(self, memory_type: str, key: str):
        return self._store.get(memory_type, {}).get(key)

    def write(self, memory_type: str, key: str, value):
        self._store.setdefault(memory_type, {})[key] = value


class FakeAgentMemory:
    def __init__(self):
        self._store: dict[str, list] = {}

    def recall(self, agent_type: str, intent: str, top_n: int = 3):
        return self._store.get(f"{agent_type}:{intent}", [])[:top_n]

    def remember(self, agent_type: str, entry: dict):
        key = f"{agent_type}:{entry.get('intent', 'unknown')}"
        self._store.setdefault(key, []).append(entry)


class FakeSessionMemory:
    def __init__(self):
        self._h: dict[str, list[str]] = {}

    def get_history(self, sid):
        return self._h.get(sid, [])

    def append(self, sid, turn):
        self._h.setdefault(sid, []).append(turn)


def test_build_memory_tools_count():
    tools = build_memory_tools(
        shared_memory=FakeSharedMemory(),
        agent_memory=FakeAgentMemory(),
        session_memory=FakeSessionMemory(),
    )
    names = {t.name for t in tools}
    assert "read_session_context" in names
    assert "write_session_context" in names
    assert "read_shared_memory" in names
    assert "write_shared_memory" in names
    assert "read_agent_memory" in names
    assert "write_agent_memory" in names


def test_read_write_shared_memory():
    sm = FakeSharedMemory()
    tools = build_memory_tools(shared_memory=sm, agent_memory=FakeAgentMemory(),
                               session_memory=FakeSessionMemory())
    write_tool = next(t for t in tools if t.name == "write_shared_memory")
    read_tool = next(t for t in tools if t.name == "read_shared_memory")
    write_tool.invoke({"memory_type": "qa_cache", "key": "q1", "value": "answer1"})
    result = read_tool.invoke({"memory_type": "qa_cache", "key": "q1"})
    assert result == "answer1"


def test_read_write_session_context():
    sess = FakeSessionMemory()
    tools = build_memory_tools(shared_memory=FakeSharedMemory(), agent_memory=FakeAgentMemory(),
                               session_memory=sess)
    write_tool = next(t for t in tools if t.name == "write_session_context")
    read_tool = next(t for t in tools if t.name == "read_session_context")
    write_tool.invoke({"session_id": "s1", "query": "hi", "answer": "hello"})
    result = read_tool.invoke({"session_id": "s1", "last_n": 5})
    assert len(result) == 1
