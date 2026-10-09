"""Tests for preprocess tools (query_rewrite, coreference_resolve, followup_merge)."""
from chimera_rag.plugins.colla_rag.tools.preprocess_tools import build_preprocess_tools


class FakeLLM:
    async def complete(self, prompt: str, **kw) -> str:
        if "rewrite" in prompt.lower() or "改写" in prompt.lower():
            return "What is Albert Einstein's birthplace?"
        if "coreference" in prompt.lower() or "指代" in prompt.lower():
            return "What is Einstein's birthplace?"
        return prompt


class FakeSessionMemory:
    def __init__(self):
        self._store: dict[str, list[str]] = {}

    def get_history(self, sid: str) -> list[str]:
        return self._store.get(sid, [])

    def append(self, sid: str, turn: str):
        self._store.setdefault(sid, []).append(turn)


def test_build_preprocess_tools_count():
    tools = build_preprocess_tools(llm=FakeLLM(), session_memory=FakeSessionMemory())
    names = {t.name for t in tools}
    assert "query_rewrite" in names
    assert "coreference_resolve" in names
    assert "followup_merge" in names
    assert len(tools) == 3


def test_query_rewrite_invocation():
    tools = build_preprocess_tools(llm=FakeLLM(), session_memory=FakeSessionMemory())
    rw = next(t for t in tools if t.name == "query_rewrite")
    result = rw.invoke({"query": "Where was he born?", "context": "Discussing Einstein"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_followup_merge_invocation():
    tools = build_preprocess_tools(llm=FakeLLM(), session_memory=FakeSessionMemory())
    fm = next(t for t in tools if t.name == "followup_merge")
    result = fm.invoke({
        "query": "What about his awards?",
        "prev_query": "Tell me about Einstein",
        "prev_answer": "Einstein was a physicist",
    })
    assert isinstance(result, str)
