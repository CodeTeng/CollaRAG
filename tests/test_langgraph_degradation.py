"""Degradation tests for the optional ``agent`` extra.

When langgraph / langchain-core are present the slot registers normally.
When absent, importing the sub-package must NOT crash the rest of the
``intent_agent`` plugin, and :func:`require_langgraph` must raise a clear,
actionable error pointing at ``uv sync --extra agent``.
"""

from __future__ import annotations

import pytest


def test_require_langgraph_raises_clear_error_when_unavailable(monkeypatch):
    import chimera_rag.plugins.intent_agent.langgraph as lg

    monkeypatch.setattr(lg, "LANGGRAPH_AVAILABLE", False)
    with pytest.raises(ImportError) as excinfo:
        lg.require_langgraph()
    msg = str(excinfo.value)
    assert "agent" in msg
    assert "uv sync --extra agent" in msg


def test_require_langgraph_is_noop_when_available():
    import chimera_rag.plugins.intent_agent.langgraph as lg

    pytest.importorskip("langgraph")
    # In the dev environment the extra IS installed, so this must not raise.
    if lg.LANGGRAPH_AVAILABLE:
        lg.require_langgraph()  # no exception


def test_langgraph_slot_registered_when_extra_present():
    pytest.importorskip("langgraph")
    import chimera_rag.plugins.intent_agent  # noqa: F401 — fires registration
    from chimera_rag.core.registry import get_registry

    names = get_registry().list_slot("retriever")
    assert "intent_agent.langgraph" in names
