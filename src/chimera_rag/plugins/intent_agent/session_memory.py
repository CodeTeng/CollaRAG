"""In-memory session history for IntentAgent.

Each session is an append-only list of turn strings, capped by
``max_history`` to keep memory bounded. A ``max_history`` of ``None``
disables truncation (useful for tests).
"""

from __future__ import annotations

from collections import deque

from chimera_rag.core.registry import register
from chimera_rag.interfaces.memory import BaseMemory


@register("memory", "intent_agent.session")
class SessionMemory(BaseMemory):
    """Per-session deque-backed history store."""

    def __init__(self, max_history: int | None = 10) -> None:
        if max_history is not None and max_history <= 0:
            raise ValueError("max_history must be positive or None")
        self.max_history = max_history
        self._store: dict[str, deque[str]] = {}

    # ------------------------------------------------------------------
    def append(self, session_id: str, turn: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = deque(maxlen=self.max_history)
        self._store[session_id].append(turn)

    def get_history(self, session_id: str) -> list[str]:
        return list(self._store.get(session_id, deque()))

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


__all__ = ["SessionMemory"]
