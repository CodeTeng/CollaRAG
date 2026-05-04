"""Stateless memory — vanilla baseline."""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.interfaces.memory import BaseMemory


@register("memory", "defaults.none")
class NoMemory(BaseMemory):
    """Discards everything; get_history always returns []."""

    def append(self, session_id: str, turn: str) -> None:
        return None

    def get_history(self, session_id: str) -> list[str]:
        return []

    def clear(self, session_id: str) -> None:
        return None


__all__ = ["NoMemory"]
