"""MemoryManager — unified facade for dual-layer memory."""
from __future__ import annotations

from pathlib import Path

from chimera_rag.plugins.colla_rag.memory.agent_memory import AgentPrivateMemory
from chimera_rag.plugins.colla_rag.memory.shared_memory import SharedMemory


class SessionMemory:
    """Simple in-memory session history (same interface as BaseMemory)."""

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}

    def append(self, session_id: str, turn: str) -> None:
        self._store.setdefault(session_id, []).append(turn)

    def get_history(self, session_id: str) -> list[str]:
        return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class MemoryManager:
    """Dual-layer memory facade: shared + agent-private + session."""

    def __init__(self, base_path: str, max_per_bucket: int = 200) -> None:
        root = Path(base_path)
        self.shared = SharedMemory(path=str(root / "shared"))
        self.agent_private = AgentPrivateMemory(path=str(root), max_per_bucket=max_per_bucket)
        self.session = SessionMemory()


__all__ = ["MemoryManager", "SessionMemory"]
