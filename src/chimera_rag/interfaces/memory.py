"""ABC for session memory."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Per-session, append-only conversation memory (used by IntentAgent).

    Concrete implementations persist history so ``follow_up`` intents can
    resolve references like "he" or "that event".
    """

    @abstractmethod
    def append(self, session_id: str, turn: str) -> None: ...

    @abstractmethod
    def get_history(self, session_id: str) -> list[str]: ...

    @abstractmethod
    def clear(self, session_id: str) -> None: ...


__all__ = ["BaseMemory"]
