"""ABC + data model for experiential (cross-session) memory.

This is innovation #2c's contract. It is **separate** from
:class:`chimera_rag.interfaces.memory.BaseMemory` (the per-session
conversation memory) on purpose: experiential memory has a different
responsibility (distil and reuse *retrieval experience* across sessions),
so extending it as an independent ABC keeps ``SessionMemory`` untouched and
preserves the project's "additive, not destructive" rule.

An :class:`ExperienceEntry` captures three kinds of experience requested by
the design (``docs/innovation_2c_experiential_memory.md`` §4):

* **successful tool sequence** — ``tool_sequence`` + ``iterations``
* **reflection lesson**        — ``outcome="lesson"`` + ``lesson``
* **high-quality QA pair**     — ``answer_text`` + ``evidence_chunk_ids``

All fields are sourced from data innovation #2b already produces (the
LangGraph ``final_state`` and the ``ToolContext`` merged result), so
recording requires no new collection logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from chimera_rag.core.types import IntentLabel

ExperienceOutcome = Literal["success", "lesson"]


class ExperienceEntry(BaseModel):
    """One distilled retrieval experience, bucketed by intent."""

    # —— index fields ——
    intent: IntentLabel
    query_text: str

    # —— type 1: successful tool sequence ——
    tool_sequence: list[str] = Field(default_factory=list)
    iterations: int = Field(default=0, ge=0)

    # —— type 2: reflection lesson ——
    final_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome: ExperienceOutcome = "success"
    lesson: str | None = None

    # —— type 3: high-quality QA pair ——
    answer_text: str | None = None
    evidence_chunk_ids: list[str] = Field(default_factory=list)

    # —— meta ——
    created_at: str = ""
    hit_count: int = Field(default=0, ge=0)


class BaseExperientialMemory(ABC):
    """Cross-session, persistent store of distilled retrieval experience.

    Concrete implementations persist entries (e.g. to disk) so that a later
    query — even in a different session — can recall how similar questions
    were successfully (or unsuccessfully) retrieved before.
    """

    @abstractmethod
    def recall(
        self, intent: str, query_text: str, top_n: int = 3
    ) -> list[ExperienceEntry]:
        """Return up to ``top_n`` past experiences for the given intent bucket."""
        ...

    @abstractmethod
    def remember(self, entry: ExperienceEntry) -> None:
        """Persist (or merge) a new experience entry."""
        ...

    @abstractmethod
    def stats(self) -> dict:
        """Return a lightweight summary (per-bucket counts, totals)."""
        ...


__all__ = [
    "BaseExperientialMemory",
    "ExperienceEntry",
    "ExperienceOutcome",
]
