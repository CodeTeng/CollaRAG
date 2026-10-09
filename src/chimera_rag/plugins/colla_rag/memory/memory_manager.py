"""MemoryManager — unified facade for dual-layer memory.

The dual-layer memory is graph-grounded (Contribution 3, graph-grounded):
the shared layer adds an :class:`EvidenceSubgraphStore` retrieved by
entity-neighborhood overlap (mechanism 3B), and the private layer adds a
:class:`PlanTemplateStore` (3A) and a :class:`GapLessonStore` (3C) that feed
G-PER's Plan and Reflect phases. All three degrade to no-ops when no graph
evidence is available, so the legacy text-hash cache and text lessons remain
as fallbacks.
"""
from __future__ import annotations

from pathlib import Path

from chimera_rag.plugins.colla_rag.memory.agent_memory import AgentPrivateMemory
from chimera_rag.plugins.colla_rag.memory.graph_memory import (
    EvidenceSubgraphStore,
    GapLessonStore,
    PlanTemplateStore,
)
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
    """Dual-layer memory facade: shared + agent-private + session.

    Graph-grounded extensions (Contribution 3):
      * ``shared_evidence``  — evidence-subgraph overlap cache (shared, 3B)
      * ``plan_templates``   — per-compositionality graph plan templates (private, 3A)
      * ``gap_lessons``      — typed structural-gap lessons (private, 3C)
    """

    def __init__(self, base_path: str, max_per_bucket: int = 200) -> None:
        root = Path(base_path)
        self.shared = SharedMemory(path=str(root / "shared"))
        self.agent_private = AgentPrivateMemory(path=str(root), max_per_bucket=max_per_bucket)
        self.session = SessionMemory()
        # Graph-grounded layers.
        self.shared_evidence = EvidenceSubgraphStore(
            path=str(root / "shared" / "evidence_subgraphs.json"),
        )
        self.plan_templates = PlanTemplateStore(
            path=str(root / "private" / "plan_templates.json"),
        )
        self.gap_lessons = GapLessonStore(
            path=str(root / "private" / "gap_lessons.json"),
            max_entries=max_per_bucket,
        )

    def persist(self) -> None:
        """Persist every graph-grounded layer (shared JSON persisted separately)."""
        self.shared_evidence.persist()
        self.plan_templates.persist()
        self.gap_lessons.persist()


__all__ = ["MemoryManager", "SessionMemory"]
