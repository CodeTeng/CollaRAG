"""NoOp pruner — vanilla baseline (keeps every triple)."""

from __future__ import annotations

from chimera_rag.core.registry import register
from chimera_rag.core.types import Triple
from chimera_rag.interfaces.pruner import BasePruner


@register("pruner", "defaults.noop")
class NoOpPruner(BasePruner):
    """Return input list unchanged."""

    def prune(self, triples: list[Triple]) -> list[Triple]:
        return triples


__all__ = ["NoOpPruner"]
