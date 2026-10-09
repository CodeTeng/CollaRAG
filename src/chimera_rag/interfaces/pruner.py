"""ABC for triple pruners."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Triple


class BasePruner(ABC):
    """Remove noisy / redundant triples before they enter the graph.

    The vanilla baseline uses a :class:`NoOpPruner`; AdaGraph contributes a
    DualRedundancyPruner that removes similarity-redundant and transitive
    triples.
    """

    @abstractmethod
    def prune(self, triples: list[Triple]) -> list[Triple]:
        """Return a filtered list of triples; may be a subset of input."""


__all__ = ["BasePruner"]
