"""AdaGraph's dual-redundancy pruner.

Two forms of redundancy are addressed:

* **similarity** — two triples have (s, p, o) strings whose pairwise
  Jaccard-over-tokens similarity meets ``similarity_threshold``. The
  earlier triple wins (stable, deterministic).
* **transitive** — closure inference (A→B ∧ B→C ⇒ A→C) removes triples
  derivable from others. Reserved for a later cycle; enabling the flag
  is a no-op until then.

Using token Jaccard instead of embedding cosine keeps the pruner free of
optional dependencies; embedding-based deduplication is an easy future
swap-in behind the same class surface.
"""

from __future__ import annotations

import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Triple
from chimera_rag.interfaces.pruner import BasePruner

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")


def _tokenise(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _triple_signature(t: Triple) -> set[str]:
    return _tokenise(f"{t.subject} {t.predicate} {t.object}")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


@register("pruner", "adagraph.dual_redundancy")
class DualRedundancyPruner(BasePruner):
    """Similarity + (reserved) transitive redundancy removal."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        enable_transitive: bool = False,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self.similarity_threshold = similarity_threshold
        self.enable_transitive = enable_transitive

    # ------------------------------------------------------------------
    def prune(self, triples: list[Triple]) -> list[Triple]:
        kept: list[Triple] = []
        kept_sigs: list[set[str]] = []
        for t in triples:
            sig = _triple_signature(t)
            # similarity check against already-kept triples
            if any(_jaccard(sig, s) >= self.similarity_threshold for s in kept_sigs):
                continue
            kept.append(t)
            kept_sigs.append(sig)

        logger.info("dual_pruner: %d -> %d triples after similarity pass", len(triples), len(kept))

        if self.enable_transitive:
            kept = self._prune_transitive(kept)

        return kept

    # ------------------------------------------------------------------
    def _prune_transitive(self, triples: list[Triple]) -> list[Triple]:
        """Reserved: transitive-closure derivable triples removal.

        Requires a predicate-transitivity map (e.g. which predicates are
        actually transitive such as 'ancestor_of', 'part_of') plus BFS on
        the graph. Will land in a follow-up TDD cycle.
        """
        logger.debug("transitive pruning reserved — returning input unchanged")
        return triples


__all__ = ["DualRedundancyPruner"]
