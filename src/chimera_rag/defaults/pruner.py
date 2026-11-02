"""Pruner implementations shipped with the core framework.

``defaults.noop`` is the vanilla baseline used by every ablation config;
``defaults.jaccard`` is a generic near-duplicate filter offered as a
plain option rather than a research contribution.
"""

from __future__ import annotations

import logging
import re

from chimera_rag.core.registry import register
from chimera_rag.core.types import Triple
from chimera_rag.interfaces.pruner import BasePruner

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")


@register("pruner", "defaults.noop")
class NoOpPruner(BasePruner):
    """Return input list unchanged."""

    def prune(self, triples: list[Triple]) -> list[Triple]:
        return triples


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


@register("pruner", "defaults.jaccard")
class JaccardPruner(BasePruner):
    """Drop triples whose (s, p, o) token Jaccard similarity to an
    already-kept triple reaches ``similarity_threshold``.

    First occurrence wins, so output order is stable and deterministic.
    Token Jaccard keeps the pruner free of optional dependencies.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self.similarity_threshold = similarity_threshold

    def prune(self, triples: list[Triple]) -> list[Triple]:
        kept: list[Triple] = []
        kept_sigs: list[set[str]] = []
        for t in triples:
            sig = _triple_signature(t)
            if any(_jaccard(sig, s) >= self.similarity_threshold for s in kept_sigs):
                continue
            kept.append(t)
            kept_sigs.append(sig)

        logger.info("jaccard_pruner: %d -> %d triples", len(triples), len(kept))
        return kept


__all__ = ["JaccardPruner", "NoOpPruner"]
