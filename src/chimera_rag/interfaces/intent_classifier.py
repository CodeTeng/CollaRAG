"""ABC for intent classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from chimera_rag.core.types import Intent, Query


class BaseIntentClassifier(ABC):
    """Map a :class:`Query` to an :class:`Intent`.

    Implementations include:

    * ``defaults.rule`` — always returns ``factual`` (vanilla baseline)
    * ``colla_rag.llm`` — LLM-powered 6-way classifier with rule fallback
    """

    @abstractmethod
    def classify(self, query: Query) -> Intent:
        """Return the classified intent."""


__all__ = ["BaseIntentClassifier"]
