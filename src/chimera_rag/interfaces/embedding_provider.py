"""ABC for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingProvider(ABC):
    """Turn a batch of texts into a 2-D float array ``(N, dim)``."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the produced vectors."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` array of embeddings."""


__all__ = ["BaseEmbeddingProvider"]
