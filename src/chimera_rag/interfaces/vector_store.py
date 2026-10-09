"""ABC for vector storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseVectorStore(ABC):
    """Dense vector index used for chunk / triple similarity search.

    MVP backend: FAISS ``IndexFlatIP`` + pickle-persisted id list.
    Returns search hits as ``list[tuple[id, score]]``.
    """

    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        """Append ``len(ids)`` vectors to the index. ``ids`` are caller-owned."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k hits as ``(id, score)`` ordered by descending score."""

    @abstractmethod
    def persist(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...


__all__ = ["BaseVectorStore"]
