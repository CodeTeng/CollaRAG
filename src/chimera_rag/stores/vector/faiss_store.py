"""FAISS-backed vector store with an external string-id map.

FAISS itself only knows integer row ids; we keep a parallel ``list[str]``
whose index i corresponds to FAISS row i. Both the ``.faiss`` index and
the ``.ids.pkl`` sidecar are persisted together.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from chimera_rag.interfaces.vector_store import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    """IndexFlat-based vector store (inner product or L2)."""

    def __init__(self, dim: int, index_type: str = "flat_ip") -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if index_type not in {"flat_ip", "flat_l2"}:
            raise ValueError(f"unsupported index_type: {index_type!r}")
        self._dim = dim
        self._index_type = index_type
        self._index = self._new_index()
        self._ids: list[str] = []

    # ------------------------------------------------------------------
    def _new_index(self) -> faiss.Index:
        return (
            faiss.IndexFlatIP(self._dim)
            if self._index_type == "flat_ip"
            else faiss.IndexFlatL2(self._dim)
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"len(ids)={len(ids)} does not match vectors rows={vectors.shape[0]}"
            )
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != index dim {self._dim}"
            )
        vecs = np.ascontiguousarray(vectors.astype(np.float32))
        self._index.add(vecs)
        self._ids.extend(ids)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if len(self._ids) == 0:
            return []
        q = np.ascontiguousarray(query_vector.astype(np.float32)).reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self._dim}")
        k = min(top_k, len(self._ids))
        distances, indices = self._index.search(q, k)

        # For IP: larger is better; for L2: smaller is better — we invert
        # L2 by negating so the caller-facing contract ("descending score")
        # holds uniformly.
        hits: list[tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:  # faiss uses -1 for "not enough neighbours"
                continue
            score = float(dist) if self._index_type == "flat_ip" else float(-dist)
            hits.append((self._ids[idx], score))
        return hits

    # ------------------------------------------------------------------
    # Persistence — writes two files: <path>.faiss and <path>.ids.pkl
    # ------------------------------------------------------------------
    def persist(self, path: str) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(base) + ".faiss")
        with open(str(base) + ".ids.pkl", "wb") as f:
            pickle.dump({"ids": self._ids, "dim": self._dim, "type": self._index_type}, f)

    def load(self, path: str) -> None:
        base = str(path)
        self._index = faiss.read_index(base + ".faiss")
        with open(base + ".ids.pkl", "rb") as f:
            meta = pickle.load(f)
        self._ids = list(meta["ids"])
        self._dim = int(meta["dim"])
        self._index_type = str(meta["type"])


__all__ = ["FAISSVectorStore"]
