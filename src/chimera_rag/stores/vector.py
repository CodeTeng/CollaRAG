"""FAISS-backed vector store — supports flat / HNSW / IVF indexes.

Design choices:

* A single class with an ``index_type`` discriminator keeps the
  external API stable (``add``, ``search``, ``persist``, ``load``) and
  hides FAISS specifics from callers.
* Scores exposed to callers are always "descending = better":

  - ``ip``  → inner product, FAISS returns higher-is-better directly
  - ``l2``  → euclidean, FAISS returns distances; we negate so the
              convention holds uniformly.

* HNSW defaults follow the FAISS authors' recommendation (M=32,
  efConstruction=200, efSearch=64) and give good recall without
  explicit training.
* IVF requires a *training* step on at least ``nlist`` points. Adding
  fewer rows raises a clear error rather than silently degrading.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from chimera_rag.core.exceptions import ChimeraError
from chimera_rag.interfaces.vector_store import BaseVectorStore

# FAISS metric constants, expressed as python ints so we don't depend on
# the faiss enum API across versions.
_METRIC_IP = faiss.METRIC_INNER_PRODUCT
_METRIC_L2 = faiss.METRIC_L2


def _resolve_metric(metric: str, index_type: str) -> int:
    """Map (metric, index_type) to a FAISS metric constant.

    For legacy ``flat_ip`` / ``flat_l2`` the metric is baked into the
    name; for newer types (hnsw / ivf_flat) the caller chooses via the
    separate ``metric`` field.
    """
    if index_type == "flat_ip":
        return _METRIC_IP
    if index_type == "flat_l2":
        return _METRIC_L2
    if metric == "ip":
        return _METRIC_IP
    if metric == "l2":
        return _METRIC_L2
    raise ValueError(f"unknown metric {metric!r} for index_type {index_type!r}")


class FAISSVectorStore(BaseVectorStore):
    """Pluggable FAISS store: flat_ip / flat_l2 / hnsw / ivf_flat."""

    SUPPORTED_INDEX_TYPES = ("flat_ip", "flat_l2", "hnsw", "ivf_flat")

    def __init__(
        self,
        dim: int,
        index_type: str = "hnsw",
        metric: str = "ip",
        hnsw: dict | None = None,
        ivf: dict | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"unsupported index_type: {index_type!r}; "
                f"expected one of {self.SUPPORTED_INDEX_TYPES}"
            )

        self._dim = dim
        self._index_type = index_type
        self._metric_str = metric
        self._faiss_metric = _resolve_metric(metric, index_type)
        # default params — real config overrides via constructor kwargs.
        self._hnsw_params = {"M": 32, "ef_construction": 200, "ef_search": 64}
        self._hnsw_params.update(hnsw or {})
        self._ivf_params = {"nlist": 100, "nprobe": 8, "train_size": None}
        self._ivf_params.update(ivf or {})

        self._index = self._new_index()
        self._ids: list[str] = []

    # ------------------------------------------------------------------
    # Index factory
    # ------------------------------------------------------------------
    def _new_index(self) -> faiss.Index:
        it = self._index_type
        if it in ("flat_ip", "flat_l2"):
            return (
                faiss.IndexFlatIP(self._dim)
                if it == "flat_ip"
                else faiss.IndexFlatL2(self._dim)
            )
        if it == "hnsw":
            M = int(self._hnsw_params["M"])
            idx = faiss.IndexHNSWFlat(self._dim, M, self._faiss_metric)
            idx.hnsw.efConstruction = int(self._hnsw_params["ef_construction"])
            idx.hnsw.efSearch = int(self._hnsw_params["ef_search"])
            return idx
        if it == "ivf_flat":
            quantizer = (
                faiss.IndexFlatIP(self._dim)
                if self._faiss_metric == _METRIC_IP
                else faiss.IndexFlatL2(self._dim)
            )
            idx = faiss.IndexIVFFlat(
                quantizer, self._dim,
                int(self._ivf_params["nlist"]),
                self._faiss_metric,
            )
            idx.nprobe = int(self._ivf_params["nprobe"])
            return idx
        raise ValueError(f"unreachable index_type: {it!r}")  # pragma: no cover

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return bool(self._index.is_trained)

    # ------------------------------------------------------------------
    # Runtime tuning (mainly for HNSW queries)
    # ------------------------------------------------------------------
    def set_ef_search(self, ef: int) -> None:
        """Adjust HNSW query-time width (``efSearch``) at runtime."""
        if self._index_type != "hnsw":
            raise ValueError("set_ef_search only applies to hnsw indexes")
        self._index.hnsw.efSearch = int(ef)
        self._hnsw_params["ef_search"] = int(ef)

    def set_nprobe(self, nprobe: int) -> None:
        """Adjust IVF probe count at runtime."""
        if self._index_type != "ivf_flat":
            raise ValueError("set_nprobe only applies to ivf_flat indexes")
        self._index.nprobe = int(nprobe)
        self._ivf_params["nprobe"] = int(nprobe)

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
            raise ValueError(f"vector dim {vectors.shape[1]} != index dim {self._dim}")
        vecs = np.ascontiguousarray(vectors.astype(np.float32))

        # IVF must be trained before first add.
        if self._index_type == "ivf_flat" and not self._index.is_trained:
            nlist = int(self._ivf_params["nlist"])
            train_size = self._ivf_params.get("train_size") or vecs.shape[0]
            if vecs.shape[0] < nlist:
                raise ChimeraError(
                    f"IVF index requires at least nlist={nlist} training points, "
                    f"got {vecs.shape[0]}. Lower nlist or provide more data."
                )
            self._index.train(vecs[:train_size])

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

        hits: list[tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            score = float(dist) if self._faiss_metric == _METRIC_IP else float(-dist)
            hits.append((self._ids[idx], score))
        return hits

    # ------------------------------------------------------------------
    # Persistence — writes two files: <path>.faiss and <path>.ids.pkl
    # ------------------------------------------------------------------
    def persist(self, path: str) -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(base) + ".faiss")
        meta = {
            "ids": self._ids,
            "dim": self._dim,
            "index_type": self._index_type,
            "metric": self._metric_str,
            "hnsw": self._hnsw_params,
            "ivf": self._ivf_params,
        }
        with open(str(base) + ".ids.pkl", "wb") as f:
            pickle.dump(meta, f)

    def load(self, path: str) -> None:
        base = str(path)
        self._index = faiss.read_index(base + ".faiss")
        with open(base + ".ids.pkl", "rb") as f:
            meta = pickle.load(f)
        self._ids = list(meta["ids"])
        self._dim = int(meta["dim"])
        self._index_type = str(meta.get("index_type", meta.get("type", "flat_ip")))
        self._metric_str = str(meta.get("metric", "ip"))
        self._faiss_metric = _resolve_metric(self._metric_str, self._index_type)
        if "hnsw" in meta:
            self._hnsw_params.update(meta["hnsw"])
        if "ivf" in meta:
            self._ivf_params.update(meta["ivf"])


__all__ = ["FAISSVectorStore"]
