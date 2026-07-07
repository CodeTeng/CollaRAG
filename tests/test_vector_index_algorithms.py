"""Tests for extended FAISSVectorStore: flat / ivf / hnsw + per-algo params."""

from __future__ import annotations

import numpy as np
import pytest

DIM = 8


def _vecs(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    # normalise so IP and cosine are equivalent
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


# ---------------------------------------------------------------------------
# Flat index (baseline behaviour must stay intact)
# ---------------------------------------------------------------------------
def test_flat_ip_backward_compat():
    from chimera_rag.stores.vector import FAISSVectorStore

    store = FAISSVectorStore(dim=DIM, index_type="flat_ip")
    ids = [f"c{i}" for i in range(10)]
    store.add(ids, _vecs(10))
    hits = store.search(_vecs(1)[0], top_k=3)
    assert len(hits) == 3
    assert hits[0][1] >= hits[1][1] >= hits[2][1]  # IP descending


def test_flat_l2_backward_compat():
    from chimera_rag.stores.vector import FAISSVectorStore

    store = FAISSVectorStore(dim=DIM, index_type="flat_l2")
    store.add(["a", "b", "c"], _vecs(3))
    hits = store.search(_vecs(1)[0], top_k=2)
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# HNSW
# ---------------------------------------------------------------------------
def test_hnsw_index_creates_without_train():
    from chimera_rag.stores.vector import FAISSVectorStore

    store = FAISSVectorStore(
        dim=DIM,
        index_type="hnsw",
        metric="ip",
        hnsw={"M": 16, "ef_construction": 100, "ef_search": 32},
    )
    store.add([f"c{i}" for i in range(50)], _vecs(50))
    hits = store.search(_vecs(1)[0], top_k=5)
    assert len(hits) == 5
    # hnsw with IP: scores descending
    assert all(hits[i][1] >= hits[i + 1][1] for i in range(len(hits) - 1))


def test_hnsw_ef_search_applied_at_query_time():
    """Raising ef_search should not break results (correctness invariant)."""
    from chimera_rag.stores.vector import FAISSVectorStore

    store = FAISSVectorStore(
        dim=DIM, index_type="hnsw", metric="ip",
        hnsw={"M": 16, "ef_construction": 100, "ef_search": 10},
    )
    vs = _vecs(100)
    store.add([f"c{i}" for i in range(100)], vs)
    # Raising ef_search should at least not drop recall
    hits_low = store.search(vs[0], top_k=5)
    store.set_ef_search(128)
    hits_high = store.search(vs[0], top_k=5)
    # The top-1 should be the vector itself in both cases
    assert hits_low[0][0] == "c0"
    assert hits_high[0][0] == "c0"


# ---------------------------------------------------------------------------
# IVF
# ---------------------------------------------------------------------------
def test_ivf_flat_index_trains_and_queries():
    from chimera_rag.stores.vector import FAISSVectorStore

    store = FAISSVectorStore(
        dim=DIM,
        index_type="ivf_flat",
        metric="ip",
        ivf={"nlist": 8, "nprobe": 4},
    )
    vs = _vecs(200)
    store.add([f"c{i}" for i in range(200)], vs)

    # After add, the internal index must be trained
    assert store.is_trained

    hits = store.search(vs[0], top_k=5)
    assert len(hits) == 5
    assert hits[0][0] == "c0"


def test_ivf_flat_refuses_add_when_corpus_too_small():
    """IVF needs >= nlist training points; small corpus should raise clearly."""
    from chimera_rag.core.exceptions import ChimeraError
    from chimera_rag.stores.vector import FAISSVectorStore

    store = FAISSVectorStore(
        dim=DIM, index_type="ivf_flat", metric="ip",
        ivf={"nlist": 64, "nprobe": 4},
    )
    with pytest.raises(ChimeraError, match="nlist|training"):
        store.add([f"c{i}" for i in range(10)], _vecs(10))


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------
def test_vector_store_config_accepts_new_index_types():
    from chimera_rag.core.config import VectorStoreConfig

    # all three should validate
    for idx in ("flat_ip", "flat_l2", "hnsw", "ivf_flat"):
        VectorStoreConfig(backend="faiss", index_type=idx, persist_path="./v")


def test_vector_store_config_defaults_to_hnsw():
    from chimera_rag.core.config import VectorStoreConfig

    cfg = VectorStoreConfig(backend="faiss", persist_path="./v")
    assert cfg.index_type == "hnsw"


def test_hnsw_params_default_in_config():
    from chimera_rag.core.config import VectorStoreConfig

    cfg = VectorStoreConfig(backend="faiss", index_type="hnsw", persist_path="./v")
    assert cfg.hnsw.M > 0
    assert cfg.hnsw.ef_construction > 0
    assert cfg.hnsw.ef_search > 0


def test_ivf_params_default_in_config():
    from chimera_rag.core.config import VectorStoreConfig

    cfg = VectorStoreConfig(backend="faiss", index_type="ivf_flat", persist_path="./v")
    assert cfg.ivf.nlist > 0
    assert cfg.ivf.nprobe > 0


# ---------------------------------------------------------------------------
# Persistence round-trip for HNSW
# ---------------------------------------------------------------------------
def test_hnsw_persist_roundtrip(tmp_path):
    from chimera_rag.stores.vector import FAISSVectorStore

    store = FAISSVectorStore(dim=DIM, index_type="hnsw", metric="ip")
    ids = [f"c{i}" for i in range(30)]
    vs = _vecs(30)
    store.add(ids, vs)
    path = tmp_path / "idx"
    store.persist(str(path))

    store2 = FAISSVectorStore(dim=DIM, index_type="hnsw", metric="ip")
    store2.load(str(path))
    hits = store2.search(vs[0], top_k=3)
    assert hits[0][0] == "c0"
