"""Tests for :mod:`chimera_rag.stores.vector`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def _faiss_available():
    pytest.importorskip("faiss")


def test_faiss_store_add_and_search_returns_nearest_first(_faiss_available):
    from chimera_rag.stores.vector import FAISSVectorStore

    s = FAISSVectorStore(dim=4, index_type="flat_ip")
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # id_a
            [0.0, 1.0, 0.0, 0.0],  # id_b
            [0.7, 0.7, 0.0, 0.0],  # id_c roughly between a and b
        ],
        dtype=np.float32,
    )
    s.add(["a", "b", "c"], vectors)

    hits = s.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert len(hits) == 2
    assert hits[0][0] == "a"  # nearest is 'a' itself
    # scores must be descending
    assert hits[0][1] >= hits[1][1]


def test_faiss_store_search_top_k_bounded_by_index_size(_faiss_available):
    from chimera_rag.stores.vector import FAISSVectorStore

    s = FAISSVectorStore(dim=4, index_type="flat_ip")
    s.add(["a", "b"], np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    hits = s.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=10)
    assert len(hits) == 2  # only 2 vectors in index


def test_faiss_store_persist_and_load_round_trip(tmp_path: Path, _faiss_available):
    from chimera_rag.stores.vector import FAISSVectorStore

    s1 = FAISSVectorStore(dim=4, index_type="flat_ip")
    s1.add(
        ["a", "b", "c"],
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
    )
    p = tmp_path / "vecs"
    s1.persist(str(p))
    # Both the .faiss file and the .ids.pkl sidecar should exist
    assert (tmp_path / "vecs.faiss").is_file()
    assert (tmp_path / "vecs.ids.pkl").is_file()

    s2 = FAISSVectorStore(dim=4, index_type="flat_ip")
    s2.load(str(p))
    hits = s2.search(np.array([0, 1, 0, 0], dtype=np.float32), top_k=1)
    assert hits[0][0] == "b"


def test_faiss_store_rejects_mismatched_ids_and_vector_count(_faiss_available):
    from chimera_rag.stores.vector import FAISSVectorStore

    s = FAISSVectorStore(dim=4, index_type="flat_ip")
    with pytest.raises(ValueError):
        s.add(["a", "b"], np.array([[1, 0, 0, 0]], dtype=np.float32))
