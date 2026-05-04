"""Tests for :mod:`chimera_rag.providers.embedding.mock_embedding`.

The mock embedding must be:
* deterministic (same text -> same vector across calls),
* shape-correct ``(N, dim)``,
* vaguely semantic (different texts -> different vectors),
* normalised when requested.
"""

from __future__ import annotations

import numpy as np


def test_mock_embedding_is_instance_of_base_provider():
    from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

    assert isinstance(MockEmbeddingProvider(dim=8), BaseEmbeddingProvider)


def test_mock_embedding_shape_is_correct():
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

    m = MockEmbeddingProvider(dim=16)
    out = m.encode(["hello", "world", "chimera"])
    assert out.shape == (3, 16)


def test_mock_embedding_is_deterministic_for_same_text():
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

    m = MockEmbeddingProvider(dim=8)
    a = m.encode(["hello"])
    b = m.encode(["hello"])
    assert np.allclose(a, b)


def test_mock_embedding_differs_across_distinct_texts():
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

    m = MockEmbeddingProvider(dim=16)
    v = m.encode(["hello", "world"])
    assert not np.allclose(v[0], v[1])


def test_mock_embedding_normalises_when_requested():
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

    m = MockEmbeddingProvider(dim=8, normalize=True)
    out = m.encode(["alpha", "beta", "gamma"])
    norms = np.linalg.norm(out, axis=1)
    # All unit norm (within fp tolerance)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_mock_embedding_dim_property():
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

    assert MockEmbeddingProvider(dim=32).dim == 32


def test_mock_embedding_empty_input_returns_zero_rows():
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

    m = MockEmbeddingProvider(dim=4)
    out = m.encode([])
    assert out.shape == (0, 4)
