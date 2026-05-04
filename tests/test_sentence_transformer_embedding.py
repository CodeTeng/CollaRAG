"""Tests for SentenceTransformerEmbeddingProvider.

The real model is mocked to keep the test suite fast and offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chimera_rag.core.config import EmbeddingConfig


def _build_cfg(dim: int = 384) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="sentence_transformer",
        model="sentence-transformers/all-MiniLM-L6-v2",
        dim=dim,
        batch_size=16,
        normalize=True,
    )


def _fake_model(dim: int = 384):
    fake = MagicMock()

    def _encode(texts, **_kw):
        arr = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # Trivial deterministic filler based on text length.
            arr[i, 0] = float(len(t) % 100) / 100.0
        return arr

    fake.encode.side_effect = _encode
    fake.get_sentence_embedding_dimension.return_value = dim
    return fake


def test_dim_property_reflects_model():
    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    with patch.object(mod, "_load_model", return_value=_fake_model(384)):
        provider = mod.SentenceTransformerEmbeddingProvider(_build_cfg(384))

    assert provider.dim == 384


def test_encode_returns_2d_array_of_right_shape():
    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    with patch.object(mod, "_load_model", return_value=_fake_model(384)):
        provider = mod.SentenceTransformerEmbeddingProvider(_build_cfg(384))
        out = provider.encode(["hello", "world", "foo"])

    assert out.shape == (3, 384)
    assert out.dtype == np.float32


def test_encode_empty_returns_empty_array():
    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    with patch.object(mod, "_load_model", return_value=_fake_model(384)):
        provider = mod.SentenceTransformerEmbeddingProvider(_build_cfg(384))
        out = provider.encode([])

    assert out.shape == (0, 384)


def test_normalize_produces_unit_vectors():
    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    fake = _fake_model(8)
    # Override encode to produce non-unit vectors so normalise has work to do.
    fake.encode.side_effect = lambda texts, **_kw: np.ones((len(texts), 8), dtype=np.float32) * 2.0

    cfg = _build_cfg(8)
    with patch.object(mod, "_load_model", return_value=fake):
        provider = mod.SentenceTransformerEmbeddingProvider(cfg)
        out = provider.encode(["a", "b"])

    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_dim_mismatch_raises():
    from chimera_rag.core.exceptions import ProviderError
    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    # Config says 384 but model reports 768 -> should raise early
    with patch.object(mod, "_load_model", return_value=_fake_model(768)):
        with pytest.raises(ProviderError, match="dim"):
            mod.SentenceTransformerEmbeddingProvider(_build_cfg(384))


def test_build_embedding_provider_routes_st():
    from chimera_rag.providers import build_embedding_provider
    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    with patch.object(mod, "_load_model", return_value=_fake_model(384)):
        provider = build_embedding_provider(_build_cfg(384))

    assert isinstance(provider, mod.SentenceTransformerEmbeddingProvider)
