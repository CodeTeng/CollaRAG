"""Tests for EmbeddingProvider (OpenAI-compatible embeddings client)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from chimera_rag.core.config import EmbeddingConfig
from chimera_rag.core.exceptions import ProviderError


def _fake_embedding_response(dim: int, texts: list[str]):
    """Return a fake response object with .data[n].embedding."""
    return SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1] * dim)
            for _ in texts
        ]
    )


def _build_cfg(**overrides) -> EmbeddingConfig:
    defaults = dict(
        model="test-emb",
        dim=1536,
        api_key_env="TEST_API_KEY",
        batch_size=32,
        normalize=True,
    )
    defaults.update(overrides)
    return EmbeddingConfig(**defaults)


def test_embedding_provider_is_instance_of_base_provider():
    from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider
    from chimera_rag.providers.embedding import EmbeddingProvider

    assert issubclass(EmbeddingProvider, BaseEmbeddingProvider)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    from chimera_rag.providers.embedding import EmbeddingProvider

    with pytest.raises(ProviderError, match="TEST_API_KEY"):
        EmbeddingProvider(_build_cfg())


def test_encode_returns_correct_shape(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import embedding

    fake_client = MagicMock()
    fake_client.embeddings.create = MagicMock(
        return_value=_fake_embedding_response(1536, ["a", "b", "c"])
    )

    with patch.object(embedding, "_make_client", return_value=fake_client):
        p = embedding.EmbeddingProvider(_build_cfg())
        out = p.encode(["a", "b", "c"])

    assert out.shape == (3, 1536)


def test_encode_batches_large_input(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import embedding

    fake_client = MagicMock()
    fake_client.embeddings.create = MagicMock(
        side_effect=lambda model, input: _fake_embedding_response(64, input)
    )

    with patch.object(embedding, "_make_client", return_value=fake_client):
        p = embedding.EmbeddingProvider(_build_cfg(dim=64, batch_size=2))
        out = p.encode(["a", "b", "c", "d", "e"])

    assert out.shape == (5, 64)
    # 3 batches: 2 + 2 + 1
    assert fake_client.embeddings.create.call_count == 3


def test_encode_normalizes(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import embedding

    fake_client = MagicMock()
    fake_client.embeddings.create = MagicMock(
        return_value=_fake_embedding_response(64, ["a"])
    )

    with patch.object(embedding, "_make_client", return_value=fake_client):
        p = embedding.EmbeddingProvider(_build_cfg(dim=64, normalize=True))
        out = p.encode(["a"])

    norm = np.linalg.norm(out)
    assert np.allclose(norm, 1.0, atol=1e-5)


def test_encode_empty_returns_zero_rows(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import embedding

    with patch.object(embedding, "_make_client", return_value=MagicMock()):
        p = embedding.EmbeddingProvider(_build_cfg())
        out = p.encode([])

    assert out.shape == (0, 1536)


def test_dim_mismatch_raises(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import embedding

    fake_client = MagicMock()
    # Server returns 768-dim, but config says 1536
    fake_client.embeddings.create = MagicMock(
        return_value=_fake_embedding_response(768, ["a"])
    )

    with patch.object(embedding, "_make_client", return_value=fake_client):
        p = embedding.EmbeddingProvider(_build_cfg(dim=1536))
        with pytest.raises(ProviderError, match="dim"):
            p.encode(["a"])
