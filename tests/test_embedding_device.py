"""Tests: embedding.device config is honoured by sentence-transformers.

Defaults to ``cpu``; ``auto`` picks GPU when available.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from chimera_rag.core.config import EmbeddingConfig


def _fake_model(dim: int = 384):
    fake = MagicMock()
    fake.encode.side_effect = lambda texts, **_kw: np.zeros((len(texts), dim), dtype=np.float32)
    fake.get_sentence_embedding_dimension.return_value = dim
    return fake


def test_embedding_config_device_defaults_to_cpu():
    cfg = EmbeddingConfig(
        provider="sentence_transformer",
        model="dummy",
        dim=384,
    )
    assert cfg.device == "cpu"


def test_embedding_config_accepts_explicit_device():
    cfg = EmbeddingConfig(
        provider="sentence_transformer",
        model="dummy",
        dim=384,
        device="cuda",
    )
    assert cfg.device == "cuda"


def test_st_provider_forwards_device_to_model():
    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    cfg = EmbeddingConfig(
        provider="sentence_transformer",
        model="dummy",
        dim=384,
        device="cpu",
    )
    with patch.object(mod, "_load_model", return_value=_fake_model(384)) as mk:
        mod.SentenceTransformerEmbeddingProvider(cfg)

    # _load_model(model_name, device=...)
    assert mk.call_args.kwargs.get("device") == "cpu" or mk.call_args.args[-1] == "cpu"


def test_st_provider_auto_resolves_to_cpu_when_no_accelerator(monkeypatch):
    # Force both CUDA and MPS to be unavailable
    import torch

    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    cfg = EmbeddingConfig(
        provider="sentence_transformer",
        model="dummy",
        dim=384,
        device="auto",
    )
    with patch.object(mod, "_load_model", return_value=_fake_model(384)) as mk:
        mod.SentenceTransformerEmbeddingProvider(cfg)

    # 'auto' should resolve to 'cpu' in this scenario
    resolved = mk.call_args.kwargs.get("device")
    assert resolved == "cpu"


def test_st_provider_auto_prefers_cuda_when_available(monkeypatch):
    import torch

    from chimera_rag.providers.embedding import sentence_transformer_embedding as mod

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    cfg = EmbeddingConfig(
        provider="sentence_transformer",
        model="dummy",
        dim=384,
        device="auto",
    )
    with patch.object(mod, "_load_model", return_value=_fake_model(384)) as mk:
        mod.SentenceTransformerEmbeddingProvider(cfg)

    assert mk.call_args.kwargs.get("device") == "cuda"
