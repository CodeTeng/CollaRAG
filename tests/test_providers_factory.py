"""Tests for the provider factories and concrete providers.

Since the autouse conftest fixture monkeypatches the factories, these tests
import the concrete provider classes directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------
def test_llm_provider_instantiation(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    from chimera_rag.core.config import LLMConfig
    from chimera_rag.providers.llm import LLMProvider

    with patch(
        "chimera_rag.providers.llm._make_async_client",
        return_value=MagicMock(),
    ):
        cfg = LLMConfig(model="test-model", api_key_env="FAKE_KEY")
        provider = LLMProvider(cfg)
        assert isinstance(provider, LLMProvider)


def test_llm_provider_raises_when_api_key_missing(monkeypatch):
    from chimera_rag.core.config import LLMConfig
    from chimera_rag.core.exceptions import ProviderError
    from chimera_rag.providers.llm import LLMProvider

    monkeypatch.delenv("FAKE_KEY", raising=False)
    cfg = LLMConfig(model="test-model", api_key_env="FAKE_KEY")
    with pytest.raises(ProviderError, match="FAKE_KEY"):
        LLMProvider(cfg)


# ---------------------------------------------------------------------------
# Embedding provider
# ---------------------------------------------------------------------------
def test_embedding_provider_instantiation(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "sk-test")
    from chimera_rag.core.config import EmbeddingConfig
    from chimera_rag.providers.embedding import EmbeddingProvider

    with patch(
        "chimera_rag.providers.embedding._make_client",
        return_value=MagicMock(),
    ):
        cfg = EmbeddingConfig(
            model="test-emb", dim=1536, api_key_env="FAKE_KEY",
        )
        provider = EmbeddingProvider(cfg)
        assert isinstance(provider, EmbeddingProvider)
        assert provider.dim == 1536


# ---------------------------------------------------------------------------
# Rerank provider
# ---------------------------------------------------------------------------
def test_rerank_provider_returns_none_when_disabled():
    from chimera_rag.core.config import RerankConfig
    from chimera_rag.providers import build_rerank_provider

    cfg = RerankConfig(enabled=False)
    assert build_rerank_provider(cfg) is None


def test_rerank_provider_instantiation():
    from chimera_rag.core.config import RerankConfig
    from chimera_rag.providers.rerank import RerankProvider

    cfg = RerankConfig(
        enabled=True,
        model="bge-reranker-v2-m3",
        base_url="http://localhost:9997/v1",
        api_key_env="FAKE_KEY",
    )
    with patch(
        "chimera_rag.providers.rerank.os.getenv",
        return_value="fake-api-key",
    ):
        provider = RerankProvider(cfg)
        assert isinstance(provider, RerankProvider)
