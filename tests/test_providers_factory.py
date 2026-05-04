"""Tests for the provider factories that read ``AppConfig`` and return
concrete :class:`BaseLLMProvider` / :class:`BaseEmbeddingProvider` instances.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------
def test_build_llm_provider_from_config_mock_branch():
    from chimera_rag.core.config import LLMConfig
    from chimera_rag.providers import build_llm_provider
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    cfg = LLMConfig(provider="mock", model="mock-llm")
    provider = build_llm_provider(cfg)
    assert isinstance(provider, MockLLMProvider)


def test_build_llm_provider_raises_for_unknown_provider_name():
    from chimera_rag.core.exceptions import ProviderError
    from chimera_rag.providers import build_llm_provider

    class _Cfg:
        provider = "martian"
        model = "x"

    with pytest.raises(ProviderError, match="unknown|unsupported"):
        build_llm_provider(_Cfg())


# ---------------------------------------------------------------------------
# Embedding factory
# ---------------------------------------------------------------------------
def test_build_embedding_provider_from_config_mock_branch():
    from chimera_rag.core.config import EmbeddingConfig
    from chimera_rag.providers import build_embedding_provider
    from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

    cfg = EmbeddingConfig(provider="mock", model="mock-embedding", dim=32, normalize=True)
    provider = build_embedding_provider(cfg)
    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.dim == 32


def test_build_embedding_provider_raises_for_unknown_provider_name():
    from chimera_rag.core.exceptions import ProviderError
    from chimera_rag.providers import build_embedding_provider

    class _Cfg:
        provider = "telegraph"
        model = "x"
        dim = 8
        normalize = True

    with pytest.raises(ProviderError, match="unknown|unsupported"):
        build_embedding_provider(_Cfg())
