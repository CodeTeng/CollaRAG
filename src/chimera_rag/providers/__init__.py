"""LLM and Embedding providers (real backends + deterministic mocks).

Exposes :func:`build_llm_provider` and :func:`build_embedding_provider`
factories that read the typed config and instantiate the right backend.
Real-backend modules (OpenAI / Ollama / SentenceTransformer) are imported
lazily so an optional dependency missing doesn't prevent the Mock path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chimera_rag.core.exceptions import ProviderError
from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

if TYPE_CHECKING:
    from chimera_rag.core.config import EmbeddingConfig, LLMConfig


def build_llm_provider(cfg: "LLMConfig | Any") -> BaseLLMProvider:
    """Instantiate a concrete LLM provider from the typed config."""
    name = getattr(cfg, "provider", None)

    if name == "mock":
        from chimera_rag.providers.llm.mock_provider import MockLLMProvider

        return MockLLMProvider()

    if name in {"openai", "deepseek"}:
        try:
            from chimera_rag.providers.llm.openai_provider import OpenAILLMProvider
        except ImportError as e:  # pragma: no cover - optional dep path
            raise ProviderError(
                f"openai extras not installed; run `uv sync --extra llm` ({e})"
            ) from e
        return OpenAILLMProvider(cfg)

    if name == "ollama":
        try:
            from chimera_rag.providers.llm.ollama_provider import OllamaLLMProvider
        except ImportError as e:  # pragma: no cover - optional dep path
            raise ProviderError(
                f"ollama extras not installed; run `uv sync --extra llm` ({e})"
            ) from e
        return OllamaLLMProvider(cfg)

    raise ProviderError(f"unknown / unsupported LLM provider: {name!r}")


def build_embedding_provider(cfg: "EmbeddingConfig | Any") -> BaseEmbeddingProvider:
    """Instantiate a concrete Embedding provider from the typed config."""
    name = getattr(cfg, "provider", None)
    dim = getattr(cfg, "dim", 64)
    normalize = getattr(cfg, "normalize", True)

    if name == "mock":
        from chimera_rag.providers.embedding.mock_embedding import MockEmbeddingProvider

        return MockEmbeddingProvider(dim=dim, normalize=normalize)

    if name == "openai":
        try:
            from chimera_rag.providers.embedding.openai_embedding import OpenAIEmbeddingProvider
        except ImportError as e:  # pragma: no cover - optional
            raise ProviderError(
                f"openai extras not installed; run `uv sync --extra llm` ({e})"
            ) from e
        return OpenAIEmbeddingProvider(cfg)

    if name == "ollama":
        try:
            from chimera_rag.providers.embedding.ollama_embedding import OllamaEmbeddingProvider
        except ImportError as e:  # pragma: no cover - optional
            raise ProviderError(
                f"ollama extras not installed; run `uv sync --extra llm` ({e})"
            ) from e
        return OllamaEmbeddingProvider(cfg)

    if name == "sentence_transformer":
        try:
            from chimera_rag.providers.embedding.sentence_transformer_embedding import (
                SentenceTransformerEmbeddingProvider,
            )
        except ImportError as e:  # pragma: no cover - optional
            raise ProviderError(
                f"sentence-transformers extras not installed; "
                f"run `uv sync --extra st` ({e})"
            ) from e
        return SentenceTransformerEmbeddingProvider(cfg)

    raise ProviderError(f"unknown / unsupported embedding provider: {name!r}")


__all__ = ["build_embedding_provider", "build_llm_provider"]
