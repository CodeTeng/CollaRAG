"""LLM, Embedding, and Rerank providers — all OpenAI-compatible.

Factories read the typed config and instantiate the right backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider
from chimera_rag.interfaces.llm_provider import BaseLLMProvider
from chimera_rag.interfaces.rerank_provider import BaseRerankProvider

if TYPE_CHECKING:
    from chimera_rag.core.config import EmbeddingConfig, LLMConfig, RerankConfig


def build_llm_provider(cfg: LLMConfig | Any) -> BaseLLMProvider:
    from chimera_rag.providers.llm import LLMProvider

    return LLMProvider(cfg)


def build_embedding_provider(cfg: EmbeddingConfig | Any) -> BaseEmbeddingProvider:
    from chimera_rag.providers.embedding import EmbeddingProvider

    return EmbeddingProvider(cfg)


def build_rerank_provider(cfg: RerankConfig | Any) -> BaseRerankProvider | None:
    if not cfg.enabled:
        return None
    from chimera_rag.providers.rerank import RerankProvider

    return RerankProvider(cfg)


__all__ = [
    "build_embedding_provider",
    "build_llm_provider",
    "build_rerank_provider",
]