"""OpenAI-compatible embedding provider.

Single implementation for any OpenAI-protocol-compatible embeddings endpoint
(OpenAI, Ollama, vLLM, Xinference, SiliconFlow, etc.).

All model / connection details live in the config file; only the API key
comes from the environment (default ``LLM_API_KEY``).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from chimera_rag.core.exceptions import ProviderError
from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider

if TYPE_CHECKING:
    from chimera_rag.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def _make_client(*, api_key: str, base_url: str | None, timeout: float):
    """Isolated factory so tests can monkeypatch without touching the openai SDK."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as e:  # pragma: no cover - optional dep
        raise ProviderError(
            f"openai SDK not installed; run `uv sync --extra llm` ({e})"
        ) from e

    kwargs: dict = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class EmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI-compatible embeddings client.

    All model/connection details come from the config object; the API key
    is resolved from the env var named by ``cfg.api_key_env`` (defaults to
    ``LLM_API_KEY``).
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self._normalize = bool(cfg.normalize)
        self._batch_size = int(cfg.batch_size)
        self._dim = int(cfg.dim)

        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise ProviderError(
                f"API key not found; set {cfg.api_key_env} in .env."
            )

        self._client = _make_client(
            api_key=api_key,
            base_url=cfg.base_url,
            timeout=30.0,
        )

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Send one batch of texts to the embeddings endpoint."""
        resp = self._client.embeddings.create(
            model=self.cfg.model,
            input=texts,
        )
        vectors = [d.embedding for d in resp.data]
        actual_dim = len(vectors[0]) if vectors else 0
        if actual_dim != self._dim:
            raise ProviderError(
                f"configured embedding dim={self._dim} does not match "
                f"endpoint dim={actual_dim} (model={self.cfg.model!r}). "
                f"Update config.embedding.dim."
            )
        return np.asarray(vectors, dtype=np.float32)

    # ------------------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        all_vectors: list[np.ndarray] = []
        bs = self._batch_size
        for i in range(0, len(texts), bs):
            all_vectors.append(self._encode_batch(texts[i : i + bs]))

        vectors = np.concatenate(all_vectors, axis=0).astype(np.float32)

        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms

        return vectors


__all__ = ["EmbeddingProvider"]
