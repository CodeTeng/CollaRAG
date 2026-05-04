"""Local sentence-transformers embedding provider.

Uses a HuggingFace sentence-transformer model (default:
``sentence-transformers/all-MiniLM-L6-v2``, 384-dim) to produce
deterministic, offline-friendly embeddings. Well-suited for DeepSeek
pipelines, as DeepSeek's API does not expose an embedding endpoint.

The model is loaded lazily via :func:`_load_model` so tests can patch it
without requiring the ~90 MB weights on disk.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from chimera_rag.core.exceptions import ProviderError
from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider

if TYPE_CHECKING:
    from chimera_rag.core.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def _load_model(model_name: str) -> Any:
    """Isolated factory so tests can monkeypatch without the heavy import."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:  # pragma: no cover - optional
        raise ProviderError(
            f"sentence-transformers not installed; run `uv sync --extra st` ({e})"
        ) from e
    logger.info("loading sentence-transformers model %r (first load may download weights)", model_name)
    return SentenceTransformer(model_name)


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Local HF sentence-transformer encoder."""

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self._normalize = bool(cfg.normalize)
        self._batch_size = int(cfg.batch_size)

        self._model = _load_model(cfg.model)

        actual_dim = int(self._model.get_sentence_embedding_dimension())
        if actual_dim != cfg.dim:
            raise ProviderError(
                f"configured embedding dim={cfg.dim} does not match model dim={actual_dim} "
                f"(model={cfg.model!r}). Update config.embedding.dim."
            )
        self._dim = actual_dim

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)

        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms

        return vectors


__all__ = ["SentenceTransformerEmbeddingProvider"]
