"""Deterministic hash-based embedding used for offline tests.

Each text is hashed with BLAKE2b and stretched to ``dim`` floats in
``[-1, 1]``. The mapping is pure (no RNG state), so identical texts
always yield identical vectors, and distinct texts yield distinct vectors
with overwhelming probability.
"""

from __future__ import annotations

import hashlib

import numpy as np

from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Pure, deterministic embedding double."""

    def __init__(self, dim: int = 64, normalize: bool = False) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        vectors = np.stack([self._encode_one(t) for t in texts]).astype(np.float32)

        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Protect against zero-vectors.
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms
        return vectors

    # ------------------------------------------------------------------
    def _encode_one(self, text: str) -> np.ndarray:
        """Hash ``text`` into ``dim`` floats in [-1, 1] deterministically."""
        # We need ``dim`` floats. Each float comes from 4 bytes, so we ask
        # blake2b for 4*dim bytes (digest_size <= 64, so chain for larger dims).
        needed = 4 * self._dim
        buf = bytearray()
        counter = 0
        while len(buf) < needed:
            h = hashlib.blake2b(
                text.encode("utf-8") + counter.to_bytes(4, "little"),
                digest_size=min(64, needed - len(buf)),
            )
            buf.extend(h.digest())
            counter += 1
        arr = np.frombuffer(bytes(buf[:needed]), dtype=np.uint32).astype(np.float64)
        # Map uint32 -> [-1, 1).
        arr = arr / (2**31) - 1.0
        return arr


__all__ = ["MockEmbeddingProvider"]
