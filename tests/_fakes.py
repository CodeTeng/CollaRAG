"""Test-only fake provider doubles (extracted from conftest.py for direct imports).

These are identical to the classes in conftest.py; they exist here so tests
that directly instantiate them (rather than relying on the autouse monkeypatch)
can import them with a simple ``from _fakes import FakeLLMProvider``.

Do NOT use these in production code.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from chimera_rag.interfaces.embedding_provider import BaseEmbeddingProvider
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

DEFAULT_LLM_RESPONSE = '{"result": "mock"}'


class FakeLLMProvider(BaseLLMProvider):
    """Offline LLM double. 100% deterministic, no network calls."""

    def __init__(
        self,
        rules: list[tuple[str, str]] | None = None,
        default_response: str = DEFAULT_LLM_RESPONSE,
    ) -> None:
        self.rules: list[tuple[str, str]] = list(rules or [])
        self.default_response = default_response
        self.calls: list[str] = []

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        for keyword, response in self.rules:
            if keyword.lower() in prompt.lower():
                return response
        return self.default_response


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Pure, deterministic embedding double (BLAKE2b hash)."""

    def __init__(self, dim: int = 64, normalize: bool = False) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim
        self._normalize = normalize

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = np.stack([self._encode_one(t) for t in texts]).astype(np.float32)
        if self._normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms
        return vectors

    def _encode_one(self, text: str) -> np.ndarray:
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
        arr = arr / (2**31) - 1.0
        return arr
