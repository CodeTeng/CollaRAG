"""ABC for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """Text-in / text-out async LLM client.

    Concrete providers (OpenAI, DeepSeek, Ollama, Mock) normalise their
    respective SDKs behind this one method. Keep the signature minimal;
    streaming / function-calling extensions go in subclasses.
    """

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> str: ...


__all__ = ["BaseLLMProvider"]
