"""Deterministic mock LLM used by the offline test suite.

Behaviour:

* for each call, the prompt is scanned against ``rules`` (a list of
  ``(keyword, response)`` pairs); the **first** keyword that appears in
  the prompt determines the response,
* if no rule matches, ``default_response`` is returned,
* default ``default_response`` is a tiny, stable JSON-like string so
  naive extractors parsing JSON still get something sensible.
"""

from __future__ import annotations

from typing import Any

from chimera_rag.interfaces.llm_provider import BaseLLMProvider


DEFAULT_RESPONSE = '{"result": "mock"}'


class MockLLMProvider(BaseLLMProvider):
    """Offline LLM double. 100% deterministic, no network calls."""

    def __init__(
        self,
        rules: list[tuple[str, str]] | None = None,
        default_response: str = DEFAULT_RESPONSE,
    ) -> None:
        self.rules: list[tuple[str, str]] = list(rules or [])
        self.default_response = default_response
        # Track calls so tests can assert behaviour without mocks.
        self.calls: list[str] = []

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(prompt)
        for keyword, response in self.rules:
            if keyword.lower() in prompt.lower():
                return response
        return self.default_response


__all__ = ["MockLLMProvider", "DEFAULT_RESPONSE"]
