"""Tests for the async ABCs under :mod:`chimera_rag.interfaces`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_base_llm_provider_is_abstract_and_subclass_works():
    from chimera_rag.interfaces.llm_provider import BaseLLMProvider

    with pytest.raises(TypeError):
        BaseLLMProvider()

    class StaticLLM(BaseLLMProvider):
        async def complete(self, prompt: str, **kwargs) -> str:
            return f"echo: {prompt}"

    out = await StaticLLM().complete("hi")
    assert out == "echo: hi"
