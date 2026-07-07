"""Tests for LLMProvider (OpenAI-compatible async chat-completion client).

We stub the ``openai.AsyncOpenAI`` client so no real network call fires;
the tests focus on: prompt forwarding, response extraction, retry on
transient errors, API key resolution.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chimera_rag.core.config import LLMConfig, RetryConfig
from chimera_rag.core.exceptions import ProviderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _build_cfg(**overrides) -> LLMConfig:
    defaults = dict(
        model="test-model",
        api_key_env="TEST_API_KEY",
        temperature=0.1,
        max_tokens=256,
        timeout=15.0,
        retry=RetryConfig(max_attempts=2, initial_wait=0.01, max_wait=0.05),
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_missing_api_key_raises_provider_error(monkeypatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    from chimera_rag.providers.llm import LLMProvider

    with pytest.raises(ProviderError, match="TEST_API_KEY"):
        LLMProvider(_build_cfg())


@pytest.mark.asyncio
async def test_complete_returns_text(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import llm

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        return_value=_fake_response("hello world")
    )

    with patch.object(llm, "_make_async_client", return_value=fake_client) as mk:
        provider = llm.LLMProvider(_build_cfg())
        out = await provider.complete("ping")

    assert out == "hello world"
    mk.assert_called_once()
    kwargs = mk.call_args.kwargs
    assert kwargs["api_key"] == "sk-fake"

    call = fake_client.chat.completions.create.await_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["messages"] == [{"role": "user", "content": "ping"}]
    assert call.kwargs["temperature"] == 0.1
    assert call.kwargs["max_tokens"] == 256


@pytest.mark.asyncio
async def test_complete_retries_on_transient_failure(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import llm

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("transient"), _fake_response("ok")]
    )

    with patch.object(llm, "_make_async_client", return_value=fake_client):
        provider = llm.LLMProvider(_build_cfg())
        out = await provider.complete("ping")

    assert out == "ok"
    assert fake_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_complete_raises_provider_error_when_all_attempts_fail(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import llm

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

    with patch.object(llm, "_make_async_client", return_value=fake_client):
        provider = llm.LLMProvider(_build_cfg())
        with pytest.raises(ProviderError, match="boom|failed"):
            await provider.complete("ping")

    assert fake_client.chat.completions.create.await_count == 2


def test_base_url_forwarded_to_client(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers import llm

    with patch.object(llm, "_make_async_client", return_value=MagicMock()) as mk:
        llm.LLMProvider(_build_cfg(base_url="https://api.example.com/v1"))

    assert mk.call_args.kwargs["base_url"] == "https://api.example.com/v1"


def test_llm_provider_is_instance_of_base_provider():
    from chimera_rag.interfaces.llm_provider import BaseLLMProvider
    from chimera_rag.providers.llm import LLMProvider

    assert issubclass(LLMProvider, BaseLLMProvider)
