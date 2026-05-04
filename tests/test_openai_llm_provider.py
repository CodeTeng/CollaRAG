"""Tests for OpenAILLMProvider (also serves DeepSeek via base_url).

We stub the ``openai.AsyncOpenAI`` client so no real network call fires;
the tests focus on contract: prompt forwarding, response extraction,
retry on transient errors, and DeepSeek base_url wiring.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chimera_rag.core.config import LLMConfig, RetryConfig
from chimera_rag.core.exceptions import ProviderError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_response(text: str):
    """Return a fake object with .choices[0].message.content = text."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _build_cfg(provider: str = "deepseek") -> LLMConfig:
    return LLMConfig(
        provider=provider,
        model="deepseek-chat" if provider == "deepseek" else "gpt-4o-mini",
        api_key_env="DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL" if provider == "deepseek" else None,
        temperature=0.1,
        max_tokens=256,
        timeout=15.0,
        retry=RetryConfig(max_attempts=2, initial_wait=0.01, max_wait=0.05),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_missing_api_key_raises_provider_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    from chimera_rag.providers.llm.openai_provider import OpenAILLMProvider

    with pytest.raises(ProviderError, match="DEEPSEEK_API_KEY"):
        OpenAILLMProvider(_build_cfg("deepseek"))


@pytest.mark.asyncio
async def test_complete_returns_text(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    from chimera_rag.providers.llm import openai_provider

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=_fake_response("hello world"))

    with patch.object(openai_provider, "_make_async_client", return_value=fake_client) as mk:
        provider = openai_provider.OpenAILLMProvider(_build_cfg("deepseek"))
        out = await provider.complete("ping")

    assert out == "hello world"
    mk.assert_called_once()
    # Confirm base_url forwarded
    kwargs = mk.call_args.kwargs
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["api_key"] == "sk-fake"

    # Confirm chat.completions.create was called with the right payload
    call = fake_client.chat.completions.create.await_args
    assert call.kwargs["model"] == "deepseek-chat"
    assert call.kwargs["messages"] == [{"role": "user", "content": "ping"}]
    assert call.kwargs["temperature"] == 0.1
    assert call.kwargs["max_tokens"] == 256


@pytest.mark.asyncio
async def test_complete_retries_on_transient_failure(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    from chimera_rag.providers.llm import openai_provider

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=[RuntimeError("transient"), _fake_response("ok")]
    )

    with patch.object(openai_provider, "_make_async_client", return_value=fake_client):
        provider = openai_provider.OpenAILLMProvider(_build_cfg("deepseek"))
        out = await provider.complete("ping")

    assert out == "ok"
    # Two attempts total: first fails, second succeeds.
    assert fake_client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_complete_raises_provider_error_when_all_attempts_fail(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")

    from chimera_rag.providers.llm import openai_provider

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

    with patch.object(openai_provider, "_make_async_client", return_value=fake_client):
        provider = openai_provider.OpenAILLMProvider(_build_cfg("deepseek"))
        with pytest.raises(ProviderError, match="boom|failed"):
            await provider.complete("ping")

    assert fake_client.chat.completions.create.await_count == 2


def test_openai_provider_uses_default_base_url_when_env_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    from chimera_rag.providers.llm import openai_provider

    with patch.object(openai_provider, "_make_async_client") as mk:
        mk.return_value = MagicMock()
        openai_provider.OpenAILLMProvider(_build_cfg("openai"))

    kwargs = mk.call_args.kwargs
    # No base_url_env configured for openai in this fixture
    assert kwargs.get("base_url") is None
    assert kwargs["api_key"] == "sk-fake"


def test_build_llm_provider_routes_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    from chimera_rag.providers import build_llm_provider
    from chimera_rag.providers.llm import openai_provider

    with patch.object(openai_provider, "_make_async_client", return_value=MagicMock()):
        provider = build_llm_provider(_build_cfg("deepseek"))

    assert isinstance(provider, openai_provider.OpenAILLMProvider)
    assert provider.model == "deepseek-chat"


# Safety: these tests must not depend on a real key being in os.environ.
def test_fixtures_do_not_leak_real_env():
    # This doesn't actually assert anything about the user's env; it just
    # documents that tests should never read a real key.
    assert os.getenv("DEEPSEEK_API_KEY_SHOULD_NOT_EXIST") is None
