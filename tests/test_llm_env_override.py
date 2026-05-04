"""Tests: LLM connection details (key/base_url/model) can be fully driven from .env.

Priority rules we enforce:

1. ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL`` env vars, when set,
   win over the values declared in ``config.yaml``.
2. When those env vars are absent, the provider falls back to
   ``cfg.api_key_env`` / ``cfg.base_url_env`` / ``cfg.model`` as before
   (backward compatibility).
3. ``cfg.model`` may be left blank in config.yaml as long as ``LLM_MODEL``
   is supplied in .env.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chimera_rag.core.config import LLMConfig, RetryConfig


def _cfg(
    provider: str = "deepseek",
    model: str = "",
    api_key_env: str | None = None,
    base_url_env: str | None = None,
) -> LLMConfig:
    return LLMConfig(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        temperature=0.1,
        max_tokens=256,
        timeout=15.0,
        retry=RetryConfig(max_attempts=1, initial_wait=0.01, max_wait=0.05),
    )


def test_llm_api_key_env_is_preferred(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "from-unified")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-legacy")
    from chimera_rag.providers.llm import openai_provider

    with patch.object(openai_provider, "_make_async_client", return_value=MagicMock()) as mk:
        openai_provider.OpenAILLMProvider(_cfg(model="deepseek-chat"))

    assert mk.call_args.kwargs["api_key"] == "from-unified"


def test_llm_base_url_env_is_preferred(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_BASE_URL", "https://custom.example.com/v1")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://legacy.example.com")
    from chimera_rag.providers.llm import openai_provider

    with patch.object(openai_provider, "_make_async_client", return_value=MagicMock()) as mk:
        openai_provider.OpenAILLMProvider(_cfg(model="m"))

    assert mk.call_args.kwargs["base_url"] == "https://custom.example.com/v1"


def test_llm_model_env_is_preferred(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
    from chimera_rag.providers.llm import openai_provider

    with patch.object(openai_provider, "_make_async_client", return_value=MagicMock()):
        provider = openai_provider.OpenAILLMProvider(_cfg(model="deepseek-chat"))

    assert provider.model == "deepseek-reasoner"


def test_model_can_be_empty_in_config_when_env_provides_it(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.setenv("LLM_MODEL", "deepseek-chat")
    from chimera_rag.providers.llm import openai_provider

    with patch.object(openai_provider, "_make_async_client", return_value=MagicMock()):
        provider = openai_provider.OpenAILLMProvider(_cfg(model=""))

    assert provider.model == "deepseek-chat"


def test_missing_both_env_and_config_model_raises(monkeypatch):
    """If the provider has no built-in default, model must be set somewhere."""
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    from chimera_rag.core.exceptions import ProviderError
    from chimera_rag.providers.llm import openai_provider

    # Patch the default-model table to return None, simulating a provider
    # without a baked-in default.
    monkeypatch.setattr(openai_provider, "_default_model", lambda _p: None)

    with patch.object(openai_provider, "_make_async_client", return_value=MagicMock()):
        with pytest.raises(ProviderError, match="model"):
            openai_provider.OpenAILLMProvider(_cfg(model=""))


def test_legacy_deepseek_env_still_works(monkeypatch):
    """Backward-compat: old .env with DEEPSEEK_* vars keeps working."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-legacy")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    from chimera_rag.providers.llm import openai_provider

    cfg = _cfg(
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
    )

    with patch.object(openai_provider, "_make_async_client", return_value=MagicMock()) as mk:
        provider = openai_provider.OpenAILLMProvider(cfg)

    assert mk.call_args.kwargs["api_key"] == "sk-legacy"
    assert mk.call_args.kwargs["base_url"] == "https://api.deepseek.com"
    assert provider.model == "deepseek-chat"
