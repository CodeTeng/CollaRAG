"""Tests: LLM provider resolves the API key from the env var named by
``cfg.api_key_env`` (default ``LLM_API_KEY``).

Model and base_url are no longer read from the environment — they live
exclusively in ``config.yaml``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chimera_rag.core.config import LLMConfig, RetryConfig


def _cfg(*, model: str = "test-model", api_key_env: str = "LLM_API_KEY") -> LLMConfig:
    return LLMConfig(
        model=model,
        api_key_env=api_key_env,
        temperature=0.1,
        max_tokens=256,
        timeout=15.0,
        retry=RetryConfig(max_attempts=1, initial_wait=0.01, max_wait=0.05),
    )


def test_api_key_resolved_from_default_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    from chimera_rag.providers import llm

    with patch.object(llm, "_make_async_client", return_value=MagicMock()) as mk:
        llm.LLMProvider(_cfg())

    assert mk.call_args.kwargs["api_key"] == "sk-test"


def test_api_key_resolved_from_custom_env(monkeypatch):
    monkeypatch.setenv("CUSTOM_KEY", "sk-custom")
    from chimera_rag.providers import llm

    with patch.object(llm, "_make_async_client", return_value=MagicMock()) as mk:
        llm.LLMProvider(_cfg(api_key_env="CUSTOM_KEY"))

    assert mk.call_args.kwargs["api_key"] == "sk-custom"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    from chimera_rag.core.exceptions import ProviderError
    from chimera_rag.providers import llm

    with patch.object(llm, "_make_async_client", return_value=MagicMock()):
        with pytest.raises(ProviderError, match="LLM_API_KEY"):
            llm.LLMProvider(_cfg())


def test_model_and_base_url_come_from_config(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-x")
    from chimera_rag.providers import llm

    with patch.object(llm, "_make_async_client", return_value=MagicMock()):
        provider = llm.LLMProvider(
            _cfg(model="my-model", api_key_env="LLM_API_KEY"),
        )

    assert provider.model == "my-model"
    assert provider.base_url is None
