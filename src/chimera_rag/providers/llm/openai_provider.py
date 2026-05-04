"""OpenAI-compatible LLM provider (serves OpenAI + DeepSeek).

DeepSeek Chat API is OpenAI-protocol-compatible: same request/response
shape, just a different ``base_url`` and a DeepSeek-issued API key. We
therefore share a single implementation and let config choose
``provider: deepseek`` vs ``provider: openai``.

Key env lookups (resolved at construction time):

* ``cfg.api_key_env``   -> the env var holding the API key (required)
* ``cfg.base_url_env``  -> the env var holding base_url (optional)

Retry policy uses ``tenacity`` with exponential backoff bounded by
``cfg.retry``. Transient errors (``RuntimeError``, ``TimeoutError``,
provider-specific exceptions when the openai SDK is importable) trigger
a retry; otherwise the error is wrapped in :class:`ProviderError`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
)

from chimera_rag.core.config import LLMConfig
from chimera_rag.core.exceptions import ProviderError
from chimera_rag.interfaces.llm_provider import BaseLLMProvider

logger = logging.getLogger(__name__)


def _default_api_key_env(provider: str) -> str:
    return {
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "OPENAI_API_KEY")


def _default_base_url_env(provider: str) -> str | None:
    return {
        "deepseek": "DEEPSEEK_BASE_URL",
        "openai": "OPENAI_BASE_URL",
    }.get(provider)


def _default_base_url(provider: str) -> str | None:
    return {
        "deepseek": "https://api.deepseek.com",
        "openai": None,  # SDK default
    }.get(provider)


def _make_async_client(*, api_key: str, base_url: str | None, timeout: float):
    """Isolated factory so tests can monkeypatch without touching the openai SDK."""
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError as e:  # pragma: no cover - optional dep
        raise ProviderError(
            f"openai SDK not installed; run `uv sync --extra llm` ({e})"
        ) from e

    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI / DeepSeek async chat-completion client."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

        provider = cfg.provider  # "openai" | "deepseek"
        api_key_env = cfg.api_key_env or _default_api_key_env(provider)
        base_url_env = cfg.base_url_env or _default_base_url_env(provider)

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ProviderError(
                f"API key not found; expected env var {api_key_env!r} to be set "
                f"(provider={provider!r})."
            )

        base_url = os.getenv(base_url_env) if base_url_env else None
        if base_url is None:
            base_url = _default_base_url(provider)

        self._client = _make_async_client(
            api_key=api_key,
            base_url=base_url,
            timeout=cfg.timeout,
        )

    # ------------------------------------------------------------------
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        retry_cfg = self.cfg.retry
        last_error: BaseException | None = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(retry_cfg.max_attempts),
                wait=wait_exponential(
                    multiplier=retry_cfg.initial_wait,
                    max=retry_cfg.max_wait,
                ),
                reraise=False,
            ):
                with attempt:
                    resp = await self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=kwargs.get("temperature", self.temperature),
                        max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    )
                    return resp.choices[0].message.content or ""
        except RetryError as re:
            last_error = re.last_attempt.exception() if re.last_attempt else re
            logger.warning("OpenAI/DeepSeek LLM call exhausted retries: %s", last_error)
            raise ProviderError(
                f"LLM call failed after {retry_cfg.max_attempts} attempts: {last_error}"
            ) from last_error
        # Should not reach here, but guard anyway.
        raise ProviderError("LLM call failed with no response")  # pragma: no cover


__all__ = ["OpenAILLMProvider"]
