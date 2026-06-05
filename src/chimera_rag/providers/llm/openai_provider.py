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


def _default_model(provider: str) -> str | None:
    return {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
    }.get(provider)


def _make_async_client(*, api_key: str, base_url: str | None, timeout: float):
    """Isolated factory so tests can monkeypatch without touching the openai SDK."""
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError as e:  # pragma: no cover - optional dep
        raise ProviderError(
            f"openai SDK not installed; run `uv sync --extra llm` ({e})"
        ) from e

    # Disable the OpenAI SDK's built-in retry layer (default: 2). Retries are
    # handled exclusively by this provider's tenacity loop in .complete(), so
    # leaving the SDK layer on would stack two independent backoff mechanisms
    # (SDK 2x * tenacity Nx), amplifying request volume against rate-limited
    # endpoints and producing the confusing "Retrying request" INFO logs.
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI / DeepSeek async chat-completion client.

    Connection detail resolution order (highest priority first):

    1. Unified env vars — ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL``
    2. Legacy per-provider env vars from ``cfg.api_key_env`` / ``cfg.base_url_env``
    3. ``cfg.model`` (from yaml)
    4. Provider defaults (e.g. ``https://api.deepseek.com`` for deepseek)
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        # Qwen3 (and similar) emit a long thinking block by default, which eats
        # the output budget, inflates latency, and can corrupt structured
        # outputs (intent JSON). The OpenAI-compatible `/v1` endpoint that
        # ollama exposes does NOT honour `/no_think` system prompts nor a
        # `think` param — the model keeps reasoning into a separate field and
        # leaves `content` empty/truncated. The only reliable switch is
        # ollama's NATIVE `/api/chat` endpoint with `think: false`. So when
        # disable_thinking is on AND we're talking to ollama, we bypass the
        # OpenAI SDK and call the native endpoint directly (see _complete_ollama_no_think).
        self.disable_thinking = bool(getattr(cfg, "disable_thinking", False))

        provider = cfg.provider  # "openai" | "deepseek"

        # ---- API key ------------------------------------------------
        # 1) unified LLM_API_KEY wins
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            api_key_env = cfg.api_key_env or _default_api_key_env(provider)
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ProviderError(
                    f"API key not found; set LLM_API_KEY in .env "
                    f"(or legacy {api_key_env!r})."
                )

        # ---- Base URL -----------------------------------------------
        base_url = os.getenv("LLM_BASE_URL")
        if not base_url:
            base_url_env = cfg.base_url_env or _default_base_url_env(provider)
            base_url = os.getenv(base_url_env) if base_url_env else None
        if not base_url:
            base_url = _default_base_url(provider)
        self.base_url = base_url

        # Detect an ollama endpoint so disable_thinking can route through the
        # native /api/chat (which is the only place `think: false` works).
        # ollama's OpenAI-compat URL looks like http://host:11434/v1; the
        # native chat URL is http://host:11434/api/chat.
        self._ollama_native_url: str | None = None
        if base_url and ("11434" in base_url or base_url.rstrip("/").endswith("/v1")):
            root = base_url.rstrip("/")
            if root.endswith("/v1"):
                root = root[: -len("/v1")]
            self._ollama_native_url = root + "/api/chat"

        # ---- Model --------------------------------------------------
        model = os.getenv("LLM_MODEL") or cfg.model or _default_model(provider)
        if not model:
            raise ProviderError(
                "model not specified; set LLM_MODEL in .env, "
                "declare llm.model in config.yaml, or use a provider with a "
                f"built-in default (got provider={provider!r})."
            )
        self.model = model

        self._client = _make_async_client(
            api_key=api_key,
            base_url=base_url,
            timeout=cfg.timeout,
        )

        # Token usage accumulator — shared across all .complete() calls on
        # this provider instance. Read via .usage / reset via .reset_usage().
        self.usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    # ------------------------------------------------------------------
    def reset_usage(self) -> None:
        for k in self.usage:
            self.usage[k] = 0

    # ------------------------------------------------------------------
    async def _complete_ollama_no_think(self, prompt: str, **kwargs: Any) -> str:
        """Call ollama's native /api/chat with think:false.

        This is the only reliable way to suppress qwen3-style reasoning when
        served via ollama; the OpenAI-compat /v1 layer ignores both
        `/no_think` prompts and a `think` param. We map our config to ollama's
        ``options`` (num_predict == max_tokens) and accumulate usage from the
        native counters (prompt_eval_count / eval_count).
        """
        import httpx

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            },
        }
        async with httpx.AsyncClient(timeout=self.cfg.timeout) as client:
            r = await client.post(self._ollama_native_url, json=payload)
            r.raise_for_status()
            data = r.json()

        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens
        self.usage["calls"] += 1

        return (data.get("message", {}) or {}).get("content", "") or ""

    # ------------------------------------------------------------------
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        retry_cfg = self.cfg.retry
        last_error: BaseException | None = None
        # When thinking is disabled and we're on ollama, the native /api/chat
        # endpoint with think:false is the only path that actually suppresses
        # the reasoning block. Route there instead of the OpenAI SDK.
        use_native = self.disable_thinking and self._ollama_native_url is not None
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
                    if use_native:
                        return await self._complete_ollama_no_think(prompt, **kwargs)
                    resp = await self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=kwargs.get("temperature", self.temperature),
                        max_tokens=kwargs.get("max_tokens", self.max_tokens),
                    )
                    # Accumulate usage if the SDK populated it
                    u = getattr(resp, "usage", None)
                    if u is not None:
                        self.usage["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                        self.usage["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
                        self.usage["total_tokens"] += int(getattr(u, "total_tokens", 0) or 0)
                    self.usage["calls"] += 1
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
