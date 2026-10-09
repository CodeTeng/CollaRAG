"""OpenAI-compatible LLM provider.

Single implementation for any OpenAI-protocol-compatible chat-completion
endpoint (OpenAI, DeepSeek, Ollama, vLLM, Xinference, SiliconFlow, etc.).

All model / connection details live in the config file; only the API key
comes from the environment (default ``LLM_API_KEY``).

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


def _make_async_client(*, api_key: str, base_url: str | None, timeout: float):
    """Isolated factory so tests can monkeypatch without touching the openai SDK."""
    try:
        from openai import AsyncOpenAI  # type: ignore
    except ImportError as e:  # pragma: no cover - optional dep
        raise ProviderError(
            f"openai SDK not installed; run `uv sync --extra llm` ({e})"
        ) from e

    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


class LLMProvider(BaseLLMProvider):
    """OpenAI / DeepSeek / Ollama async chat-completion client.

    All model/connection details come from the config object; the API key
    is resolved from the env var named by ``cfg.api_key_env`` (defaults to
    ``LLM_API_KEY``).
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.disable_thinking = bool(cfg.disable_thinking)

        # ---- API key ------------------------------------------------
        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise ProviderError(
                f"API key not found; set {cfg.api_key_env} in .env."
            )

        # ---- Base URL -----------------------------------------------
        self.base_url = cfg.base_url

        # Detect an ollama endpoint so disable_thinking can route through the
        # native /api/chat (which is the only place `think: false` works).
        self._ollama_native_url: str | None = None
        if cfg.base_url and ("11434" in cfg.base_url or cfg.base_url.rstrip("/").endswith("/v1")):
            root = cfg.base_url.rstrip("/")
            if root.endswith("/v1"):
                root = root[: -len("/v1")]
            self._ollama_native_url = root + "/api/chat"

        # ---- Model --------------------------------------------------
        self.model = cfg.model

        self._client = _make_async_client(
            api_key=api_key,
            base_url=cfg.base_url,
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
                    u = getattr(resp, "usage", None)
                    if u is not None:
                        self.usage["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                        self.usage["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
                        self.usage["total_tokens"] += int(getattr(u, "total_tokens", 0) or 0)
                    self.usage["calls"] += 1
                    return resp.choices[0].message.content or ""
        except RetryError as re:
            last_error = re.last_attempt.exception() if re.last_attempt else re
            logger.warning("LLM call exhausted retries: %s", last_error)
            raise ProviderError(
                f"LLM call failed after {retry_cfg.max_attempts} attempts: {last_error}"
            ) from last_error
        raise ProviderError("LLM call failed with no response")  # pragma: no cover


__all__ = ["LLMProvider"]
