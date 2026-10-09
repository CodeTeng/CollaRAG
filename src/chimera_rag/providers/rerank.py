"""OpenAI-compatible rerank (cross-encoder) provider.

Calls an OpenAI-compatible ``/rerank`` endpoint (the de-facto contract used
by vLLM, Xinference, SiliconFlow, Jina, and other inference servers).

All model / connection details live in the config file; only the API key
comes from the environment (default ``LLM_API_KEY``).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
)

from chimera_rag.core.exceptions import ProviderError
from chimera_rag.interfaces.rerank_provider import BaseRerankProvider

if TYPE_CHECKING:
    from chimera_rag.core.config import RerankConfig
    from chimera_rag.core.types import Query, RetrievalResult

logger = logging.getLogger(__name__)


class RerankProvider(BaseRerankProvider):
    """Cross-encoder reranker via an OpenAI-compatible ``/rerank`` endpoint.

    All model/connection details come from the config object; the API key
    is resolved from the env var named by ``cfg.api_key_env`` (defaults to
    ``LLM_API_KEY``).
    """

    def __init__(self, cfg: RerankConfig) -> None:
        self.cfg = cfg
        self.top_n = cfg.top_n

        api_key = os.getenv(cfg.api_key_env)
        if not api_key:
            raise ProviderError(
                f"API key not found; set {cfg.api_key_env} in .env."
            )
        self._api_key = api_key
        self._base_url = cfg.base_url.rstrip("/") if cfg.base_url else None
        self._timeout = cfg.timeout

    # ------------------------------------------------------------------
    async def rerank(
        self,
        query: Query,
        result: RetrievalResult,
        top_n: int | None = None,
    ) -> RetrievalResult:
        import httpx

        from chimera_rag.core.types import RetrievalResult

        if not result.chunks:
            return result

        top_n = top_n or self.top_n
        texts = [c.text for c in result.chunks]

        payload = {
            "model": self.cfg.model,
            "query": query.text,
            "documents": texts,
            "top_n": min(top_n, len(texts)),
        }

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
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        headers = {
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        }
                        r = await client.post(
                            f"{self._base_url}/rerank",
                            json=payload,
                            headers=headers,
                        )
                        r.raise_for_status()
                        data = r.json()
        except RetryError as re:
            last_error = re.last_attempt.exception() if re.last_attempt else re
            logger.warning("Rerank call exhausted retries: %s", last_error)
            raise ProviderError(
                f"Rerank call failed after {retry_cfg.max_attempts} attempts: {last_error}"
            ) from last_error

        # Parse the de-facto standard response shape:
        # {"results": [{"index": int, "relevance_score": float}, ...]}
        scored = data.get("results", [])
        if not scored:
            return result

        # Reorder chunks by relevance score descending.
        idx_map = {s["index"]: s["relevance_score"] for s in scored}
        paired = [
            (idx_map[i], result.chunks[i])
            for i in range(len(result.chunks))
            if i in idx_map
        ]
        paired.sort(key=lambda x: x[0], reverse=True)

        reranked_chunks = [c for _, c in paired[:top_n]]
        reranked_scores = [s for s, _ in paired[:top_n]]

        return RetrievalResult(
            chunks=reranked_chunks,
            scores=reranked_scores,
            metadata={
                **result.metadata,
                "reranked": True,
                "rerank_model": self.cfg.model,
            },
        )


__all__ = ["RerankProvider"]
