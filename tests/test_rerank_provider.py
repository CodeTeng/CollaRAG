"""Tests for RerankProvider (OpenAI-compatible rerank client)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from chimera_rag.core.config import RerankConfig, RetryConfig
from chimera_rag.core.types import Chunk, Query, RetrievalResult


def _build_cfg(**overrides) -> RerankConfig:
    defaults = dict(
        enabled=True,
        model="bge-reranker-v2-m3",
        base_url="http://localhost:9997/v1",
        api_key_env="TEST_API_KEY",
        top_n=3,
        timeout=15.0,
        retry=RetryConfig(max_attempts=2, initial_wait=0.01, max_wait=0.05),
    )
    defaults.update(overrides)
    return RerankConfig(**defaults)


def _fake_chunks(n: int = 5) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", text=f"document text {i}", doc_id=f"doc{i}", index=i)
        for i in range(n)
    ]


def _fake_result(chunks: list[Chunk] | None = None) -> RetrievalResult:
    if chunks is None:
        chunks = _fake_chunks()
    return RetrievalResult(chunks=chunks, scores=[1.0] * len(chunks))


def _fake_rerank_response(indices: list[int], scores: list[float]):
    """Emulate the de-facto /rerank response shape."""
    return {
        "results": [
            {"index": i, "relevance_score": s}
            for i, s in zip(indices, scores)
        ]
    }


@pytest.mark.asyncio
async def test_rerank_reorders_by_score(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers.rerank import RerankProvider

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.json = lambda: _fake_rerank_response(
            [2, 0, 4, 3, 1], [0.95, 0.80, 0.60, 0.40, 0.20]
        )

        provider = RerankProvider(_build_cfg(top_n=3))
        query = Query(text="test query")
        result = _fake_result()
        reranked = await provider.rerank(query, result)

    assert len(reranked.chunks) == 3
    assert reranked.chunks[0].chunk_id == "c2"  # highest score
    assert reranked.chunks[1].chunk_id == "c0"
    assert reranked.chunks[2].chunk_id == "c4"
    assert reranked.scores == [0.95, 0.80, 0.60]


@pytest.mark.asyncio
async def test_rerank_empty_result_returns_as_is(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers.rerank import RerankProvider

    provider = RerankProvider(_build_cfg())
    query = Query(text="test")
    result = RetrievalResult(chunks=[], scores=[])
    reranked = await provider.rerank(query, result)

    assert reranked.chunks == []
    assert reranked.scores == []


@pytest.mark.asyncio
async def test_rerank_retries_on_failure(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.providers.rerank import RerankProvider

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock()
        mock_post.return_value.raise_for_status = lambda: None
        # First call errors, second succeeds
        mock_post.side_effect = [
            RuntimeError("transient"),
            _make_mock_response(_fake_rerank_response([0], [1.0])),
        ]

        provider = RerankProvider(_build_cfg(top_n=1))
        query = Query(text="test")
        result = _fake_result()
        reranked = await provider.rerank(query, result)

    assert len(reranked.chunks) == 1


def _make_mock_response(data):
    m = AsyncMock()
    m.raise_for_status = lambda: None
    m.json = lambda: data
    return m


@pytest.mark.asyncio
async def test_rerank_raises_when_all_attempts_fail(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "sk-fake")

    from chimera_rag.core.exceptions import ProviderError
    from chimera_rag.providers.rerank import RerankProvider

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.side_effect = RuntimeError("boom")

        provider = RerankProvider(_build_cfg())
        query = Query(text="test")
        result = _fake_result()
        with pytest.raises(ProviderError, match="boom|failed"):
            await provider.rerank(query, result)


def test_rerank_provider_is_instance_of_base_provider():
    from chimera_rag.interfaces.rerank_provider import BaseRerankProvider
    from chimera_rag.providers.rerank import RerankProvider

    assert issubclass(RerankProvider, BaseRerankProvider)


def test_rerank_provider_missing_api_key(monkeypatch):
    monkeypatch.delenv("TEST_API_KEY", raising=False)

    from chimera_rag.core.exceptions import ProviderError
    from chimera_rag.providers.rerank import RerankProvider

    with pytest.raises(ProviderError, match="TEST_API_KEY"):
        RerankProvider(_build_cfg())
