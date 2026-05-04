"""Tests for the 3 pluggable hooks under :mod:`chimera_rag.plugins.intent_agent.hooks`.

Hooks extend IntentAgent's Execution loop:

* ReflectionHook      — evaluates answer quality (satisfies BaseReflector)
* RerankingHook       — cross-encoder-like reranker (satisfies BaseReranker)
* ParallelPoolHook    — asyncio worker pool for fan-out retrievals

MVP implementations are deliberately light but non-trivial; future TDD
cycles can swap in LLM-backed reflection or a real cross-encoder without
changing the hook interfaces.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# ReflectionHook
# ---------------------------------------------------------------------------
async def test_reflection_hook_implements_base_reflector():
    from chimera_rag.interfaces.reflector import BaseReflector
    from chimera_rag.plugins.intent_agent.hooks.reflection_hook import (
        HeuristicReflectionHook,
    )

    assert isinstance(HeuristicReflectionHook(), BaseReflector)


async def test_reflection_hook_flags_low_quality_for_unknown_answer():
    from chimera_rag.core.types import Answer, Query
    from chimera_rag.plugins.intent_agent.hooks.reflection_hook import (
        HeuristicReflectionHook,
    )

    fb = await HeuristicReflectionHook().evaluate(
        Query(text="who?"), Answer(text="I do not know.")
    )
    assert fb.quality < 0.5


async def test_reflection_hook_reports_high_quality_for_confident_answer():
    from chimera_rag.core.types import Answer, Query
    from chimera_rag.plugins.intent_agent.hooks.reflection_hook import (
        HeuristicReflectionHook,
    )

    fb = await HeuristicReflectionHook().evaluate(
        Query(text="who invented dynamite?"),
        Answer(text="Alfred Nobel invented dynamite in 1867.", evidence_chunk_ids=["c0"]),
    )
    assert fb.quality >= 0.7


# ---------------------------------------------------------------------------
# RerankingHook
# ---------------------------------------------------------------------------
async def test_reranking_hook_implements_base_reranker():
    from chimera_rag.interfaces.reranker import BaseReranker
    from chimera_rag.plugins.intent_agent.hooks.reranking_hook import (
        TokenOverlapRerankerHook,
    )

    assert isinstance(TokenOverlapRerankerHook(), BaseReranker)


async def test_reranking_hook_reorders_by_token_overlap():
    from chimera_rag.core.types import Chunk, Query, RetrievalResult
    from chimera_rag.plugins.intent_agent.hooks.reranking_hook import (
        TokenOverlapRerankerHook,
    )

    chunks = [
        Chunk(chunk_id="a", doc_id="d", index=0, text="completely unrelated"),
        Chunk(chunk_id="b", doc_id="d", index=1, text="Alfred Nobel invented dynamite"),
    ]
    res = RetrievalResult(chunks=chunks, scores=[0.9, 0.1])  # wrong order
    reranked = await TokenOverlapRerankerHook().rerank(
        Query(text="nobel dynamite"), res
    )
    # 'b' should now come first.
    assert reranked.chunks[0].chunk_id == "b"


# ---------------------------------------------------------------------------
# ParallelPoolHook
# ---------------------------------------------------------------------------
async def test_parallel_pool_hook_runs_coroutines_with_max_workers_limit():
    from chimera_rag.plugins.intent_agent.hooks.parallel_pool_hook import (
        ParallelRetrievalPool,
    )

    async def work(x: int) -> int:
        return x * 2

    pool = ParallelRetrievalPool(max_workers=2)
    results = await pool.run([work(1), work(2), work(3), work(4)])
    assert sorted(results) == [2, 4, 6, 8]


async def test_parallel_pool_hook_preserves_order_of_input():
    from chimera_rag.plugins.intent_agent.hooks.parallel_pool_hook import (
        ParallelRetrievalPool,
    )

    async def echo(x: str) -> str:
        return x

    pool = ParallelRetrievalPool(max_workers=2)
    out = await pool.run([echo("a"), echo("b"), echo("c")])
    assert out == ["a", "b", "c"]
