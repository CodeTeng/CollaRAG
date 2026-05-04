"""Tests for the async ABCs under :mod:`chimera_rag.interfaces`."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_base_retrieval_strategy_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Intent, Query, ReasoningPlan, RetrievalResult
    from chimera_rag.interfaces.strategy import BaseRetrievalStrategy

    with pytest.raises(TypeError):
        BaseRetrievalStrategy()

    class Echo(BaseRetrievalStrategy):
        async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult:
            return RetrievalResult(metadata={"echo": query.text})

    plan = ReasoningPlan(intent=Intent(label="factual", confidence=1.0), strategy_name="x")
    result = await Echo().execute(Query(text="hi"), plan)
    assert result.metadata["echo"] == "hi"


async def test_base_reflector_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Answer, QualityFeedback, Query
    from chimera_rag.interfaces.reflector import BaseReflector

    with pytest.raises(TypeError):
        BaseReflector()

    class AlwaysPerfect(BaseReflector):
        async def evaluate(self, query: Query, answer: Answer) -> QualityFeedback:
            return QualityFeedback(quality=1.0)

    fb = await AlwaysPerfect().evaluate(Query(text="x"), Answer(text="y"))
    assert fb.quality == 1.0


async def test_base_reranker_is_abstract_and_subclass_works():
    from chimera_rag.core.types import Chunk, Query, RetrievalResult
    from chimera_rag.interfaces.reranker import BaseReranker

    with pytest.raises(TypeError):
        BaseReranker()

    class ReverseReranker(BaseReranker):
        async def rerank(self, query: Query, retrieval: RetrievalResult) -> RetrievalResult:
            return RetrievalResult(
                chunks=list(reversed(retrieval.chunks)),
                triples=retrieval.triples,
                scores=list(reversed(retrieval.scores)),
            )

    ch1 = Chunk(chunk_id="a", doc_id="d", index=0, text="A")
    ch2 = Chunk(chunk_id="b", doc_id="d", index=1, text="B")
    res = RetrievalResult(chunks=[ch1, ch2], scores=[0.9, 0.8])
    out = await ReverseReranker().rerank(Query(text="q"), res)
    assert out.chunks[0].chunk_id == "b"


async def test_base_llm_provider_is_abstract_and_subclass_works():
    from chimera_rag.interfaces.llm_provider import BaseLLMProvider

    with pytest.raises(TypeError):
        BaseLLMProvider()

    class StaticLLM(BaseLLMProvider):
        async def complete(self, prompt: str, **kwargs) -> str:
            return f"echo: {prompt}"

    out = await StaticLLM().complete("hi")
    assert out == "echo: hi"
