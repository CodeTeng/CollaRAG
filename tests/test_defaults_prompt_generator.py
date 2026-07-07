"""Tests for :mod:`chimera_rag.defaults.prompt_generator`."""

from __future__ import annotations


def test_prompt_generator_uses_retrieval_as_context_and_returns_answer():
    from _fakes import FakeLLMProvider
    from chimera_rag.core.types import Chunk, Query, RetrievalResult
    from chimera_rag.defaults.generator import PromptBasedGenerator

    mock = FakeLLMProvider(
        rules=[("invented", "Alfred Nobel invented dynamite.")],
        default_response="unknown",
    )
    g = PromptBasedGenerator(llm=mock)

    retrieval = RetrievalResult(
        chunks=[Chunk(chunk_id="c0", doc_id="d", index=0, text="Alfred Nobel invented dynamite in 1867.")],
    )
    ans = g.generate(Query(text="Who invented dynamite?"), retrieval)

    assert ans.text == "Alfred Nobel invented dynamite."
    # Evidence chunks should be propagated for traceability.
    assert ans.evidence_chunk_ids == ["c0"]


def test_prompt_generator_handles_empty_retrieval_gracefully():
    from _fakes import FakeLLMProvider
    from chimera_rag.core.types import Query, RetrievalResult
    from chimera_rag.defaults.generator import PromptBasedGenerator

    mock = FakeLLMProvider(default_response="I do not know.")
    g = PromptBasedGenerator(llm=mock)
    ans = g.generate(Query(text="?"), RetrievalResult())
    assert isinstance(ans.text, str) and len(ans.text) > 0
    assert ans.evidence_chunk_ids == []


def test_prompt_generator_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.generator import PromptBasedGenerator

    assert get_registry().get("generator", "defaults.prompt") is PromptBasedGenerator
