"""Tests for :mod:`chimera_rag.defaults.llm_extractor`."""

from __future__ import annotations

# extract() is sync, but the LLM underneath is async. We run tests as plain
# sync; the extractor handles running async internally.


def _make_chunk(text: str = "Alfred Nobel invented dynamite in 1867."):
    from chimera_rag.core.types import Chunk

    return Chunk(chunk_id="c1", doc_id="d1", index=0, text=text)


def test_simple_llm_extractor_parses_json_triples_from_mock_llm():
    from _fakes import FakeLLMProvider
    from chimera_rag.defaults.extractor import SimpleLLMExtractor

    json_response = (
        '[{"subject": "Alfred Nobel", "predicate": "invented", "object": "dynamite"},'
        ' {"subject": "dynamite", "predicate": "invented_in", "object": "1867"}]'
    )
    mock = FakeLLMProvider(default_response=json_response)
    ext = SimpleLLMExtractor(llm=mock)

    triples = ext.extract(_make_chunk())
    assert len(triples) == 2
    assert triples[0].subject == "Alfred Nobel"
    assert triples[0].predicate == "invented"
    assert triples[0].object == "dynamite"
    # Every triple must be linked back to the source chunk.
    assert all(t.source_chunk_id == "c1" for t in triples)
    # L1 is the default layer label for explicit extraction.
    assert all(t.layer == "L1" for t in triples)


def test_simple_llm_extractor_handles_malformed_llm_output():
    """If the LLM returns garbage, we must not crash the ingest pipeline."""
    from _fakes import FakeLLMProvider
    from chimera_rag.defaults.extractor import SimpleLLMExtractor

    mock = FakeLLMProvider(default_response="absolutely not JSON")
    ext = SimpleLLMExtractor(llm=mock)
    triples = ext.extract(_make_chunk())
    assert triples == []


def test_simple_llm_extractor_tolerates_fenced_json_blocks():
    """LLMs frequently wrap JSON in ```json ... ``` — strip and parse."""
    from _fakes import FakeLLMProvider
    from chimera_rag.defaults.extractor import SimpleLLMExtractor

    fenced = (
        "Sure, here are the triples:\n"
        '```json\n'
        '[{"subject": "a", "predicate": "p", "object": "b"}]\n'
        '```'
    )
    mock = FakeLLMProvider(default_response=fenced)
    ext = SimpleLLMExtractor(llm=mock)
    triples = ext.extract(_make_chunk())
    assert len(triples) == 1
    assert triples[0].subject == "a"


def test_simple_llm_extractor_respects_max_triples_per_chunk():
    from _fakes import FakeLLMProvider
    from chimera_rag.defaults.extractor import SimpleLLMExtractor

    # LLM returns 10 triples.
    json_response = "[" + ",".join(
        f'{{"subject": "s{i}", "predicate": "p", "object": "o{i}"}}' for i in range(10)
    ) + "]"
    mock = FakeLLMProvider(default_response=json_response)
    ext = SimpleLLMExtractor(llm=mock, max_triples_per_chunk=3)
    triples = ext.extract(_make_chunk())
    assert len(triples) == 3


def test_simple_llm_extractor_registers_under_slot():
    import chimera_rag.defaults  # noqa: F401 - trigger registration
    from chimera_rag.core.registry import get_registry
    from chimera_rag.defaults.extractor import SimpleLLMExtractor

    assert get_registry().get("extractor", "defaults.simple_llm") is SimpleLLMExtractor
