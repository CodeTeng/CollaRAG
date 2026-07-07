"""Tests for :mod:`chimera_rag.plugins.adagraph.layered_extractor`.

AdaGraph's layered extractor is architected for four extraction passes:

* L1 (explicit)         — reuses SimpleLLMExtractor's prompt; triples have layer='L1'
* L2 (implicit)         — coreference / causal / temporal inferences; reserved
* L3 (schema-guided)    — seeded by a domain schema / entity types; reserved
* L4 (cross-sentence)   — entity-linking across distant sentences; reserved

Only L1 is implemented in this MVP round; the other three return empty
lists and log a single DEBUG line so the extractor remains usable and
future TDD cycles can fill them in without touching the interface.
"""

from __future__ import annotations


def _make_chunk(text: str = "Alfred Nobel invented dynamite."):
    from chimera_rag.core.types import Chunk

    return Chunk(chunk_id="c0", doc_id="d0", index=0, text=text)


def test_layered_extractor_implements_abc():
    from _fakes import FakeLLMProvider
    from chimera_rag.interfaces.extractor import BaseTripleExtractor
    from chimera_rag.plugins.adagraph.layered_extractor import LayeredTripleExtractor

    assert isinstance(
        LayeredTripleExtractor(llm=FakeLLMProvider()),
        BaseTripleExtractor,
    )


def test_layered_extractor_l1_returns_explicit_triples_with_layer_tag():
    from _fakes import FakeLLMProvider
    from chimera_rag.plugins.adagraph.layered_extractor import LayeredTripleExtractor

    mock = FakeLLMProvider(
        default_response='[{"subject": "Alfred Nobel", "predicate": "invented", "object": "dynamite"}]'
    )
    ext = LayeredTripleExtractor(llm=mock, layers_enabled=["L1"])
    triples = ext.extract(_make_chunk())
    assert len(triples) == 1
    assert triples[0].layer == "L1"


def test_layered_extractor_unimplemented_layers_return_empty():
    """L2 / L3 / L4 are reserved; enabling them must not crash, yields []."""
    from _fakes import FakeLLMProvider
    from chimera_rag.plugins.adagraph.layered_extractor import LayeredTripleExtractor

    mock = FakeLLMProvider(default_response="[]")
    ext = LayeredTripleExtractor(llm=mock, layers_enabled=["L2", "L3", "L4"])
    triples = ext.extract(_make_chunk())
    assert triples == []


def test_layered_extractor_default_layers_enabled_is_l1_only():
    from _fakes import FakeLLMProvider
    from chimera_rag.plugins.adagraph.layered_extractor import LayeredTripleExtractor

    ext = LayeredTripleExtractor(llm=FakeLLMProvider())
    assert ext.layers_enabled == ["L1"]


def test_layered_extractor_registers_under_slot():
    import chimera_rag.plugins.adagraph  # noqa: F401
    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.adagraph.layered_extractor import LayeredTripleExtractor

    assert (
        get_registry().get("extractor", "adagraph.layered")
        is LayeredTripleExtractor
    )
