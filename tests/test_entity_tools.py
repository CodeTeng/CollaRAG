"""Tests for entity extraction, relation extraction, and entity linking tools."""
import json

import pytest

from chimera_rag.core.types import Triple
from chimera_rag.plugins.colla_rag.tools.entity_tools import (
    _parse_json_list,
    build_entity_tools,
)


# ---------------------------------------------------------------------------
# Fake doubles
# ---------------------------------------------------------------------------
class _FakeLLM:
    """LLM that returns predefined JSON responses keyed by prompt content."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self.calls: list[tuple[str, int]] = []

    async def complete(self, prompt: str, max_tokens: int = 200) -> str:
        self.calls.append((prompt, max_tokens))
        for key, value in self._responses.items():
            if key in prompt:
                return value
        # Default: return empty JSON object
        return "{}"


class _FakeGraphStore:
    """Graph store with known entities for linking tests."""

    def __init__(self, triples: list[Triple] | None = None):
        self._triples = triples or []

    def all_triples(self):
        return list(self._triples)


# ---------------------------------------------------------------------------
# _parse_json_list tests
# ---------------------------------------------------------------------------
def test_parse_json_list_standard():
    result = _parse_json_list('{"entities": ["Einstein", "Bohr", "quantum mechanics"]}')
    assert result == ["Einstein", "Bohr", "quantum mechanics"]


def test_parse_json_list_with_markdown_fence():
    result = _parse_json_list(
        '```json\n{"entities": ["Einstein", "Bohr"]}\n```'
    )
    assert result == ["Einstein", "Bohr"]


def test_parse_json_list_flat_list():
    result = _parse_json_list('["Einstein", "Bohr", "quantum"]')
    assert result == ["Einstein", "Bohr", "quantum"]


def test_parse_json_list_malformed_returns_empty():
    result = _parse_json_list("this is not json at all")
    assert result == []


def test_parse_json_list_empty_object():
    result = _parse_json_list("{}")
    assert result == []


# ---------------------------------------------------------------------------
# entity_extract tool tests
# ---------------------------------------------------------------------------
def test_entity_extract_returns_entities():
    llm = _FakeLLM({"entities": '{"entities": ["Einstein", "Bohr", "quantum mechanics"]}'})
    graph = _FakeGraphStore()
    tools = build_entity_tools(llm=llm, graph_store=graph)
    et_tool = next(t for t in tools if t.name == "entity_extract")

    result = et_tool.invoke({"query": "Compare Einstein and Bohr on quantum mechanics"})
    assert result["tool"] == "entity_extract"
    assert "Einstein" in result["entities"]
    assert "Bohr" in result["entities"]
    assert result["count"] >= 2


def test_entity_extract_deduplicates():
    llm = _FakeLLM({"entities": '{"entities": ["Einstein", "einstein", "EINSTEIN", "Bohr"]}'})
    graph = _FakeGraphStore()
    tools = build_entity_tools(llm=llm, graph_store=graph)
    et_tool = next(t for t in tools if t.name == "entity_extract")

    result = et_tool.invoke({"query": "test"})
    # Case-insensitive dedup: "Einstein", "einstein", "EINSTEIN" → 1
    assert result["count"] == 2


def test_entity_extract_llm_failure_graceful():
    class _FailingLLM:
        async def complete(self, prompt, max_tokens):
            raise RuntimeError("simulated LLM failure")

    tools = build_entity_tools(llm=_FailingLLM(), graph_store=_FakeGraphStore())
    et_tool = next(t for t in tools if t.name == "entity_extract")
    result = et_tool.invoke({"query": "test"})
    assert result["entities"] == []
    assert result["count"] == 0


# ---------------------------------------------------------------------------
# relation_extract tool tests
# ---------------------------------------------------------------------------
def test_relation_extract_returns_relations():
    llm = _FakeLLM({"relation": json.dumps({
        "relations": [
            {"subject": "Einstein", "predicate": "contributed to", "object": "quantum mechanics"},
            {"subject": "Bohr", "predicate": "developed", "object": "atomic model"},
        ],
    })})
    graph = _FakeGraphStore()
    tools = build_entity_tools(llm=llm, graph_store=graph)
    rt_tool = next(t for t in tools if t.name == "relation_extract")

    result = rt_tool.invoke({"query": "How did Einstein and Bohr contribute to quantum theory?"})
    assert result["tool"] == "relation_extract"
    assert result["count"] == 2
    assert result["relations"][0]["subject"] == "Einstein"
    assert result["relations"][1]["predicate"] == "developed"


def test_relation_extract_empty():
    llm = _FakeLLM({"relation": '{"relations": []}'})
    graph = _FakeGraphStore()
    tools = build_entity_tools(llm=llm, graph_store=graph)
    rt_tool = next(t for t in tools if t.name == "relation_extract")
    result = rt_tool.invoke({"query": "Hello"})
    assert result["relations"] == []
    assert result["count"] == 0


# ---------------------------------------------------------------------------
# entity_link tool tests
# ---------------------------------------------------------------------------
def test_entity_link_with_llm_match():
    llm = _FakeLLM({"link": '{"Einstein": "Albert Einstein", "Bohr": "Niels Bohr"}'})
    graph = _FakeGraphStore([
        Triple(subject="Albert Einstein", predicate="born_in", object="Ulm"),
        Triple(subject="Niels Bohr", predicate="developed", object="atomic model"),
    ])
    tools = build_entity_tools(llm=llm, graph_store=graph)
    el_tool = next(t for t in tools if t.name == "entity_link")

    result = el_tool.invoke({"entities": ["Einstein", "Bohr"]})
    assert result["tool"] == "entity_link"
    assert result["mapping"]["Einstein"] == "Albert Einstein"
    assert result["mapping"]["Bohr"] == "Niels Bohr"
    assert result["matched"] == 2


def test_entity_link_fallback_substring():
    """When LLM returns empty, fallback to substring matching."""
    llm = _FakeLLM({"link": "{}"})  # LLM returns empty mapping
    graph = _FakeGraphStore([
        Triple(subject="Albert Einstein", predicate="born_in", object="Ulm"),
    ])
    tools = build_entity_tools(llm=llm, graph_store=graph)
    el_tool = next(t for t in tools if t.name == "entity_link")

    result = el_tool.invoke({"entities": ["Einstein"]})
    # Fallback substring should match "Einstein" → "Albert Einstein"
    assert result["mapping"]["Einstein"] == "Albert Einstein"


def test_entity_link_fallback_exact_case_insensitive():
    llm = _FakeLLM({"link": "{}"})
    graph = _FakeGraphStore([
        Triple(subject="Niels Bohr", predicate="developed", object="atomic model"),
    ])
    tools = build_entity_tools(llm=llm, graph_store=graph)
    el_tool = next(t for t in tools if t.name == "entity_link")

    result = el_tool.invoke({"entities": ["niels bohr"]})
    assert result["mapping"]["niels bohr"] == "Niels Bohr"


def test_entity_link_empty_graph():
    llm = _FakeLLM()
    graph = _FakeGraphStore([])
    tools = build_entity_tools(llm=llm, graph_store=graph)
    el_tool = next(t for t in tools if t.name == "entity_link")

    result = el_tool.invoke({"entities": ["Einstein"]})
    assert result["matched"] == 0
    assert result["total"] == 1


def test_entity_link_unmatched_entity():
    llm = _FakeLLM({"link": '{"Einstein": "Albert Einstein", "UnknownX": null}'})
    graph = _FakeGraphStore([
        Triple(subject="Albert Einstein", predicate="born_in", object="Ulm"),
    ])
    tools = build_entity_tools(llm=llm, graph_store=graph)
    el_tool = next(t for t in tools if t.name == "entity_link")

    result = el_tool.invoke({"entities": ["Einstein", "UnknownX"]})
    assert result["mapping"]["Einstein"] == "Albert Einstein"
    assert result["mapping"]["UnknownX"] is None
    assert result["matched"] == 1


# ---------------------------------------------------------------------------
# Integration: all three tools exist
# ---------------------------------------------------------------------------
def test_build_entity_tools_returns_three():
    tools = build_entity_tools(llm=_FakeLLM(), graph_store=_FakeGraphStore())
    names = {t.name for t in tools}
    assert names == {"entity_extract", "relation_extract", "entity_link"}