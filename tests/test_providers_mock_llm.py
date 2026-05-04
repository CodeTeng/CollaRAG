"""Tests for :mod:`chimera_rag.providers.llm.mock_provider`.

The mock LLM is the backbone of our offline test suite: it must be
deterministic (same prompt -> same response), reasonably steerable
(keyword-based response rules), and implement BaseLLMProvider.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


async def test_mock_llm_is_instance_of_base_provider():
    from chimera_rag.interfaces.llm_provider import BaseLLMProvider
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    assert isinstance(MockLLMProvider(), BaseLLMProvider)


async def test_mock_llm_default_response_is_deterministic():
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    m = MockLLMProvider()
    r1 = await m.complete("hello")
    r2 = await m.complete("hello")
    assert r1 == r2  # deterministic for the same prompt
    assert isinstance(r1, str) and len(r1) > 0


async def test_mock_llm_rules_match_first_keyword_found():
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    m = MockLLMProvider(
        rules=[
            ("triple", '{"triples": []}'),
            ("intent", '{"label": "factual", "confidence": 0.9}'),
        ]
    )
    assert await m.complete("extract triples from: ...") == '{"triples": []}'
    assert await m.complete("classify intent of this") == '{"label": "factual", "confidence": 0.9}'


async def test_mock_llm_falls_back_to_default_when_no_rule_matches():
    from chimera_rag.providers.llm.mock_provider import MockLLMProvider

    m = MockLLMProvider(default_response="DEFAULT")
    assert await m.complete("totally unrelated text") == "DEFAULT"
