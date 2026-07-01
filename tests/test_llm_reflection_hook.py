"""Tests for LLMReflectionHook and Self-Reflection integration.

Covers:
- LLMReflectionHook registration, interface conformance
- Numeric scoring (refusal, length, overlap)
- Verbal reflection generation (mock LLM)
- High-quality answers skip LLM call
- IntentReasoner.replan() fallback ladder
- IntentAgentRetriever multi-round reflection loop (mock)
- build_experience_entry with reflection_text
- Facade _wire_reflector wiring
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Registration & interface conformance
# ---------------------------------------------------------------------------
def test_llm_reflection_registers_under_slot():
    from chimera_rag.core.registry import get_registry
    from chimera_rag.plugins.intent_agent.hooks.llm_reflection_hook import LLMReflectionHook

    assert get_registry().get("reflector", "intent_agent.llm_reflection") is LLMReflectionHook


async def test_llm_reflection_implements_base_reflector():
    from chimera_rag.interfaces.reflector import BaseReflector
    from chimera_rag.plugins.intent_agent.hooks.llm_reflection_hook import LLMReflectionHook

    assert issubclass(LLMReflectionHook, BaseReflector)


# ---------------------------------------------------------------------------
# Numeric scoring (same signals as HeuristicReflectionHook)
# ---------------------------------------------------------------------------
async def test_llm_reflection_numeric_score_flags_refusal():
    from unittest.mock import MagicMock

    from chimera_rag.core.types import Answer, Query
    from chimera_rag.plugins.intent_agent.hooks.llm_reflection_hook import LLMReflectionHook

    mock_llm = MagicMock()
    hook = LLMReflectionHook(llm=mock_llm)

    fb = await hook.evaluate(Query(text="who?"), Answer(text="I do not know."))
    assert fb.quality < 0.5
    assert fb.reflection_text is None  # refusal detected before LLM call


async def test_llm_reflection_numeric_score_high_quality():
    from unittest.mock import MagicMock

    from chimera_rag.core.types import Answer, Query
    from chimera_rag.plugins.intent_agent.hooks.llm_reflection_hook import LLMReflectionHook

    mock_llm = MagicMock()
    hook = LLMReflectionHook(llm=mock_llm)

    fb = await hook.evaluate(
        Query(text="who invented dynamite?"),
        Answer(text="Alfred Nobel invented dynamite in 1867.", evidence_chunk_ids=["c0"]),
    )
    assert fb.quality >= 0.7
    assert fb.reflection_text is None  # high quality skips LLM call


# ---------------------------------------------------------------------------
# Verbal reflection via mock LLM
# ---------------------------------------------------------------------------
async def test_llm_reflection_generates_verbal_critique_for_low_quality():
    from unittest.mock import AsyncMock, MagicMock

    from chimera_rag.core.types import Answer, Query
    from chimera_rag.plugins.intent_agent.hooks.llm_reflection_hook import LLMReflectionHook

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="The answer is too short and lacks evidence about Nobel's invention.")
    hook = LLMReflectionHook(llm=mock_llm, min_length=50)

    # Short answer => low quality => triggers LLM call
    fb = await hook.evaluate(Query(text="who invented dynamite?"), Answer(text="Nobel."))
    assert fb.quality < 0.7
    assert fb.reflection_text is not None
    assert "Nobel" in fb.reflection_text
    assert fb.suggestion is not None


async def test_llm_reflection_handles_llm_failure_gracefully():
    from unittest.mock import AsyncMock, MagicMock

    from chimera_rag.core.types import Answer, Query
    from chimera_rag.plugins.intent_agent.hooks.llm_reflection_hook import LLMReflectionHook

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("API down"))
    hook = LLMReflectionHook(llm=mock_llm, min_length=50)

    fb = await hook.evaluate(Query(text="who invented dynamite?"), Answer(text="Nobel."))
    assert fb.quality < 0.7
    assert fb.reflection_text is None  # LLM call failed, no verbal critique
    # suggestion should still be None (no reflection_text to extract from)


# ---------------------------------------------------------------------------
# IntentReasoner.replan()
# ---------------------------------------------------------------------------
def test_replan_escalates_strategy_on_ladder():
    from chimera_rag.core.types import Intent, Query
    from chimera_rag.plugins.intent_agent.reasoning import IntentReasoner

    reasoner = IntentReasoner()
    intent = Intent(label="factual", confidence=0.9)
    query = Query(text="who invented dynamite?")
    original = reasoner.plan(query, intent)
    assert original.strategy_name == "direct"

    next_plan = reasoner.replan(original, reflection_text="direct search missed graph evidence")
    assert next_plan is not None
    assert next_plan.strategy_name == "multi_hop"
    assert next_plan.params["reflection_text"] == "direct search missed graph evidence"
    assert next_plan.params["previous_strategy"] == "direct"


def test_replan_returns_none_for_unknown_strategy():
    from chimera_rag.core.types import Intent, ReasoningPlan
    from chimera_rag.plugins.intent_agent.reasoning import IntentReasoner

    reasoner = IntentReasoner()
    # Create a plan with a strategy not on the ladder
    plan = ReasoningPlan(
        intent=Intent(label="factual", confidence=0.9),
        strategy_name="unknown_strategy",
        steps=["fallback"],
        params={},
    )
    result = reasoner.replan(plan)
    assert result is None


def test_replan_ladder_is_complete():
    """Every strategy in the lookup table has a fallback."""
    from chimera_rag.plugins.intent_agent.reasoning import (
        _INTENT_TO_STRATEGY,
        _STRATEGY_FALLBACK_LADDER,
    )

    for strategy_name in _INTENT_TO_STRATEGY.values():
        name = strategy_name[0] if isinstance(strategy_name, tuple) else strategy_name
        assert name in _STRATEGY_FALLBACK_LADDER, f"{name} missing from fallback ladder"


# ---------------------------------------------------------------------------
# build_experience_entry with reflection_text
# ---------------------------------------------------------------------------
def test_experience_entry_uses_reflection_text_as_lesson():
    from chimera_rag.core.types import Chunk, RetrievalResult
    from chimera_rag.plugins.intent_agent.langgraph.experience import build_experience_entry

    result = RetrievalResult(chunks=[Chunk(chunk_id="c0", doc_id="d", index=0, text="some text")])
    entry = build_experience_entry(
        intent="factual",
        query_text="who?",
        tool_sequence=["vector_search"],
        iterations=1,
        quality=0.3,
        result=result,
        quality_threshold=0.6,
        reflection_text="direct search missed key graph evidence",
    )
    assert entry.outcome == "lesson"
    assert entry.lesson == "direct search missed key graph evidence"


def test_experience_entry_falls_back_to_rule_template_without_reflection_text():
    from chimera_rag.core.types import Chunk, RetrievalResult
    from chimera_rag.plugins.intent_agent.langgraph.experience import build_experience_entry

    result = RetrievalResult(chunks=[Chunk(chunk_id="c0", doc_id="d", index=0, text="some text")])
    entry = build_experience_entry(
        intent="factual",
        query_text="who?",
        tool_sequence=["vector_search"],
        iterations=1,
        quality=0.3,
        result=result,
        quality_threshold=0.6,
        reflection_text=None,
    )
    assert entry.outcome == "lesson"
    assert "建议补充" in entry.lesson  # rule template fallback


# ---------------------------------------------------------------------------
# QualityFeedback reflection_text field
# ---------------------------------------------------------------------------
def test_quality_feedback_has_reflection_text_field():
    from chimera_rag.core.types import QualityFeedback

    fb = QualityFeedback(quality=0.5, reflection_text="missing evidence")
    assert fb.reflection_text == "missing evidence"

    fb2 = QualityFeedback(quality=0.8)
    assert fb2.reflection_text is None


# ---------------------------------------------------------------------------
# Facade _wire_reflector wiring
# ---------------------------------------------------------------------------
def test_wire_reflector_disabled_leaves_wiring_unchanged(mock_intent_only_yaml):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.config import load_config

    config = load_config(mock_intent_only_yaml)
    wiring: dict = {}
    call_params: dict = {}
    rag = ChimeraRAG.__new__(ChimeraRAG)
    rag.config = config
    rag._wire_reflector(wiring, call_params)
    assert "reflector" not in wiring


def test_wire_reflector_heuristic_type(mock_intent_only_yaml):
    from chimera_rag import ChimeraRAG
    from chimera_rag.core.config import load_config
    from chimera_rag.plugins.intent_agent.hooks.reflection_hook import HeuristicReflectionHook

    config = load_config(mock_intent_only_yaml)
    config.plugins.intent_agent.hooks["reflection"] = {
        "enabled": True,
        "type": "heuristic",
    }

    wiring: dict = {}
    call_params: dict = {}
    rag = ChimeraRAG.__new__(ChimeraRAG)
    rag.config = config
    rag.llm = None  # heuristic doesn't need LLM
    rag._wire_reflector(wiring, call_params)
    assert "reflector" in wiring
    assert isinstance(wiring["reflector"], HeuristicReflectionHook)


def test_wire_reflector_llm_reflection_type(mock_intent_only_yaml):
    from unittest.mock import MagicMock

    from chimera_rag import ChimeraRAG
    from chimera_rag.core.config import load_config
    from chimera_rag.plugins.intent_agent.hooks.llm_reflection_hook import LLMReflectionHook

    config = load_config(mock_intent_only_yaml)
    config.plugins.intent_agent.hooks["reflection"] = {
        "enabled": True,
        "type": "llm_reflection",
        "max_iterations": 2,
        "quality_threshold": 0.5,
    }

    wiring: dict = {}
    call_params: dict = {}
    rag = ChimeraRAG.__new__(ChimeraRAG)
    rag.config = config
    rag.llm = MagicMock()
    rag._wire_reflector(wiring, call_params)
    assert "reflector" in wiring
    assert isinstance(wiring["reflector"], LLMReflectionHook)
    assert call_params["max_reflection_iterations"] == 2
    assert call_params["reflection_quality_threshold"] == 0.5
