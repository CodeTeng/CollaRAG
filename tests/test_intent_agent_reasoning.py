"""Tests for :mod:`chimera_rag.plugins.intent_agent.reasoning`.

The IntentReasoner maps an :class:`Intent` to a :class:`ReasoningPlan`
that names the strategy to execute and any per-strategy parameters.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "intent_label,expected_strategy",
    [
        ("factual",     "direct"),
        ("analytical",  "iterative"),
        ("comparative", "parallel"),
        ("multi_hop",   "multi_hop"),
        ("exploratory", "expansion"),
        ("follow_up",   "context_aware"),
    ],
)
def test_reasoner_maps_each_intent_to_expected_strategy(
    intent_label: str, expected_strategy: str
):
    from chimera_rag.core.types import Intent, Query
    from chimera_rag.plugins.intent_agent.reasoning import IntentReasoner

    plan = IntentReasoner().plan(
        Query(text="?"),
        Intent(label=intent_label, confidence=1.0),
    )
    assert plan.strategy_name == expected_strategy
    assert plan.intent.label == intent_label


def test_reasoner_includes_steps_description_for_debug():
    from chimera_rag.core.types import Intent, Query
    from chimera_rag.plugins.intent_agent.reasoning import IntentReasoner

    plan = IntentReasoner().plan(
        Query(text="why?"), Intent(label="analytical", confidence=1.0)
    )
    assert plan.steps  # non-empty list of human-readable steps
    assert any("iterative" in s.lower() or "decompose" in s.lower() for s in plan.steps)


def test_reasoner_preserves_params_hints_from_intent():
    """Custom params in Intent.rationale etc. flow into the plan for tracing."""
    from chimera_rag.core.types import Intent, Query
    from chimera_rag.plugins.intent_agent.reasoning import IntentReasoner

    plan = IntentReasoner().plan(
        Query(text="?", session_id="s-1"),
        Intent(label="follow_up", confidence=0.8),
    )
    assert plan.params.get("session_id") == "s-1"
