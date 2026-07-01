"""Reasoning: convert an :class:`Intent` into a :class:`ReasoningPlan`.

The mapping (from the design doc) is a simple lookup table — kept pure
so it's trivially testable and easy to override. When an intent has no
entry we fall back to ``direct`` as the safest default.

Self-Reflection (innovation #2d) adds :meth:`replan`, which takes a
failed ``ReasoningPlan`` plus a verbal ``reflection_text`` and picks a
*stronger* strategy via the fallback ladder. This is the mechanism
behind the static-route version's multi-round reflection loop.
"""

from __future__ import annotations

from chimera_rag.core.types import Intent, Query, ReasoningPlan

# Intent label -> (strategy name, human-readable steps)
_INTENT_TO_STRATEGY: dict[str, tuple[str, list[str]]] = {
    "factual": (
        "direct",
        [
            "Embed the question",
            "Fetch top-k chunks by vector similarity",
            "Generate the answer directly",
        ],
    ),
    "analytical": (
        "iterative",
        [
            "Decompose the question into sub-questions",
            "Iterative retrieval for each sub-question",
            "Synthesize a causal answer",
        ],
    ),
    "comparative": (
        "parallel",
        [
            "Identify entities A and B",
            "Retrieve evidence for each in parallel",
            "Contrast and produce a comparative answer",
        ],
    ),
    "multi_hop": (
        "multi_hop",
        [
            "Identify bridge entities",
            "BFS / multi-hop expansion over the graph",
            "Join evidence across hops",
        ],
    ),
    "exploratory": (
        "expansion",
        [
            "Vector search for seed chunks",
            "Expand by neighbouring entities on the graph",
            "Produce an open-ended overview",
        ],
    ),
    "follow_up": (
        "context_aware",
        [
            "Resolve anaphora against session memory",
            "Inject prior entities into the retrieval",
            "Generate the follow-up answer",
        ],
    ),
}

# Fallback ladder: when a strategy fails, try the next one.
# Each strategy escalates to a *stronger* (more evidence-gathering) one.
# This ladder is used by IntentReasoner.replan() for the Self-Reflection
# multi-round loop.  It deliberately does NOT include "direct" as a
# fallback target — if direct already failed, escalating to multi_hop
# is the right move (more evidence, not less).
_STRATEGY_FALLBACK_LADDER: dict[str, str] = {
    "direct": "multi_hop",
    "multi_hop": "iterative",
    "iterative": "parallel",
    "parallel": "expansion",
    "expansion": "context_aware",
    "context_aware": "multi_hop",  # wrap around for exhaustive coverage
}


class IntentReasoner:
    """Pure intent -> ReasoningPlan mapper with Self-Reflection replanning."""

    def plan(self, query: Query, intent: Intent) -> ReasoningPlan:
        strategy_name, steps = _INTENT_TO_STRATEGY.get(
            intent.label, ("direct", ["fallback: direct retrieval"])
        )

        params: dict = {}
        if query.session_id:
            params["session_id"] = query.session_id

        return ReasoningPlan(
            intent=intent,
            strategy_name=strategy_name,
            steps=list(steps),
            params=params,
        )

    def replan(
        self,
        failed_plan: ReasoningPlan,
        reflection_text: str | None = None,
    ) -> ReasoningPlan | None:
        """Return a stronger plan by escalating the strategy, or None if exhausted.

        This is the core of the Self-Reflection loop: after a round produces
        low-quality results, ``replan`` picks the next strategy on the fallback
        ladder. The ``reflection_text`` is injected into the new plan's params
        so downstream strategies can read it and adjust their behavior.

        Returns ``None`` when the current strategy has no fallback (shouldn't
        happen with the current ladder, but guards against infinite loops).
        """
        next_strategy = _STRATEGY_FALLBACK_LADDER.get(failed_plan.strategy_name)
        if next_strategy is None:
            return None

        # Build steps for the new strategy from the lookup table; if the
        # strategy isn't in the table (shouldn't happen), use a generic step.
        _, steps = _INTENT_TO_STRATEGY.get(
            next_strategy, (next_strategy, [f"fallback: {next_strategy} retrieval"])
        )

        params = dict(failed_plan.params)
        params["reflection_text"] = reflection_text or ""
        params["previous_strategy"] = failed_plan.strategy_name

        return ReasoningPlan(
            intent=failed_plan.intent,
            strategy_name=next_strategy,
            steps=list(steps),
            params=params,
        )


__all__ = ["IntentReasoner"]
