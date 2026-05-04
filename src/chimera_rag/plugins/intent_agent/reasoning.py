"""Reasoning: convert an :class:`Intent` into a :class:`ReasoningPlan`.

The mapping (from the design doc) is a simple lookup table — kept pure
so it's trivially testable and easy to override. When an intent has no
entry we fall back to ``direct`` as the safest default.
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


class IntentReasoner:
    """Pure intent -> ReasoningPlan mapper."""

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


__all__ = ["IntentReasoner"]
