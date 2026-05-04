"""IntentAgent (innovation #2) — intent-aware retrieval agent.

Four execution stages (M-R-D-E):

1. **Memory**     — per-session conversation history + entity cache
2. **Reasoning**  — 6-way intent classification -> ReasoningPlan
3. **Decision**   — strategy selection from the 6-strategy library
4. **Execution**  — adaptive loop with reflection/reranking/parallel hooks

The package top-level imports all sub-modules to fire ``@register``
side effects, mirroring the ``defaults/`` and ``plugins/adagraph/`` layout.
"""

from chimera_rag.plugins.intent_agent import (  # noqa: F401
    hooks,
    intent_classifier,
    memory,
    reasoning,
    strategies,
)
