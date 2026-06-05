"""IntentAgent (innovation #2) — intent-aware retrieval agent.

Four execution stages (M-R-D-E):

1. **Memory**     — per-session conversation history + entity cache
2. **Reasoning**  — 6-way intent classification -> ReasoningPlan
3. **Decision**   — strategy selection from the 6-strategy library
4. **Execution**  — adaptive loop with reflection/reranking/parallel hooks

The package top-level imports all sub-modules to fire ``@register``
side effects, mirroring the ``defaults/`` and ``plugins/adagraph/`` layout.
"""

# Innovation #2b (LangGraph agentic retrieval) is gated behind the optional
# ``agent`` extra. Importing the sub-package is self-guarding: if langgraph /
# langchain-core are missing it logs a hint and skips registration rather than
# breaking the whole intent_agent plugin import.
from chimera_rag.plugins.intent_agent import (  # noqa: F401
    agent,
    experiential_memory,
    hooks,
    intent_classifier,
    langgraph,
    memory,
    reasoning,
    strategies,
)
