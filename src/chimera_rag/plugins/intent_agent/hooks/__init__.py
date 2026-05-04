"""Pluggable hooks attached to IntentAgent's Execution loop."""

from chimera_rag.plugins.intent_agent.hooks import (  # noqa: F401
    parallel_pool_hook,
    reflection_hook,
    reranking_hook,
)
