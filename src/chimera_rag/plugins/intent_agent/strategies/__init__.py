"""Retrieval strategies available to IntentAgent.

Each strategy implements :class:`BaseRetrievalStrategy` and is registered
under ``("strategy", "intent_agent.<name>")``. They share a common
constructor signature: ``(retriever: BaseRetriever, ...)`` so the
StrategySelector can instantiate them uniformly.
"""

# Trigger @register side-effects on import.
from chimera_rag.plugins.intent_agent.strategies import (  # noqa: F401
    context_aware,
    direct,
    expansion,
    iterative,
    multi_hop,
    parallel,
)
