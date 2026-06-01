"""Innovation #2b — LangGraph Agentic retrieval.

This sub-package upgrades IntentAgent's *Decision* stage from a static
``intent -> single strategy`` lookup (see :mod:`..agent`) into an
**LLM-driven dynamic orchestration**: an agent that autonomously sekalects
and composes multiple retrieval *tools* inside a LangGraph ``StateGraph``,
with an explicit reflection loop.

The whole sub-package depends on the optional ``agent`` extra
(``langgraph`` + ``langchain-core``, 1.x LTS). Importing it without those
installed raises a clear, actionable error rather than a bare
``ImportError`` deep in the call stack.

Public retriever slot registered here::

    @register("retriever", "intent_agent.langgraph")
    class LangGraphAgentRetriever(BaseRetriever): ...

Enable via ``query.retriever.active: intent_agent.langgraph`` in config.yaml.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import langchain_core  # noqa: F401
    import langgraph  # noqa: F401

    LANGGRAPH_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised in degradation test
    LANGGRAPH_AVAILABLE = False
    logger.warning(
        "intent_agent.langgraph requires the optional 'agent' extra "
        "(langgraph>=1.0, langchain-core>=1.0). Install with: "
        "`uv sync --extra agent`. Underlying import error: %s",
        exc,
    )


def require_langgraph() -> None:
    """Raise a clear error if the optional ``agent`` extra is missing."""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "The 'intent_agent.langgraph' retriever needs the optional "
            "'agent' extra. Install it with: `uv sync --extra agent` "
            "(brings in langgraph>=1.0 and langchain-core>=1.0)."
        )


def _register_slots() -> None:
    """Import the retriever module to fire its ``@register`` side-effect.

    Kept as a function (rather than a top-level import) so that importing a
    single leaf module such as ``...langgraph.tools`` does not eagerly pull
    in the whole sub-package and risk circular imports during construction.
    The retriever module transitively imports ``tools`` / ``llm_adapter`` /
    ``graph``; it is the only module carrying the ``@register`` decorator.
    """
    if not LANGGRAPH_AVAILABLE:
        return
    from chimera_rag.plugins.intent_agent.langgraph import retriever  # noqa: F401


# Fire @register side-effects on package import, but never let a partially
# developed sub-module break the whole import chain.
if LANGGRAPH_AVAILABLE:
    try:
        _register_slots()
    except ImportError:  # pragma: no cover - only during incremental dev
        logger.debug("langgraph retriever slot not yet importable", exc_info=True)

__all__ = ["LANGGRAPH_AVAILABLE", "require_langgraph"]
