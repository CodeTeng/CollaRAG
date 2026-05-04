"""Vanilla "bare KG-RAG" implementations.

This package is *auto-registered on import* — loading
:mod:`chimera_rag.defaults` triggers ``@register`` side effects for all
default implementations so the :class:`Registry` is populated before any
pipeline is built.
"""

# Eagerly import submodules to fire the @register decorators.
from chimera_rag.defaults import (  # noqa: F401
    fixed_chunker,
    llm_extractor,
    no_memory,
    no_reflection,
    no_reranker,
    noop_pruner,
    rule_intent_classifier,
)
