"""Vanilla "bare KG-RAG" implementations.

This package is *auto-registered on import* — loading
:mod:`chimera_rag.defaults` triggers ``@register`` side effects for all 9
default implementations so the :class:`Registry` is populated before any
pipeline is built.
"""

# Eagerly import submodules to fire the @register decorators.
from chimera_rag.defaults import fixed_chunker  # noqa: F401
from chimera_rag.defaults import llm_extractor  # noqa: F401
