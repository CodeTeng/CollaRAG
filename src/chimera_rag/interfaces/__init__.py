"""Abstract base classes (ABCs) defining Chimera-RAG's pluggable slots.

Every concrete implementation (in :mod:`chimera_rag.defaults`,
:mod:`chimera_rag.plugins.adagraph`, or :mod:`chimera_rag.plugins.colla_rag`)
must subclass one of these. The :class:`Registry` then stores the class under
``(slot, name)`` so ``Pipeline`` can look it up at runtime.
"""
