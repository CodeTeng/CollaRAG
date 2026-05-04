"""Chimera-RAG pluggable innovations.

Each sub-package (``adagraph``, ``intent_agent``) is imported on demand
by :class:`ChimeraRAG` when the corresponding ``config.plugins.<name>.enabled``
flag is True.
"""
