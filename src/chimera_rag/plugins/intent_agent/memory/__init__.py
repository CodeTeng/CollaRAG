"""Dual-layer memory system for multi-agent architecture."""
# Re-export legacy SessionMemory so old imports like
# `from chimera_rag.plugins.intent_agent.memory import SessionMemory` still work.
from chimera_rag.plugins.intent_agent.session_memory import SessionMemory

__all__ = ["SessionMemory"]
