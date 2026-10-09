"""Dual-layer memory system for multi-agent architecture.

Graph-grounded (Contribution 3): the shared layer adds an evidence-subgraph
overlap cache, and the private layer adds graph plan templates and structural
gap lessons. See :mod:`chimera_rag.plugins.colla_rag.memory.graph_memory`.
"""

from chimera_rag.plugins.colla_rag.memory.agent_memory import AgentPrivateMemory
from chimera_rag.plugins.colla_rag.memory.graph_memory import (
    EvidenceSubgraph,
    EvidenceSubgraphStore,
    GapLesson,
    GapLessonStore,
    PlanTemplateStore,
)
from chimera_rag.plugins.colla_rag.memory.memory_manager import MemoryManager, SessionMemory
from chimera_rag.plugins.colla_rag.memory.shared_memory import SharedMemory

__all__ = [
    "AgentPrivateMemory",
    "EvidenceSubgraph",
    "EvidenceSubgraphStore",
    "GapLesson",
    "GapLessonStore",
    "MemoryManager",
    "PlanTemplateStore",
    "SessionMemory",
    "SharedMemory",
]
