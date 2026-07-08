"""Specialized agents for multi-agent RAG architecture."""
from chimera_rag.plugins.colla_rag.agents.multi_hop import MultiHopAgent
from chimera_rag.plugins.colla_rag.agents.other import OtherAgent
from chimera_rag.plugins.colla_rag.agents.preprocessor import PreprocessorAgent
from chimera_rag.plugins.colla_rag.agents.single_hop import SingleHopAgent
from chimera_rag.plugins.colla_rag.agents.summarization import SummarizationAgent

__all__ = ["MultiHopAgent", "OtherAgent", "PreprocessorAgent", "SingleHopAgent", "SummarizationAgent"]
