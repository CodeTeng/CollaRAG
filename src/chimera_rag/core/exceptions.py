"""Error hierarchy for Chimera-RAG.

Callers can catch :class:`ChimeraError` broadly or specific subclasses
narrowly. All errors raised by Chimera-RAG inherit from this root.
"""

from __future__ import annotations


class ChimeraError(Exception):
    """Root of every error raised by the framework."""


class ConfigError(ChimeraError):
    """Raised when ``config.yaml`` is malformed or references unknown plugins."""


class PluginNotFoundError(ChimeraError):
    """Raised when a config-referenced plugin is not in the Registry."""

    def __init__(self, slot: str, name: str, message: str | None = None) -> None:
        self.slot = slot
        self.name = name
        super().__init__(
            message
            or f"No plugin registered for slot={slot!r}, name={name!r}. "
            f"Did you forget to import chimera_rag.plugins.<pkg>?"
        )


class ProviderError(ChimeraError):
    """Raised when an LLM / Embedding provider fails (network, auth, parse)."""


class DatasetError(ChimeraError):
    """Raised when a dataset file is missing, mis-formatted, or incomplete."""


class PipelineError(ChimeraError):
    """Raised by IngestionPipeline / QueryPipeline for orchestration errors."""


__all__ = [
    "ChimeraError",
    "ConfigError",
    "DatasetError",
    "PipelineError",
    "PluginNotFoundError",
    "ProviderError",
]
