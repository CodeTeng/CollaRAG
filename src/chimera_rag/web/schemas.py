"""Pydantic request/response schemas for the web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    version: str
    active_plugins: dict[str, bool]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    dataset: str | None = None
    # Alternative: inline documents
    documents: list[dict[str, Any]] | None = None


class IngestResponse(BaseModel):
    stats: dict[str, Any]
    state: dict[str, Any]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    text: str
    session_id: str | None = None
    top_k: int | None = None


class TriplePayload(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    layer: str = "L1"
    source_chunk_id: str | None = None


class ChunkPayload(BaseModel):
    chunk_id: str
    doc_id: str
    index: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    intent: str | None = None
    strategy: str | None = None
    confidence: float = 1.0
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    chunks: list[ChunkPayload] = Field(default_factory=list)
    triples: list[TriplePayload] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
class GraphNode(BaseModel):
    id: str
    label: str


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    layer: str = "L1"


class GraphResponse(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Plugins
# ---------------------------------------------------------------------------
class PluginStatus(BaseModel):
    name: str
    enabled: bool
    active_slots: dict[str, str] = Field(default_factory=dict)


class PluginsResponse(BaseModel):
    plugins: list[PluginStatus]
    active_implementations: dict[str, str]


class PluginToggleRequest(BaseModel):
    name: str
    enabled: bool


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class ConfigResponse(BaseModel):
    config: dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class EvaluateRequest(BaseModel):
    dataset: str
    limit: int | None = None


class EvaluateResponse(BaseModel):
    config_name: str
    metrics: dict[str, float]
    per_example: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "ChunkPayload",
    "ConfigResponse",
    "ConfigUpdateRequest",
    "EvaluateRequest",
    "EvaluateResponse",
    "GraphEdge",
    "GraphNode",
    "GraphResponse",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "PluginStatus",
    "PluginToggleRequest",
    "PluginsResponse",
    "QueryRequest",
    "QueryResponse",
    "TriplePayload",
]
