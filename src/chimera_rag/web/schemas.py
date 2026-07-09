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


class AgentTracePayload(BaseModel):
    agent_type: str | None = None
    rewritten_query: str | None = None
    original_query: str | None = None
    plan: list[dict[str, Any]] = Field(default_factory=list)
    step_results: list[str] = Field(default_factory=list)
    tool_call_log: list[str] = Field(default_factory=list)
    reflect_count: int = 0
    quality: float | None = None
    map_count: int | None = None
    web_results_count: int | None = None
    iterations: int | None = None


class QueryResponse(BaseModel):
    answer: str
    intent: str | None = None
    strategy: str | None = None
    confidence: float = 1.0
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    chunks: list[ChunkPayload] = Field(default_factory=list)
    triples: list[TriplePayload] = Field(default_factory=list)
    trace: dict[str, Any] = Field(default_factory=dict)
    agent_trace: AgentTracePayload = Field(default_factory=AgentTracePayload)


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
# Agents (CollaRAG Innovation 2)
# ---------------------------------------------------------------------------
class AgentInfo(BaseModel):
    agent_type: str
    cn_name: str
    description: str
    icon: str
    color: str
    tool_count: int
    tool_names: list[str] = Field(default_factory=list)


class ToolCategoryInfo(BaseModel):
    category: str
    cn_name: str
    tools: list[dict[str, Any]] = Field(default_factory=list)


class AgentsOverviewResponse(BaseModel):
    enabled: bool
    agent_count: int
    tool_count: int
    llm_tool_count: int
    deterministic_tool_count: int
    agents: list[AgentInfo] = Field(default_factory=list)
    tool_categories: list[ToolCategoryInfo] = Field(default_factory=list)


class ToolMatrixResponse(BaseModel):
    matrix: dict[str, list[str]]
    agent_types: list[str]
    tool_totals: dict[str, int]


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
    "AgentInfo",
    "AgentsOverviewResponse",
    "AgentTracePayload",
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
    "ToolCategoryInfo",
    "ToolMatrixResponse",
    "TriplePayload",
]
