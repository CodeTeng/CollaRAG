// TypeScript mirrors of the FastAPI Pydantic schemas in
// src/chimera_rag/web/schemas.py. Kept intentionally close to the backend
// shapes so the API client stays type-safe end-to-end.

import { AxiosError } from "axios";

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------
export interface HealthResponse {
  status: string;
  version: string;
  active_plugins: Record<string, boolean>;
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------
export interface IngestRequest {
  dataset?: string;
  documents?: Array<Record<string, unknown>>;
}

export interface IngestStats {
  documents: number;
  chunks: number;
  triples_raw: number;
  triples_pruned: number;
  [key: string]: number;
}

export interface IngestState {
  chunks: number;
  triples: number;
  [key: string]: number;
}

export interface IngestResponse {
  stats: IngestStats;
  state: IngestState;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------
export interface QueryRequest {
  text: string;
  session_id?: string;
  top_k?: number;
}

export interface TriplePayload {
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
  layer: string;
  source_chunk_id?: string | null;
}

export interface ChunkPayload {
  chunk_id: string;
  doc_id: string;
  index: number;
  text: string;
  metadata: Record<string, unknown>;
}

export interface QueryResponse {
  answer: string;
  intent?: string | null;
  strategy?: string | null;
  confidence: number;
  evidence_chunk_ids: string[];
  chunks: ChunkPayload[];
  triples: TriplePayload[];
  trace: Record<string, unknown> & { retrieval_metadata?: unknown };
  agent_trace: AgentTracePayload;
}

export interface AgentTracePayload {
  agent_type?: string | null;
  rewritten_query?: string | null;
  original_query?: string | null;
  plan: Array<Record<string, unknown>>;
  step_results: string[];
  tool_call_log: string[];
  reflect_count: number;
  quality?: number | null;
  map_count?: number | null;
  web_results_count?: number | null;
  iterations?: number | null;
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------
export interface GraphNode {
  id: string;
  label: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  layer: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: Record<string, unknown>;
}

export interface GraphQuery {
  entity?: string;
  hops?: number;
  limit_edges?: number;
}

// ---------------------------------------------------------------------------
// Plugins
// ---------------------------------------------------------------------------
export interface PluginStatus {
  name: string;
  enabled: boolean;
  active_slots: Record<string, string>;
}

export interface PluginsResponse {
  plugins: PluginStatus[];
  active_implementations: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
export interface ConfigResponse {
  config: Record<string, unknown> & {
    datasets?: Record<string, unknown>;
  };
}

export interface ConfigUpdateAck {
  config: Record<string, unknown>;
  active_implementations: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
export interface EvaluateRequest {
  dataset: string;
  limit?: number;
}

export interface EvaluateExample {
  qid: string | number;
  question: string;
  reference: string;
  prediction: string;
  em?: number;
  f1?: number;
  [key: string]: unknown;
}

export interface EvaluateResponse {
  config_name: string;
  metrics: Record<string, number>;
  per_example: EvaluateExample[];
}

// ---------------------------------------------------------------------------
// Agents (CollaRAG Innovation 2)
// ---------------------------------------------------------------------------
export interface AgentInfo {
  agent_type: string;
  cn_name: string;
  description: string;
  icon: string;
  color: string;
  tool_count: number;
  tool_names: string[];
}

export interface ToolItem {
  name: string;
  cn_name: string;
  uses_llm: boolean;
}

export interface ToolCategoryInfo {
  category: string;
  cn_name: string;
  tools: ToolItem[];
}

export interface AgentsOverviewResponse {
  enabled: boolean;
  agent_count: number;
  tool_count: number;
  llm_tool_count: number;
  deterministic_tool_count: number;
  agents: AgentInfo[];
  tool_categories: ToolCategoryInfo[];
}

export interface ToolMatrixResponse {
  matrix: Record<string, string[]>;
  agent_types: string[];
  tool_totals: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------
export function extractErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return String(error);
}
