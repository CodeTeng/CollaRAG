// Thin axios wrapper around the Chimera-RAG FastAPI backend.
// All endpoints live under /api; in dev the Vite server proxies /api to
// 127.0.0.1:8000, in prod the same origin hosts both static UI and API.

import axios from "axios";
import type {
  AgentsOverviewResponse,
  ConfigResponse,
  ConfigUpdateAck,
  EvaluateRequest,
  EvaluateResponse,
  GraphQuery,
  GraphResponse,
  HealthResponse,
  IngestRequest,
  IngestResponse,
  PluginsResponse,
  QueryRequest,
  QueryResponse,
  ToolMatrixResponse,
} from "./types";

const http = axios.create({
  baseURL: "/api",
  headers: { "Content-Type": "application/json" },
});

export const ChimeraAPI = {
  async health(): Promise<HealthResponse> {
    const { data } = await http.get<HealthResponse>("/health");
    return data;
  },

  async ingest(body: IngestRequest): Promise<IngestResponse> {
    const { data } = await http.post<IngestResponse>("/ingest", body);
    return data;
  },

  async query(body: QueryRequest): Promise<QueryResponse> {
    const { data } = await http.post<QueryResponse>("/query", body);
    return data;
  },

  async graph(params?: GraphQuery): Promise<GraphResponse> {
    const { data } = await http.get<GraphResponse>("/graph", { params });
    return data;
  },

  async plugins(): Promise<PluginsResponse> {
    const { data } = await http.get<PluginsResponse>("/plugins");
    return data;
  },

  async togglePlugin(name: string, enabled: boolean): Promise<PluginsResponse> {
    const { data } = await http.post<PluginsResponse>("/plugins/toggle", {
      name,
      enabled,
    });
    return data;
  },

  async getConfig(): Promise<ConfigResponse> {
    const { data } = await http.get<ConfigResponse>("/config");
    return data;
  },

  async putConfig(config: Record<string, unknown>): Promise<ConfigUpdateAck> {
    const { data } = await http.put<ConfigUpdateAck>("/config", { config });
    return data;
  },

  async evaluate(body: EvaluateRequest): Promise<EvaluateResponse> {
    const { data } = await http.post<EvaluateResponse>("/evaluate", body);
    return data;
  },

  async agentsOverview(): Promise<AgentsOverviewResponse> {
    const { data } = await http.get<AgentsOverviewResponse>("/agents/overview");
    return data;
  },

  async toolMatrix(): Promise<ToolMatrixResponse> {
    const { data } = await http.get<ToolMatrixResponse>("/agents/matrix");
    return data;
  },
};
