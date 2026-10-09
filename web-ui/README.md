# Chimera-RAG Web UI

React + Vite + Tailwind + shadcn-style components; talks to the FastAPI
backend through `/api/*`.

## Dev

```bash
# In another terminal, start the backend:
uv run python main.py serve --config ../configs/full.yaml
# Then the Vite dev server:
cd web-ui
pnpm install
pnpm dev
# Browse http://localhost:5173 — API calls are proxied to 127.0.0.1:8000
```

## Build for production (single-port deployment)

```bash
pnpm build
# Copy dist to FastAPI's static mount point:
rm -rf ../src/chimera_rag/web/static
mkdir -p ../src/chimera_rag/web/static
cp -r dist/* ../src/chimera_rag/web/static/
# Now `uv run python main.py serve` alone hosts both API + UI.
```

## Pages

| Route | Purpose |
|-------|---------|
| `/ingest` | Build the knowledge graph from a configured dataset or inline text. |
| `/query` | Ask a question; see answer + intent + strategy + evidence chunks + triples. |
| `/graph` | Cytoscape visualisation of the KG; filter by entity. |
| `/evaluate` | Run `Evaluator` on a named dataset; see metrics + per-example table. |
| `/plugins` | Toggle AdaGraph / CollaRAG on the fly. |
| `/config` | Edit the live config.yaml (Monaco); PUT /api/config to hot-reload. |
| `/trace` | Client-side history of recent queries with full trace payloads. |
