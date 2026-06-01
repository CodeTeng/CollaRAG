# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Python (uv-managed, Python 3.11+)

```bash
uv sync --all-extras --dev          # Install all dependencies
uv run pytest -q                    # Run all tests (mock providers, no API keys needed)
uv run pytest tests/test_core_config.py -q                     # Run a single test file
uv run pytest tests/test_core_config.py::test_load_missing_file -q  # Run a single test
uv run ruff check .                 # Lint
uv run ruff format --check .        # Format check
uv run python main.py serve --config configs/deepseek_vanilla.yaml  # Start API server
```

### Web UI (pnpm, in `web-ui/`)

```bash
cd web-ui && pnpm install     # Install dependencies
cd web-ui && pnpm dev          # Vite dev server on :5173, proxies /api to :8000
cd web-ui && pnpm build       # Production build → dist/
cd web-ui && pnpm lint         # ESLint
```

### CLI Subcommands

```bash
uv run python main.py ingest --config <cfg> --dataset <name>
uv run python main.py query --config <cfg> "<question>"
uv run python main.py infer --config <cfg> --dataset <name> [--limit N]
uv run python main.py export --config <cfg> --out ./output/
uv run python main.py serve --config <cfg>
```

### Evaluation (two-stage)

```bash
uv run python main.py infer --config configs/deepseek_vanilla.yaml --dataset hotpotqa --limit 10
uv run python eval.py --predictions_dir output/hotpotqa/deepseek_vanilla
```

## Architecture

Chimera-RAG is a pluggable knowledge-graph RAG framework with two optional research innovations that degrade gracefully when disabled.

### Six-layer architecture (top to bottom)

| Layer | Location | Role |
|---|---|---|
| Entry | `main.py` + `cli.py` | Six argparse subcommands (ingest/query/infer/export/serve) |
| Facade | `src/chimera_rag/__init__.py::ChimeraRAG` | Assembles all slots, exposes `ingest`/`query`/`stats`/`toggle_plugin` |
| Core | `core/` | `types.py`, `config.py`, `registry.py`, `pipeline.py`, `logging.py`, `exceptions.py`, `env_loader.py` |
| Interfaces | `interfaces/` | 14 ABCs — all slot abstractions |
| Defaults | `defaults/` | 9 vanilla implementations registered as `defaults.*` — the "bare KG-RAG" baseline |
| Plugins | `plugins/adagraph/` + `plugins/intent_agent/` | Two innovations, registered as `<plugin>.*` |

**Critical rule**: `core/` never imports from `defaults/` or `plugins/`. The dependency direction is one-way: core → interfaces, defaults/plugins → interfaces + core.

### Plugin system: Registry + Slots

The `Registry` (`core/registry.py`) is a two-level map `slot → name → class`. Implementations register via the `@register(slot, name)` decorator:

```python
@register("chunker", "adagraph.dynamic")
class DynamicChunker(BaseChunker): ...
```

`ChimeraRAG._build()` resolves each slot by reading `config.<slot>.active` (e.g., `"defaults.fixed"` or `"adagraph.dynamic"`) and calling `registry.get(slot, name)`. Plugins are only `importlib`-imported if their `config.plugins.<name>.enabled` is `true` — so optional dependencies don't break when a plugin is off.

### Two pipelines share one PipelineState

- **IngestionPipeline** (`core/pipeline.py`): `documents → chunker → embedder + extractor → pruner → graph_store + vector_store`. Produces knowledge graph.
- **QueryPipeline**: `query → intent_classifier → retriever → generator → Answer`. Both share `PipelineState` (graph_store, vector_store, chunk_lookup).

### The two innovations

1. **AdaGraph** — occupies 3 ingestion slots: `chunker` (adaptive size based on complexity/density), `extractor` (4-layer extraction L1-L4), `pruner` (similarity + transitive redundancy removal)
2. **IntentAgent** — occupies 2 query slots (`intent_classifier`, `retriever`), internally has Memory/Reasoning/Decision/Execution stages + 6 strategies + 3 pluggable hooks (reflection, reranking, parallel pool)

### Configuration

Single YAML config drives everything with 10 top-level domains: `app`, `logging`, `llm`, `embedding`, `storage`, `datasets`, `ingestion`, `query`, `plugins`, `web`, `evaluation`. Five config variants in `configs/` let you run vanilla, AdaGraph-only, IntentAgent-only, or full. Plugin toggling (`toggle_plugin`) rebuilds pipelines but preserves `PipelineState` — no re-ingestion needed.

### Data contracts (Pydantic v2)

All cross-layer types in `core/types.py`: `Document`, `Chunk`, `Triple`, `Entity`, `Query`, `Intent`, `ReasoningPlan`, `RetrievalResult`, `Answer`, `QualityFeedback`, `QAExample`, `EvalResult`.

### Web system

FastAPI at `src/chimera_rag/web/` — 7 route modules under `routers/`: health, ingest, query, graph, plugins, config, evaluate. React frontend at `web-ui/` (Vite + TypeScript + Tailwind v4 + Cytoscape for graph viz + Zustand for state). Dev: Vite on :5173 proxies `/api` to FastAPI on :8000. Production: `pnpm build` output mounted as static files by FastAPI at `/`.

### Testing

All tests use `MockLLMProvider` + `MockEmbeddingProvider` — no real API keys needed. Conftest at `tests/conftest.py` provides shared fixtures. Tests cover: registry, config, types, pipeline end-to-end (vanilla/AdaGraph/IntentAgent), individual defaults, individual plugins, stores, datasets, evaluation metrics, and the web API.
