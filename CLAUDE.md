# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Python (uv-managed, Python 3.11+)

```bash
uv sync --all-extras --dev          # Install all dependencies
uv run pytest -q                    # Run all tests (fake providers, no API keys needed)
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
uv run python main.py infer --config configs/deepseek_intent_only.yaml --dataset hotpotqa --limit 10
uv run python eval.py --predictions_dir output/hotpotqa/deepseek_intent_only
```

## Architecture

Chimera-RAG is a pluggable knowledge-graph RAG framework with two optional research innovations that degrade gracefully when disabled.

### Six-layer architecture (top to bottom)

| Layer | Location | Role |
|---|---|---|
| Entry | `main.py` + `cli.py` | Six argparse subcommands (ingest/query/infer/export/serve) |
| Facade | `src/chimera_rag/__init__.py::ChimeraRAG` | Assembles all slots, exposes `ingest`/`query`/`stats`/`toggle_plugin` |
| Core | `core/` | `types.py`, `config.py`, `registry.py`, `pipeline.py`, `logging.py`, `exceptions.py`, `env_loader.py` |
| Interfaces | `interfaces/` | ABCs — all slot abstractions (`BaseChunker`, `BaseRetriever`, `BaseIntentClassifier`, `BaseAgent`, etc.) |
| Defaults | `defaults/` | Vanilla implementations registered as `defaults.*` — the "bare KG-RAG" baseline |
| Plugins | `plugins/adagraph/` + `plugins/intent_agent/` | Two innovations, registered as `<plugin>.*` |

**Critical rule**: `core/` never imports from `defaults/` or `plugins/`. The dependency direction is one-way: core → interfaces, defaults/plugins → interfaces + core.

### Plugin system: Registry + Slots

The `Registry` (`core/registry.py`) is a two-level map `slot → name → class`. Implementations register via the `@register(slot, name)` decorator. `ChimeraRAG._build()` resolves each slot by reading `config.<slot>.active` and calling `registry.get(slot, name)`. Plugins are only `importlib`-imported if `config.plugins.<name>.enabled` is `true`.

### Two pipelines share one PipelineState

- **IngestionPipeline**: `documents → chunker → embedder + extractor → pruner → graph_store + vector_store`
- **QueryPipeline**: `query → intent_classifier → retriever → generator → Answer`
- Both share `PipelineState` (graph_store, vector_store, chunk_lookup).

### The two innovations

1. **AdaGraph** — occupies 3 ingestion slots: `chunker` (adaptive size based on complexity/density), `extractor` (4-layer extraction L1-L4), `pruner` (similarity + transitive redundancy removal)

2. **IntentAgent (Multi-Agent Architecture)** — occupies 2 query slots (`intent_classifier` as `intent_agent.tree`, `retriever` as `intent_agent.multi_agent`). Inside the `MultiAgentRetriever` facade:
   - **Preprocessor Agent**: query rewriting, coreference resolution, follow-up merging
   - **Tree-Intent Classifier**: Level-1 greeting filter → Level-2 LLM + rule-fallback classification into `greeting | single_hop | multi_hop | summarization | other`
   - **Native RAG Agent**: single-round hybrid retrieval for simple factual queries
   - **ReAct Agent**: multi-step reasoning with tool selection + reflection loop for multi-hop/comparative/exploratory queries
   - **Map-Reduce Agent**: retrieve → map (per-chunk LLM summary) → reduce (final synthesis) for summarization queries
   - **WebSearch + LLM Agent**: DuckDuckGo search + optional local knowledge for out-of-scope queries
   - **Shared Tool Layer**: 19 tools (vector/BM25/hybrid/graph retrieval, sub-query decompose, assess evidence, rerank, query rewrite, memory read/write, web search, file IO)
   - **Dual-Layer Memory**: SharedMemory (QA cache, routing stats, entity aliases) + AgentPrivateMemory (per-agent per-intent JSONL experience store)

### Key design patterns (IntentAgent)

- **Strategy**: each Agent implements `BaseAgent.do_execute()`
- **Factory**: `AgentFactory.create(intent, all_tools, ...)` — maps intent label to Agent class
- **Builder**: `ToolSetBuilder.for_agent(agent_type, all_tools)` — filters tools per TOOL_MATRIX
- **Template Method**: `BaseAgent.execute()` calls `_pre_execute → do_execute → _post_execute`
- **Facade**: `MultiAgentRetriever.retrieve()` hides the full orchestration behind `BaseRetriever`

### Configuration

Single YAML config with 11 top-level domains: `app`, `logging`, `llm`, `embedding`, `rerank`, `storage`, `datasets`, `ingestion`, `query`, `plugins`, `web`, `evaluation`. Six config variants in `configs/`:

| Config | AdaGraph | IntentAgent |
|---|:---:|:---:|
| `default.yaml` | — | — |
| `deepseek_vanilla.yaml` | off | off |
| `deepseek_adagraph_only.yaml` | on | off |
| `deepseek_intent_only.yaml` | off | on |
| `deepseek_full.yaml` | on | on |
| `ollama_intent_only.yaml` | off | on |

Plugin toggling (`toggle_plugin`) rebuilds pipelines but preserves `PipelineState` — no re-ingestion needed.

### Data contracts (Pydantic v2)

All cross-layer types in `core/types.py`: `Document`, `Chunk`, `Triple`, `Entity`, `Query`, `Intent`, `ReasoningPlan`, `RetrievalResult`, `Answer`, `QualityFeedback`, `PreprocessResult`, `AgentExperienceEntry`, `QAExample`, `EvalResult`.

`MultiAgentIntentLabel = Literal["greeting", "single_hop", "multi_hop", "summarization", "other"]`

### IntentAgent plugin structure

```
plugins/intent_agent/
├── __init__.py              # Plugin entry, triggers @register
├── classifier.py            # TreeIntentClassifier (greeting + LLM + rule fallback)
├── orchestrator.py          # MultiAgentRetriever facade
├── agent_factory.py         # AgentFactory + ToolSetBuilder + TOOL_MATRIX
├── agents/                  # 5 specialized agents
│   ├── preprocessor.py      # PreprocessorAgent
│   ├── native_rag.py        # NativeRAGAgent
│   ├── react_agent.py       # ReActAgent
│   ├── map_reduce.py        # MapReduceAgent
│   └── web_search.py        # WebSearchAgent
├── tools/                   # 19 shared tools
│   ├── retrieval_tools.py   # vector/BM25/hybrid/graph (+ BM25Index, ToolAccumulator)
│   ├── reasoning_tools.py   # sub_query_decompose/assess_evidence/rerank
│   ├── preprocess_tools.py  # query_rewrite/coreference_resolve/followup_merge
│   ├── memory_tools.py      # read/write session/shared/agent memory
│   ├── web_tools.py         # web_search/web_fetch
│   ├── io_tools.py          # save_to_file/load_from_file
│   └── quality_evaluator.py # HeuristicQualityEvaluator
└── memory/                  # Dual-layer memory
    ├── shared_memory.py     # SharedMemory (JSON-backed)
    ├── agent_memory.py      # AgentPrivateMemory (JSONL per-agent per-intent)
    └── memory_manager.py    # MemoryManager facade + SessionMemory
```

### Web system

FastAPI at `src/chimera_rag/web/` — 7 route modules. React frontend at `web-ui/` (Vite + TypeScript + Tailwind v4 + Cytoscape). Dev: Vite on :5173 proxies `/api` to FastAPI on :8000. Production: `pnpm build` output mounted as static files.

### Testing

All tests use `FakeLLMProvider` + `FakeEmbeddingProvider` (test-only doubles in
`tests/conftest.py`) — no real API keys needed. Tests cover: registry, config,
types, pipeline end-to-end (vanilla/AdaGraph/IntentAgent/full), individual
defaults, agents, tools, memory, stores, datasets, and evaluation.
