# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Python (uv-managed, Python 3.11+)

```bash
uv sync --extra web --extra agent --extra llm --extra vec --dev   # agent / ragas extras conflict — do NOT use --all-extras
uv run pytest -q                    # Run all tests (fake providers, no API keys needed)
uv run pytest tests/test_core_config.py -q                     # Run a single test file
uv run pytest tests/test_core_config.py::test_load_missing_file -q  # Run a single test
uv run ruff check .                 # Lint
uv run ruff format --check .        # Format check
uv run python main.py serve --config configs/vanilla.yaml  # Start API server
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
uv run python main.py infer --config configs/colla_rag_only.yaml --dataset hotpotqa --limit 10
uv run python experiments/eval.py --predictions output/hotpotqa/colla_rag_only
```

### Experiments (paper reproduction)

All 9 paper experiments live in `experiments/` with a shared `utils.py` toolkit:

```bash
bash experiments/run_all.sh --limit 50          # Smoke test all experiments (50 samples)
bash experiments/run_all.sh --limit 500         # Larger run (500 samples/dataset; the paper's numbers come from the full eval sets)
bash experiments/run_all.sh --exp 1 2 3         # Run only specific experiments

# Run individual experiments
uv run python experiments/exp1_main_results.py --datasets nq --methods collarag --backbone qwen3-8b
uv run python experiments/exp2_ablation.py --datasets hotpotqa --limit 100
uv run python experiments/exp3_hyperparameter_sensitivity.py --param quality_threshold
uv run python experiments/exp4_intent_classification.py
uv run python experiments/exp5_retrieval_quality.py --methods collarag vanilla_rag
uv run python experiments/exp6_routing_counterfactual.py
uv run python experiments/exp7_routing_distribution.py
uv run python experiments/exp8_hop_depth.py --methods collarag hipporag2
uv run python experiments/exp9_memory_accumulation.py --window-size 50
```

Experiment output goes to `output/experiments/<exp_name>/` with `.json` + `.md` reports.

## Architecture

Chimera-RAG is a pluggable knowledge-graph RAG framework with two optional research innovations that degrade gracefully when disabled.

### Six-layer architecture (top to bottom)

| Layer | Location | Role |
|---|---|---|
| Entry | `main.py` + `cli.py` | Five argparse subcommands (ingest/query/infer/export/serve) |
| Facade | `src/chimera_rag/__init__.py::ChimeraRAG` | Assembles all slots, exposes `ingest`/`query`/`stats`/`toggle_plugin` |
| Core | `core/` | `types.py`, `config.py`, `registry.py`, `pipeline.py`, `logging.py`, `exceptions.py`, `env_loader.py` |
| Interfaces | `interfaces/` | ABCs — all slot abstractions (`BaseChunker`, `BaseRetriever`, `BaseIntentClassifier`, `BaseAgent`, etc.) |
| Defaults | `defaults/` | Vanilla implementations registered as `defaults.*` — the "bare KG-RAG" baseline |
| Plugins | `plugins/adagraph/` + `plugins/colla_rag/` | Two innovations, registered as `<plugin>.*` |

**Critical rule**: `core/` never imports from `defaults/` or `plugins/`. The dependency direction is one-way: core → interfaces, defaults/plugins → interfaces + core.

### Innovation status (keep this framing in mind)

The repo targets **two claimed innovations**:

1. **Innovation #1 — a novel chunking method (build side). NOT YET IMPLEMENTED.** The existing `plugins/adagraph/` is the build-side *baseline* that reserves the slots for it — do not present AdaGraph as a claimed contribution.
2. **Innovation #2 — CollaRAG (query side). Implemented.** Four sub-contributions:
   - **C1 Intent-Aware Query Triage** — `classifier.py` (tree intent classifier), `agents/`
   - **C2 G-PER** (Graph-Grounded Plan-Execute-Reflect) — `agents/multi_hop.py`, `gper/completeness.py` (schema-validated graph-op plans, frontier-budget execution, structural completeness check with typed gaps, adversarial probe)
   - **C3 Graph-Grounded Dual-Layer Memory** — `memory/graph_memory.py` (evidence-subgraph overlap cache 3B, plan templates 3A, gap lessons 3C)
   - **C4 Graph-Conditioned Capability Specialization** — `agent_factory.py` (`GraphFootprint` dynamically grows/shrinks each agent's tool subset)

### Plugin system: Registry + Slots

The `Registry` (`core/registry.py`) is a two-level map `slot → name → class`. Implementations register via the `@register(slot, name)` decorator. `ChimeraRAG._build()` resolves each slot by reading `config.<slot>.active` and calling `registry.get(slot, name)`. Plugins are only `importlib`-imported if `config.plugins.<name>.enabled` is `true`.

### Two pipelines share one PipelineState

- **IngestionPipeline**: `documents → chunker → embedder + extractor → pruner → graph_store + vector_store`
- **QueryPipeline**: `query → intent_classifier → retriever → generator → Answer`
- Both share `PipelineState` (graph_store, vector_store, chunk_lookup).

### The two slots packages

1. **AdaGraph (build-side baseline, NOT a claimed innovation)** — occupies 3 ingestion slots: `chunker` (adaptive size based on complexity/density), `extractor` (4-layer extraction, L1 implemented / L2-L4 TODO), `pruner` (similarity implemented / transitive TODO)

2. **CollaRAG (innovation #2)** — occupies 2 query slots (`intent_classifier` as `colla_rag.tree`, `retriever` as `colla_rag.multi_agent`). Inside the `MultiAgentRetriever` facade:
   - **Preprocessor Agent**: query rewriting, coreference resolution, follow-up merging (runs AFTER classification, on the original query; only the dispatched agent sees the rewritten query)
   - **Tree-Intent Classifier**: Level-1 greeting filter → Level-2 LLM + rule-fallback classification into `greeting | single_hop | multi_hop | summarization | other`
   - **SingleHop Agent**: single-round hybrid retrieval for simple factual queries
   - **MultiHop Agent**: G-PER — schema-validated graph-op plan (link/traverse/attribute/aggregate), frontier-budget execution, structural completeness check emitting typed gaps (missing_attribute/path/relation) + adversarial contradiction probe; degrades to text-only PER when graph_store is None
   - **Summarization Agent**: retrieve → map (per-chunk LLM summary) → reduce (final synthesis) for summarization queries
   - **Other Agent**: DuckDuckGo search + optional local knowledge for out-of-scope queries
   - **Shared Tool Layer**: 34 tools built across 10 modules (retrieval/entity/reasoning/preprocess/memory/web/io/verification/graph-advanced/evidence); all 34 enter `TOOL_MATRIX`. Memory offers read+write variants for session context / shared / agent memory; write variants are granted to the specialists that produce the corresponding data (SH/MH/SM for shared, SH/MH/SM/OT for agent-private, PP/MH for session context)
   - **Dual-Layer Memory**: SharedMemory (routing stats, QA cache, entity aliases, rewrite templates) + AgentPrivateMemory (per-agent per-intent JSONL) + graph-grounded layers (EvidenceSubgraphStore, PlanTemplateStore, GapLessonStore)

### Key design patterns (CollaRAG)

- **Strategy**: each Agent implements `BaseAgent.do_execute()`
- **Factory**: `AgentFactory.create(intent, all_tools, ...)` — maps intent label to Agent class
- **Builder**: `ToolSetBuilder.for_agent(agent_type, all_tools)` — filters tools per TOOL_MATRIX
- **Template Method**: `BaseAgent.execute()` calls `_pre_execute → do_execute → _post_execute`
- **Facade**: `MultiAgentRetriever.retrieve()` hides the full orchestration behind `BaseRetriever`

### Configuration

Single YAML config with 12 top-level domains: `app`, `logging`, `llm`, `embedding`, `rerank`, `storage`, `datasets`, `ingestion`, `query`, `plugins`, `web`, `evaluation`. Six config variants in `configs/`:

| Config | AdaGraph | CollaRAG |
|---|:---:|:---:|
| `default.yaml` | — | — |
| `vanilla.yaml` | off | off |
| `adagraph_only.yaml` | on | off |
| `colla_rag_only.yaml` | off | on |
| `full.yaml` | on | on |
| `ollama_colla_rag_only.yaml` | off | on |

Plugin toggling (`toggle_plugin`) rebuilds pipelines but preserves `PipelineState` — no re-ingestion needed.

### Data contracts (Pydantic v2)

All cross-layer types in `core/types.py`: `Document`, `Chunk`, `Triple`, `Entity`, `Query`, `Intent`, `ReasoningPlan`, `RetrievalResult`, `Answer`, `QualityFeedback`, `PreprocessResult`, `AgentExperienceEntry`, `QAExample`, `EvalResult`.

`MultiAgentIntentLabel = Literal["greeting", "single_hop", "multi_hop", "summarization", "other"]`

### CollaRAG plugin structure

```
plugins/colla_rag/
├── __init__.py              # Plugin entry, triggers @register
├── prompts.py               # Centralized prompt templates (24)
├── classifier.py            # TreeIntentClassifier (greeting + LLM + rule fallback)
├── orchestrator.py          # MultiAgentRetriever facade
├── agent_factory.py         # AgentFactory + ToolSetBuilder + TOOL_MATRIX + GraphFootprint
├── agents/                  # 5 specialized agents
│   ├── preprocessor.py      # PreprocessorAgent
│   ├── single_hop.py        # SingleHopAgent
│   ├── multi_hop.py         # MultiHopAgent (G-PER: Plan-Execute-Reflect)
│   ├── summarization.py     # SummarizationAgent
│   └── other.py             # OtherAgent
├── tools/                   # 34 tools built over 10 modules; all 34 enter TOOL_MATRIX
│   ├── retrieval_tools.py      # vector/BM25/hybrid/graph (+ BM25Index, ToolAccumulator)
│   ├── entity_tools.py         # entity_extract/relation_extract/entity_link
│   ├── reasoning_tools.py      # sub_query_decompose/assess_evidence/rerank
│   ├── preprocess_tools.py     # query_rewrite/coreference_resolve/followup_merge
│   ├── memory_tools.py         # 3 read + 3 write (session/shared/agent); write_* in no matrix
│   ├── web_tools.py            # web_search/web_fetch
│   ├── io_tools.py             # save_to_file/load_from_file
│   ├── verification_tools.py   # answer_verify/claim_decompose/source_attribution
│   ├── graph_advanced_tools.py # graph_subgraph_extract/graph_statistics
│   ├── evidence_tools.py       # temporal_filter/evidence_dedup/chunk_summarize/confidence_calibrate
│   └── quality_evaluator.py    # HeuristicQualityEvaluator (helper class, not a tool)
├── gper/                    # G-PER structural completeness (C2)
│   └── completeness.py      # ShapeSpec + typed EvidenceGap (attribute/path/relation)
└── memory/                  # Dual-layer memory (C3)
    ├── shared_memory.py     # SharedMemory (JSON-backed)
    ├── agent_memory.py      # AgentPrivateMemory (JSONL per-agent per-intent)
    ├── graph_memory.py      # EvidenceSubgraphStore / PlanTemplateStore / GapLessonStore
    └── memory_manager.py    # MemoryManager facade + SessionMemory
```

### Web system

FastAPI at `src/chimera_rag/web/` — 8 route modules, 12 endpoints. React frontend at `web-ui/` (Vite + TypeScript + Tailwind v4 + Cytoscape). Dev: Vite on :5173 proxies `/api` to FastAPI on :8000. Production: `pnpm build` output mounted as static files.

### Experiments

All 9 paper experiments live in `experiments/` as standalone Python scripts. Each script corresponds to one document in `docs/experiments/`:

| # | Script | What it tests |
|---|---|---|
| 1 | `exp1_main_results.py` | Full benchmark: 7 datasets × 10 methods × 2 backbones (Table 1 & 2) |
| 2 | `exp2_ablation.py` | Ablation: w/o Query Triage, w/o PER, w/o Memory |
| 3 | `exp3_hyperparameter_sensitivity.py` | Sweep 5 hyperparams on HotpotQA |
| 4 | `exp4_intent_classification.py` | TreeIntentClassifier accuracy on 430-sample annotated set |
| 5 | `exp5_retrieval_quality.py` | Recall@k, faithfulness, hallucination rate |
| 6 | `exp6_routing_counterfactual.py` | 4 routing strategies (normal / all-single / all-multi / random) |
| 7 | `exp7_routing_distribution.py` | Per-dataset intent label distribution |
| 8 | `exp8_hop_depth.py` | MuSiQue 2/3/4-hop breakdown + decay rates |
| 9 | `exp9_memory_accumulation.py` | Cold-start → warm-up memory flywheel curve |

Shared utilities in `experiments/utils.py`: config generation, inference runner, metric computation, Markdown table rendering. Master orchestrator: `experiments/run_all.sh`. Output: `output/experiments/<exp_name>/`.

### Testing

All tests use `FakeLLMProvider` + `FakeEmbeddingProvider` (test-only doubles in
`tests/_fakes.py`, wired in by `tests/conftest.py`) — no real API keys needed. Tests cover: registry, config,
types, pipeline end-to-end (vanilla/AdaGraph/CollaRAG/full), individual
defaults, agents, tools, memory, stores, datasets, and evaluation.
