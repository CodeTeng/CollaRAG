# CollaRAG: Graph-Grounded Division of Labor for Multi-Agent Retrieval

> A pluggable KG-RAG framework whose query side implements **CollaRAG** — a multi-agent retrieval architecture in which every routing, reasoning, memory, and capability decision is grounded in a shared knowledge graph.
> A single switch in `config.yaml` degrades the system to a vanilla baseline or activates the full plugin set, with zero changes to `core/`.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="uv" src="https://img.shields.io/badge/pkg-uv-orange.svg">
</p>

---

## The Four Contributions of CollaRAG

| # | Contribution | Mechanism in one sentence |
|---|---|---|
| **C1** | **Intent-Aware Query Triage** | A two-level cascade classifier (regex + LLM + keyword fallback) routes each query in one shot to one of 5 specialist agents, replacing the one-pipeline-fits-all design |
| **C2** | **G-PER** (Graph-Grounded Plan-Execute-Reflect) | A multi-hop reasoning protocol: **Plan** emits graph operations (link/traverse/attribute/aggregate) validated against the KG schema; **Execute** shares a frontier budget so easy steps donate surplus budget to hard ones; **Reflect** runs structural completeness checks (typed gaps) instead of scalar scoring, plus an adversarial contradiction probe |
| **C3** | **Graph-Grounded Dual-Layer Memory** | A shared layer retrieves evidence subgraphs by **entity-neighborhood overlap (Jaccard)** rather than text hash; a private layer stores graph plan templates and typed gap lessons |
| **C4** | **Graph-Conditioned Capability Specialization** | Each agent's tool permissions are dynamically adjusted by the query's structural footprint on the KG (hitRate / hasPath / density) — graph-poor regions automatically withdraw traversal tools and add text fallbacks |

**Graceful degradation**: every graph-grounded mechanism in C2/C3/C4 degrades to a text-only counterpart when the `graph_store` is absent or sparse (this is the graph-poor ablation setting in the paper).

---

## What It Does

| Scenario | Capability |
|---|---|
| Corpus → Graph | Automatic chunking, LLM triple extraction, dual graph + vector storage |
| Question → Answer | Tree intent classification → routing to 5 specialist agents → evidence-grounded generation |
| Ablation | Four configurations on the same dataset: `vanilla / adagraph_only / colla_rag_only / full` |
| Offline evaluation | EM / F1 / ROUGE-L + latency and token distributions + multi-config comparison reports |
| Single-port deployment | FastAPI + React, one `serve` command |

---

## Quick Start

### 1. Installation

The `agent` and `ragas` extras conflict (see `[tool.uv] conflicts` in `pyproject.toml`) — **do not use `--all-extras`**; install what you need:

```bash
# Run tests + use the CollaRAG multi-agent plugin (recommended)
uv sync --extra web --extra agent --extra llm --extra vec --dev

# If RAGAS semantic evaluation is needed (mutually exclusive with agent)
uv sync --extra ragas --dev
```

### 2. Run tests (zero API keys)

```bash
uv run pytest -q
```

All tests run against `FakeLLMProvider` / `FakeEmbeddingProvider`; no real key is required.

### 3. Configure the LLM

Create a `.env` in the repository root (one line is enough):

```bash
LLM_API_KEY=sk-xxxxxxxx
```

Model name, base_url, etc. all live in `config.yaml`; `.env` holds only secrets:

```yaml
llm:
  model: deepseek-chat
  base_url: https://api.deepseek.com

embedding:
  model: text-embedding-3-small
  base_url: https://api.openai.com/v1
  dim: 1536

rerank:
  enabled: false
```

All providers speak the OpenAI-compatible protocol (`/v1/chat/completions`, `/v1/embeddings`, `/rerank`); one client adapts DeepSeek / OpenAI / Ollama / vLLM / SiliconFlow.

### 4. End-to-end demo

```bash
uv run python main.py infer --config configs/colla_rag_only.yaml --dataset mock_wiki --limit 10
uv run python experiments/eval.py --predictions output/mock_wiki/colla_rag_only --metrics em,f1,rouge_l
```

`mock_wiki` is a bundled 100-item synthetic QA set, so the demo works out of the box.

### 5. Real-dataset experiments (HotpotQA / 2Wiki / MuSiQue / NQ / PopQA / ASQA)

Download the datasets into `data/` (gitignored; see `data/README.md`), then:

```bash
uv run python main.py ingest --config configs/colla_rag_only.yaml --dataset musique --limit 500 --rebuild
uv run python main.py infer  --config configs/colla_rag_only.yaml --dataset musique --limit 500
uv run python experiments/eval.py --predictions output/musique/colla_rag_only --metrics em,f1,rouge_l
```

### 6. Reproducing the paper experiments

```bash
bash experiments/run_all.sh --limit 50    # smoke test (50 questions per dataset)
bash experiments/run_all.sh --limit 500   # larger run (500 per dataset; paper results use the full evaluation sets)
```

---

## Architecture at a Glance

```
                    ┌────────────────────────────────────────┐
                    │          ChimeraRAG (facade)           │
                    └──────────┬─────────────────────────────┘
                               │
      ┌────────────────────────┼───────────────────────────┐
      │                        │                           │
┌─────▼──────┐         ┌───────▼───────┐          ┌────────▼────────┐
│IngestionPL │         │ QueryPipeline │          │  PipelineState  │
│chunk→embed │         │intent→retr→gen│          │ graph + vector  │
│→extract    │         │  (MultiAgent  │          │ + chunk_lookup  │
│→prune      │         │  Retriever)   │          │                 │
└────────────┘         └───────────────┘          └─────────────────┘
      ↑                        ↑
  defaults.*              defaults.*
  adagraph.*              colla_rag.*
```

Three dependency invariants, all guarded by tests:

1. `core/` must not import `defaults/` or `plugins/`
2. `plugins/adagraph/` and `plugins/colla_rag/` must not depend on each other
3. All cross-boundary calls go through `interfaces/*` ABCs + the registry

---

## Configuration Presets

`configs/` ships 6 YAML presets:

| File | AdaGraph | CollaRAG | Purpose |
|---|:---:|:---:|---|
| `default.yaml` | — | — | Fully annotated template |
| `vanilla.yaml` | off | off | Bare KG-RAG baseline |
| `adagraph_only.yaml` | on | off | Construction-side baseline only |
| `colla_rag_only.yaml` | off | on | CollaRAG only (cloud DeepSeek) |
| `full.yaml` | on | on | Everything enabled |
| `ollama_colla_rag_only.yaml` | off | on | CollaRAG only (local Ollama, no API cost) |

---

## Project Structure

```
chimera-rag/
├── main.py                       # CLI entry (ingest/query/infer/export/serve)
├── configs/                      # 6 YAML presets
├── data/                         # Datasets (gitignored except bundled mock_wiki)
├── experiments/                  # Reproducible experiment scripts (9 experiments)
│   ├── README.md                 # Experiment guide
│   ├── eval.py                   # Offline scoring (EM/F1/ROUGE-L)
│   ├── eval_ragas.py             # RAGAS semantic evaluation
│   ├── run_all.sh                # One-command full run
│   └── exp1..exp9_*.py           # 9 paper experiments
│
├── src/chimera_rag/
│   ├── __init__.py               # ChimeraRAG facade (graph persistence / fingerprint / hot swap)
│   ├── cli.py                    # 5 subcommands
│   ├── core/                     # types / config / registry / pipeline / logging / exceptions
│   ├── interfaces/               # ABC interface layer
│   ├── defaults/                 # Vanilla implementations (fixed chunker / simple extractor /
│   │                             #   rule classifier / direct retriever / prompt generator)
│   ├── providers/                # LLM / Embedding / Rerank: OpenAI-compatible clients
│   ├── stores/                   # NetworkX (graph) + FAISS (vector)
│   ├── plugins/
│   │   ├── adagraph/             # Construction-side baseline plugin
│   │   └── colla_rag/            # CollaRAG: intent-aware multi-agent retrieval
│   │       ├── classifier.py     #   TreeIntentClassifier (C1)
│   │       ├── orchestrator.py   #   MultiAgentRetriever orchestrator (facade)
│   │       ├── agent_factory.py  #   AgentFactory + ToolSetBuilder + TOOL_MATRIX + GraphFootprint (C4)
│   │       ├── prompts.py        #   Centralized prompt templates
│   │       ├── agents/           #   5 specialist agents
│   │       ├── tools/            #   10 modules, 34 tools in the permission matrix
│   │       ├── memory/           #   Dual-layer graph-grounded memory (C3)
│   │       └── gper/             #   G-PER structural completeness checks (C2)
│   ├── datasets/                 # Dataset loaders
│   ├── evaluation/               # Inference runner + scorers + RAGAS reports
│   └── web/                      # FastAPI + React statics
│
└── tests/                        # 69 test files
```

---

## Web System

```bash
uv run python main.py serve --config configs/full.yaml --dataset mock_wiki
# http://localhost:8000
```

8 route modules with 12 endpoints (all under `/api`): health, ingest, query, graph visualization, agent overview / tool-permission matrix, plugin hot swap, online config read/write, and batch evaluation. The frontend lives in `web-ui/` (Vite + TypeScript + Tailwind + Cytoscape); in development the Vite dev server proxies `/api` to port 8000, and `pnpm build` output is mounted as static files in production.

---

## License

[MIT](./LICENSE)
