# Chimera-RAG

> 基于知识图谱的**可插拔 RAG 框架**。以希腊神话融合体 *Chimera* 为名,象征"知识构建"与"智能检索"两端的融合。通过 `config.yaml` 一行开关,可完全降级为 vanilla 基线,也可完全激活插件——`core/` 代码一行不动。

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="uv" src="https://img.shields.io/badge/pkg-uv-orange.svg">
</p>

---

## 项目状态:两个创新点

| # | 创新点 | 作用阶段 | 状态 |
|---|---|---|---|
| **一** | 面向知识构建的**新型分块方法** | Ingestion | 🚧 **规划中,尚未实装** |
| **二** | **CollaRAG** —— 图谱锚定的多 Agent 检索分工 | Query | ✅ **已实装**(含 4 个子贡献) |

> 📌 **关于 `plugins/adagraph/`**:它是构建侧的**前置基线**,为创新点一提供插槽承载(动态分块 / 分层提取 / 双重冗余剪枝的部分实装)。
> 它**本身不构成本仓库宣称的创新点**,请勿把它当作创新点一。

### 创新点二 CollaRAG 的四个子贡献

| # | 子贡献 | 一句话机制 |
|---|---|---|
| **C1** | **Intent-Aware Query Triage** | 两级级联分类器(正则 + LLM + 关键词回退),把查询路由到 5 类专职 Agent,取代"一条流水线打天下" |
| **C2** | **G-PER**(Graph-Grounded Plan-Execute-Reflect) | 多跳推理协议:**Plan** 产出图操作(link/traverse/attribute/aggregate)并过 KG schema 校验;**Execute** 用前沿式共享预算,简单步骤把富余预算捐给困难步骤;**Reflect** 做结构化完整性检查(typed gaps)而非标量打分,并做对抗性矛盾探针 |
| **C3** | **Graph-Grounded Dual-Layer Memory** | 共享层按**实体邻域重叠(Jaccard)**检索证据子图,而非文本哈希;私有层存图规划模板 + 类型化 gap lessons |
| **C4** | **Graph-Conditioned Capability Specialization** | 按查询在 KG 上的结构足迹(hitRate / hasPath / density)**动态收放** Agent 的工具权限——图贫乏区域自动撤回遍历工具、补文本回退 |

论文:*CollaRAG: Graph-Grounded Division of Labor for Multi-Agent Retrieval*,源码见 [`paper/my/current/`](paper/my/current/)。

**优雅退化**:C2/C3/C4 的所有图接地机制在 `graph_store` 缺席时各自退化为 text-only 版本(这也是论文里的 graph-poor 消融设置)。

---

## 它能做什么

| 场景 | 能力 |
|---|---|
| 语料 → 图谱 | 文档自动分块、LLM 抽三元组、图谱 + 向量双路存储 |
| 问题 → 答案 | 树形意图分类 → 5 类专职 Agent 路由 → 基于证据的生成 |
| 消融对比 | 同一数据集下 `vanilla / adagraph_only / colla_rag_only / full` 四组对照 |
| 离线评测 | EM / F1 / ROUGE-L + 耗时分布 + Token 分布 + 多配置对比报告 |
| 单端口部署 | FastAPI + React,`serve` 一条命令 |

---

## 快速开始

### 1. 安装

`agent` 与 `ragas` 两个 extra 存在依赖冲突(见 `pyproject.toml` 的 `[tool.uv] conflicts`),**不要用 `--all-extras`**,按需选装:

```bash
# 跑测试 + 用 CollaRAG 多Agent(推荐)
uv sync --extra web --extra agent --extra llm --extra vec --dev

# 若需要 RAGAS 语义评测(与 agent 互斥)
uv sync --extra ragas --dev
```

### 2. 跑测试(零 API Key)

```bash
uv run pytest -q
```

全部测试走 `FakeLLMProvider` / `FakeEmbeddingProvider`,不需要任何真实 key。

### 3. 配 LLM

在仓库根目录建 `.env`(只需一行):

```bash
LLM_API_KEY=sk-xxxxxxxx
```

模型名、base_url 等**全部写在 `config.yaml` 中**,`.env` 只放 secret:

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

所有 provider 走 OpenAI 兼容协议(`/v1/chat/completions`、`/v1/embeddings`、`/rerank`),同一套客户端适配 DeepSeek / OpenAI / Ollama / vLLM / SiliconFlow。

### 4. 端到端 demo

```bash
uv run python main.py infer --config configs/colla_rag_only.yaml --dataset mock_wiki --limit 10
uv run python experiments/eval.py --predictions output/mock_wiki/colla_rag_only --metrics em,f1,rouge_l
```

### 5. 真实数据集实验(HotpotQA / 2Wiki / MuSiQue)

```bash
uv run python main.py ingest --config configs/colla_rag_only.yaml --dataset musique --limit 500 --rebuild
uv run python main.py infer  --config configs/colla_rag_only.yaml --dataset musique --limit 500
uv run python experiments/eval.py --predictions output/musique/colla_rag_only --metrics em,f1,rouge_l
```

### 6. 论文实验复现

```bash
bash experiments/run_all.sh --limit 50    # 烟雾测试(每数据集 50 题)
bash experiments/run_all.sh --limit 500   # 较大规模运行(每数据集 500 题;论文结果取完整评测集)
```

---

## 架构一览

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

三条**由测试守护**的依赖硬约束:

1. `core/` 不允许 import `defaults/` 或 `plugins/`
2. `plugins/adagraph/` 与 `plugins/colla_rag/` 互不依赖
3. 所有跨边界调用只走 `interfaces/*` ABC + `registry`

详见 [docs/architecture_overview.md](docs/architecture_overview.md)。

---

## 配置文件

`configs/` 提供 6 份 yaml 预设:

| 文件 | AdaGraph | CollaRAG | 用途 |
|---|:---:|:---:|---|
| `default.yaml` | — | — | 全字段注释模板 |
| `vanilla.yaml` | off | off | 裸 KG-RAG 基线 |
| `adagraph_only.yaml` | on | off | 仅测构建侧基线 |
| `colla_rag_only.yaml` | off | on | 仅测创新点二(云端 DeepSeek) |
| `full.yaml` | on | on | 全启用 |
| `ollama_colla_rag_only.yaml` | off | on | 仅测创新点二(本地 Ollama,免 API 费用) |

---

## 项目结构

```
chimera-rag/
├── main.py                       # CLI 入口 (ingest/query/infer/export/serve)
├── configs/                      # 6 份 yaml 预设
├── data/                         # 数据集
├── docs/                         # 设计文档 + 实验结果
├── experiments/                  # 论文实验可复现脚本（9 项实验）
│   ├── README.md                 # 实验说明与运行指南
│   ├── eval.py                   # 离线打分（EM/F1/ROUGE-L）
│   ├── eval_ragas.py             # RAGAS 语义评测（忠实性/相关性/精度/召回）
│   ├── utils.py                  # 共享工具（配置生成/结果收集/渲染）
│   ├── run_all.sh                # 一键运行全部实验
│   └── exp1..exp9_*.py           # 9 项论文实验
│
├── src/chimera_rag/
│   ├── __init__.py               # ChimeraRAG 门面（含图谱持久化/指纹校验/热切换）
│   ├── cli.py                    # 5 个子命令
│   ├── core/                     # types / config / registry / pipeline / logging / exceptions / env_loader
│   ├── interfaces/               # ABC 接口层
│   ├── defaults/                 # vanilla 实现：fixed chunker / simple_llm extractor /
│   │                             #   noop pruner / rule classifier / direct retriever / prompt generator
│   ├── providers/                # LLM / Embedding / Rerank：OpenAI-compatible 客户端
│   ├── stores/                   # NetworkX（图）+ FAISS（向量）
│   ├── plugins/
│   │   ├── adagraph/             # 构建侧前置基线（非本仓库宣称的创新点）
│   │   │   ├── dynamic_chunker.py      # 自适应分块（已实装）
│   │   │   ├── layered_extractor.py    # L1 已实装，L2–L4 TODO
│   │   │   └── dual_pruner.py          # 相似性已实装，传递性 TODO
│   │   └── colla_rag/            # 创新点二：多Agent意图感知检索
│   │       ├── classifier.py     #   TreeIntentClassifier（C1）
│   │       ├── orchestrator.py   #   MultiAgentRetriever 编排器（Facade）
│   │       ├── agent_factory.py  #   AgentFactory + ToolSetBuilder + TOOL_MATRIX + GraphFootprint（C4）
│   │       ├── prompts.py        #   集中式 prompt 模板
│   │       ├── agents/           #   5 个专职 Agent
│   │       ├── tools/            #   10 个模块 · 34 个工具全部进入工具矩阵
│   │       ├── memory/           #   双层记忆（含图接地三层，C3）
│   │       │   ├── shared_memory.py     # 共享层：路由统计 / QA 缓存 / 实体别名 / 改写模板
│   │       │   ├── agent_memory.py      # 私有层：按 Agent × 意图分桶的 JSONL
│   │       │   ├── graph_memory.py      # 图接地：EvidenceSubgraph / PlanTemplate / GapLesson
│   │       │   └── memory_manager.py    # MemoryManager 门面 + SessionMemory
│   │       └── gper/             #   G-PER 结构化完整性检查（C2）
│   │           └── completeness.py      # ShapeSpec + 类型化 gap（missing_attribute/path/relation）
│   ├── datasets/                 # 数据集 loader
│   ├── evaluation/               # 推理运行器 + 打分器 + RAGAS 报告
│   └── web/                      # FastAPI + React 静态
│
└── tests/                        # 69 个测试文件
```

---

## Web 系统

```bash
uv run python main.py serve --config configs/full.yaml --dataset mock_wiki
# http://localhost:8000
```

8 个路由模块、12 个端点(全部挂载在 `/api` 前缀下):

| 路径 | 用途 |
|---|---|
| `GET /api/health` | 健康检查 + 插件状态 |
| `POST /api/ingest` | 摄取数据集或内联文档 |
| `POST /api/query` | 单次问答 |
| `GET /api/graph` · `GET /api/graph/stats` | 图谱可视化与统计 |
| `GET /api/agents/overview` · `GET /api/agents/matrix` | Agent 总览与工具权限矩阵 |
| `GET /api/plugins` · `POST /api/plugins/toggle` | 插件热切换 |
| `GET /api/config` · `PUT /api/config` | 在线读 / 写配置 |
| `POST /api/evaluate` | 批量评测入口 |

前端在 `web-ui/`(Vite + TypeScript + Tailwind + Cytoscape)。开发时 Vite dev server 把 `/api` 代理到 8000;生产 `pnpm build` 后作为静态文件挂载。

---

## 文档索引

| 文档 | 说明 |
|---|---|
| [docs/experiments/](docs/experiments/) | **实验结果**:10 份实验数据文档 |
| [experiments/](experiments/) | **实验代码**:9 项论文实验的可复现脚本 |
| [docs/architecture_overview.md](docs/architecture_overview.md) | 架构:注册表、插槽、依赖注入、管线数据流 |
| [docs/innovation-1/adagraph.md](docs/innovation-1/adagraph.md) | 构建侧前置基线的实装细节(非创新点) |
| [docs/innovation-2/colla-rag.md](docs/innovation-2/colla-rag.md) | CollaRAG 多Agent架构实装细节 |
| [docs/innovation-2/design.md](docs/innovation-2/design.md) | 多Agent架构设计文档 |

---

## License

[MIT](./LICENSE)
