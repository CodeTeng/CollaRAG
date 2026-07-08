# Chimera-RAG

> 基于知识图谱的可插拔 RAG 框架，以希腊神话融合体 *Chimera* 为名，象征两大创新点的协同：**AdaGraph**（自适应图谱构建）+ **CollaRAG**（多Agent意图感知智能检索）。通过 `config.yaml` 一键开关，可完全降级为 vanilla 基线，也可完全激活双创新点。

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-303%20passing-brightgreen.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="uv" src="https://img.shields.io/badge/pkg-uv-orange.svg">
</p>

---

## 它能做什么

| 场景 | 能力 |
|---|---|
| 语料 → 图谱 | 文本文档自动 chunk、LLM 抽三元组、图谱 + 向量双路存储 |
| 问题 → 答案 | 树形意图分类 → 5 类专职 Agent 路由 → 基于证据的生成 |
| 消融对比 | 同一数据集下 `vanilla / adagraph_only / colla_rag_only / full` 四组对比 |
| 离线评测 | EM / F1 / ROUGE-L + 耗时分布 + Token 分布 + 多配置对比报告 |
| 单端口部署 | FastAPI + React，`serve` 一条命令 |

---

## 快速开始

### 1. 安装

```bash
uv sync --all-extras --dev
```

### 2. 跑测试（零 API Key）

```bash
uv run pytest -q    # 303 passed
```

### 3. 配 LLM

在仓库根目录建 `.env`（只需一行 API Key）：

```bash
# 所有 LLM / Embedding / Rerank 共用一个 key（默认 LLM_API_KEY）
LLM_API_KEY=sk-xxxxxxxx
```

模型名、base_url 等**全部写在 `config.yaml` 中**，`.env` 只放 secret。示例：

```yaml
# configs/deepseek_vanilla.yaml
llm:
  model: deepseek-chat
  base_url: https://api.deepseek.com

embedding:
  model: text-embedding-3-small
  base_url: https://api.openai.com/v1
  dim: 1536

rerank:
  enabled: false
  # model: bge-reranker-v2-m3
  # base_url: http://localhost:9997/v1
```

所有 provider 走 OpenAI 兼容协议（`/v1/chat/completions`、`/v1/embeddings`、`/rerank`），同一套客户端代码适配 DeepSeek / OpenAI / Ollama / vLLM / SiliconFlow 等任意兼容端点。

### 4. 端到端 demo

```bash
uv run python main.py infer \
    --config configs/deepseek_colla_rag_only.yaml \
    --dataset mock_wiki --limit 10

uv run python eval.py \
    --predictions output/mock_wiki/deepseek_colla_rag_only/ \
    --metrics em,f1,rouge_l
```

### 5. 真实数据集实验（HotpotQA / 2Wiki / MuSiQue）

详见 **[docs/experiments/experiment-guide.md](docs/experiments/experiment-guide.md)**，核心三步：

```bash
uv run python main.py ingest --config configs/deepseek_colla_rag_only.yaml --dataset musique --limit 500 --rebuild
uv run python main.py infer  --config configs/deepseek_colla_rag_only.yaml --dataset musique --limit 500
uv run python eval.py --predictions output/musique/deepseek_colla_rag_only --metrics em,f1,rouge_l
```

---

## 两大创新点

### 创新点 1 - AdaGraph（作用于构建阶段）

通过 `plugins.adagraph.enabled: true` 启用，取代 3 个默认插槽：

| 默认 | AdaGraph | 解决的问题 |
|---|---|---|
| 定长 Chunker | 自适应 Chunker | 根据文本复杂度 + 实体密度自适应分块大小 |
| 单层 Extractor | 4 层 Extractor | L1 显式 / L2 隐式 / L3 模式 / L4 跨句 |
| 无裁剪 Pruner | 双重冗余 Pruner | 相似性 + 传递性双重剪枝 |

详见 [docs/innovation-1/adagraph.md](docs/innovation-1/adagraph.md)

### 创新点 2 - CollaRAG 多Agent架构（作用于查询阶段）

通过 `plugins.colla_rag.enabled: true` 启用。采用**多Agent协作架构**：

```
query → Preprocessor Agent → Tree-Intent Classifier
              │
     ┌────────┼────────┬────────────┬──────────┐
  greeting  single   multi_hop  summarization  other
     │       hop       │           │            │
     ▼       ▼         ▼           ▼            ▼
  LLM直答 SingleHop MultiHop   Summariz-  Other
          Agent    Agent       ation Agent Agent
                 (Plan-Execute
                  -Reflect)
```

| 意图类别 (5 类) | 路由到的 Agent | 典型问法 |
|---|---|---|
| `greeting` | LLM 直答（短路） | "你好" / "Hello" |
| `single_hop` | SingleHop Agent | "爱因斯坦的出生地？" |
| `multi_hop` | MultiHop Agent（Plan-Execute-Reflect） | "比较 A 和 B 的贡献差异" |
| `summarization` | Summarization Agent | "总结这篇文档的主要观点" |
| `other` | Other Agent | "今天东京天气怎样？" |

**核心组件**：
- **22 个共享工具**：向量检索 / BM25 / 混合检索(RRF) / 图邻居展开 / 图路径搜索 / 实体提取 / 关系提取 / 实体链接 / 子查询分解 / 证据评估 / 重排序 / 查询改写 / 指代消解 / 记忆读写 / 联网搜索 / 文件 IO
- **双层记忆**：SharedMemory（QA 缓存 + 路由统计 + 实体别名）+ AgentPrivateMemory（按 Agent × 意图分桶的 JSONL 经验存储）
- **设计模式**：Strategy / Factory / Builder / Template Method / Facade

详见 [docs/innovation-2/design.md](docs/innovation-2/design.md)

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
  adagraph.*             colla_rag.*
```

依赖方向硬约束（由测试守护）：
1. `core/` 不允许 import `defaults/` 或 `plugins/`
2. `plugins/adagraph/` 与 `plugins/colla_rag/` 互不依赖
3. 所有跨边界调用只走 `interfaces/*` ABC + `registry`

详见 [docs/architecture_overview.md](docs/architecture_overview.md)

---

## 配置文件

`configs/` 提供 6 份 yaml 预设：

| 文件 | AdaGraph | CollaRAG | 用途 |
|---|:---:|:---:|---|
| `default.yaml` | — | — | 全字段注释模板 |
| `deepseek_vanilla.yaml` | off | off | 裸 KG-RAG 基线 |
| `deepseek_adagraph_only.yaml` | on | off | 仅测创新点 1 |
| `deepseek_colla_rag_only.yaml` | off | on | 仅测创新点 2（云端 DeepSeek） |
| `deepseek_full.yaml` | on | on | 全启用 |
| `ollama_colla_rag_only.yaml` | off | on | 仅测创新点 2（本地 Ollama，免 API 费用） |

---

## 项目结构

```
chimera-rag/
├── main.py                       # CLI 入口 (ingest/query/infer/export/serve)
├── eval.py                       # 离线打分（EM/F1/ROUGE-L）
├── configs/                      # 5 份 yaml 预设
├── data/                         # 数据集
├── docs/                         # 设计文档 + 实验指南
│
├── src/chimera_rag/
│   ├── __init__.py               # ChimeraRAG 门面
│   ├── cli.py                    # 5 个子命令
│   ├── core/                     # types / config / registry / pipeline
│   ├── interfaces/               # ABC 接口层
│   ├── defaults/                 # vanilla 实现
│   ├── providers/                # LLM/Embedding/Rerank: OpenAI-compatible 客户端
│   ├── stores/                   # NetworkX + FAISS
│   ├── plugins/
│   │   ├── adagraph/             # 创新点 1：自适应图谱构建
│   │   └── colla_rag/         # 创新点 2：多Agent意图感知检索
│   │       ├── classifier.py     #   树形意图分类器
│   │       ├── orchestrator.py   #   MultiAgentRetriever 编排器
│   │       ├── agent_factory.py  #   AgentFactory + ToolSetBuilder
│   │       ├── agents/           #   5 个专职 Agent
│   │       ├── tools/            #   22 个共享工具
│   │       └── memory/           #   双层记忆系统
│   ├── datasets/                 # 数据集 loader
│   ├── evaluation/               # 推理运行器 + 打分器
│   └── web/                      # FastAPI + React 静态
│
└── tests/                        # 280+ 测试
```

---

## Web 系统

```bash
uv run python main.py serve --config configs/deepseek_full.yaml --dataset mock_wiki
# http://localhost:8000
```

| 路径 | 用途 |
|---|---|
| `/api/health` | 健康检查 + 插件状态 |
| `/api/ingest` | 摄取数据集或内联文档 |
| `/api/query` | 单次问答 |
| `/api/graph` | 图谱可视化与统计 |
| `/api/plugins` | 插件热切换 |
| `/api/config` | 在线读/写配置 |
| `/api/evaluate` | 批量评测入口 |

---

## 文档索引

| 文档 | 说明 |
|---|---|
| [docs/experiments/experiment-guide.md](docs/experiments/experiment-guide.md) | **实验指南**：三阶段工作流、数据集格式、操作手册 |
| [docs/architecture_overview.md](docs/architecture_overview.md) | 架构：注册表、插槽、依赖注入、管线数据流 |
| [docs/innovation-1/adagraph.md](docs/innovation-1/adagraph.md) | AdaGraph 3 个子模块实装细节 |
| [docs/innovation-2/colla-rag.md](docs/innovation-2/colla-rag.md) | CollaRAG 多Agent架构实装细节 |
| [docs/innovation-2/design.md](docs/innovation-2/design.md) | 多Agent架构设计文档 |

---

## License

[MIT](./LICENSE)
