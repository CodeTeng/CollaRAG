# Chimera-RAG

> 基于知识图谱的可插拔 RAG 框架，以希腊神话融合体 *Chimera* 为名，象征两大创新点的协同：**AdaGraph**（自适应图谱构建）+ **IntentAgent**（意图感知智能体）。通过 `config.yaml` 一键开关，可完全降级为 vanilla 基线，也可完全激活双创新点。

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-348%20passing-brightgreen.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="uv" src="https://img.shields.io/badge/pkg-uv-orange.svg">
</p>

---

## 它能做什么

| 场景 | 能力 |
|---|---|
| 语料 → 图谱 | 文本文档自动 chunk、LLM 抽三元组、图谱 + 向量双路存储 |
| 问题 → 答案 | 6 类意图识别 → 6 种检索策略路由 → 基于证据的生成 |
| 消融对比 | 同一数据集下 `vanilla / adagraph_only / intent_only / full` 四组对比 |
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
uv run pytest -q    # 348 passed
```

### 3. 配 LLM

在仓库根目录建 `.env`（三选一）：

```bash
# DeepSeek（云端）
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# Ollama（本地）
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:8b

# OpenAI
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

### 4. 端到端 demo（mock_wiki 小数据集）

```bash
# 一条命令搞定（mock_wiki 只有 25 篇文档，自动建图秒级完成）
uv run python main.py infer \
    --config configs/deepseek_intent_only.yaml \
    --dataset mock_wiki --limit 10

# 打分
uv run python eval.py \
    --predictions output/mock_wiki/deepseek_intent_only/ \
    --metrics em,f1,rouge_l
```

### 5. 真实数据集实验（HotpotQA / 2Wiki / MuSiQue）

真实数据集有几万篇语料，**必须先手动 ingest 再 infer**（demo 用的 mock_wiki 语料极小，infer 会自动建图，所以跳过了 ingest）。

完整流程和原理说明见 **[docs/experiment_guide.md](docs/experiment_guide.md)**，核心三步：

```bash
# 1. 建图（--limit 500 = 只取前 500 条 QA 及其附带的全部 paragraph 作为语料）
uv run python main.py ingest --config configs/ollama_intent_only.yaml --dataset musique --limit 500 --rebuild

# 2. 推理（500 题问答）
uv run python main.py infer  --config configs/ollama_intent_only.yaml --dataset musique --limit 500

# 3. 评测
uv run python eval.py --predictions output/musique/ollama_intent_only --metrics em,f1,rouge_l
```

> 为什么 mock_wiki 不用 ingest？为什么真实数据集需要 `--limit`？数据集格式是怎样的？—— 全部在 [实验指南](docs/experiment_guide.md) 里。

---

## 两大创新点

### 创新点 1 - AdaGraph（作用于构建阶段）

通过 `plugins.adagraph.enabled: true` 启用，取代 3 个默认插槽：

| 默认 | AdaGraph | 解决的问题 |
|---|---|---|
| 定长 Chunker | 自适应 Chunker | 根据文本复杂度 + 实体密度自适应分块大小 |
| 单层 Extractor | 4 层 Extractor | L1 显式 / L2 隐式 / L3 模式 / L4 跨句 |
| 无裁剪 Pruner | 双重冗余 Pruner | 相似性 + 传递性双重剪枝 |

详见 [docs/innovation_1_adagraph.md](docs/innovation_1_adagraph.md)

### 创新点 2 - IntentAgent（作用于查询阶段）

通过 `plugins.intent_agent.enabled: true` 启用。四阶段：Memory → Reasoning → Decision → Execution。

| 意图类别 (6 类) | 路由到的策略 | 典型问法 |
|---|---|---|
| `factual` | `direct` | "X 是什么？" |
| `analytical` | `iterative` | "为什么 X 会导致 Y？" |
| `comparative` | `parallel` | "比较 A 和 B" |
| `multi_hop` | `multi_hop` | "收购了 X 雇主的是哪家？" |
| `exploratory` | `expansion` | "关于 X 你能告诉我什么？" |
| `follow_up` | `context_aware` | "那它和前面的 Y 有关系吗？" |

进阶演化：
- **2b - LangGraph 动态编排**：LLM Agent 自主选择并组合多个检索工具
- **2c - 经验型记忆**：跨 session 持久化成功路径 + 反思教训

详见 [docs/innovation_intent_agent.md](docs/innovation_intent_agent.md)

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
│→extract    │         │               │          │ + chunk_lookup  │
│→prune      │         │               │          │                 │
└────────────┘         └───────────────┘          └─────────────────┘
      ↑                        ↑
  defaults.*              defaults.*
  adagraph.*             intent_agent.*
```

依赖方向硬约束（由测试守护）：
1. `core/` 不允许 import `defaults/` 或 `plugins/`
2. `plugins/adagraph/` 与 `plugins/intent_agent/` 互不依赖
3. 所有跨边界调用只走 `interfaces/*` ABC + `registry`

详见 [docs/architecture_overview.md](docs/architecture_overview.md)

---

## 配置文件

`configs/` 提供 6 份 yaml 预设：

| 文件 | AdaGraph | IntentAgent | 用途 |
|---|:---:|:---:|---|
| `default.yaml` | — | — | 全字段注释模板 |
| `deepseek_vanilla.yaml` | - | - | 裸 KG-RAG 基线 |
| `deepseek_adagraph_only.yaml` | + | - | 仅测创新点 1 |
| `deepseek_intent_only.yaml` | - | + | 仅测创新点 2 |
| `deepseek_intent_langgraph.yaml` | - | + | 创新点 2b 动态编排 |
| `deepseek_full.yaml` | + | + | 全启用 |
| `ollama_intent_only.yaml` | - | + | 本地 Qwen3-8B 实验 |

详见 [configs/README.md](configs/README.md)

---

## 推理 & 评测

```
  main.py ingest    →    storage/ (图谱+向量)
                              ↓ 复用
  main.py infer     →    output/ (predictions.jsonl)
                              ↓
  eval.py           →    report.md (EM/F1/ROUGE-L)
```

- **ingest**：语料 → chunk → embedding + LLM 抽三元组 → 知识图谱（慢，只跑一次）
- **infer**：逐题 意图分类 → 检索 → 生成答案（中速，复用图谱）
- **eval**：对比 prediction vs reference（纯计算，秒级）

详细说明、参数速查、产物路径约定见 **[docs/experiment_guide.md](docs/experiment_guide.md)**

### 阶段 3️⃣ · RAGAS 语义评测（可选）

除了 EM / F1 / ROUGE-L 等词面匹配指标，还可使用 [RAGAS](https://docs.ragas.io/) 进行**语义级评测**，计算四个指标：

| 指标 | 含义 |
|---|---|
| **Faithfulness** | 回答是否忠于检索到的上下文（不编造） |
| **Answer Relevancy** | 回答与问题的语义相关度 |
| **Context Precision** | 检索结果中有多少是和正确答案相关的 |
| **Context Recall** | 正确答案所需的信息在检索结果中覆盖了多少 |

```bash
# 1. 安装 RAGAS 依赖（可选依赖组）
uv sync --extra ragas --extra chunk

# 2. 重跑 infer 以产出 evidence_texts（RAGAS 的 context 类指标需要检索原文）
uv run python main.py infer \
    --config configs/deepseek_intent_only.yaml \
    --dataset mock_wiki --limit 10

# 3. 运行 RAGAS 评测
uv run python eval_ragas.py \
    --predictions output/mock_wiki/deepseek_intent_only/ \
    --config configs/deepseek_intent_only.yaml

# 产出：ragas_report.md / ragas_report.json / ragas_report.csv
```

RAGAS 的 judge LLM 和 embedding 均复用 `config.yaml` 中的 `llm` / `embedding` 配置（以及 `.env` 中的 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`），无需额外配参。

> **⚠️ `ragas` 与 `agent` extra 不能同时安装**
>
> RAGAS 经 `langchain-community` 依赖 `langchain-core < 1.0`，而 `agent` extra（LangGraph 1.x）要求 `langchain-core >= 1.0`，两者对 `langchain-core` 的版本要求直接冲突。项目已通过 `[tool.uv] conflicts` 将二者声明为互斥 extra，uv 会分开解析：
>
> ```bash
> # 使用 RAGAS 评测时
> uv sync --extra ragas --extra chunk --extra st --extra llm --extra vec
>
> # 使用 LangGraph Agent 时
> uv sync --extra agent --extra st --extra llm --extra vec
> ```
>
> 两套 extra 各自独立安装，不可混用 `--extra ragas --extra agent`。

#### LangChain 分块器（可选）

新增基于 `RecursiveCharacterTextSplitter` 的分块器，注册名 `defaults.langchain`，在 `config.yaml` 中切换：

```yaml
ingestion:
  chunker:
    active: defaults.langchain   # 替代 defaults.fixed
    params:
      chunk_size: 500
      chunk_overlap: 50
```

安装分块器依赖：`uv sync --extra chunk`

---

## Web 系统

```bash
uv run python main.py serve --config configs/deepseek_full.yaml --dataset mock_wiki
# http://localhost:8000
```

后端 FastAPI + 前端 React 打包到 `src/chimera_rag/web/static/`，单端口部署。

| 路径 | 用途 |
|---|---|
| `/api/health` | 健康检查 + 插件状态 |
| `/api/ingest` | 摄取数据集或内联文档 |
| `/api/query` | 单次问答 |
| `/api/graph` | 图谱可视化与统计 |
| `/api/plugins` | 插件热切换 |
| `/api/config` | 在线读/写配置 |
| `/api/evaluate` | 批量评测入口 |

前端开发：`cd web-ui && pnpm install && pnpm dev`

---

## 项目结构

```
chimera-rag/
├── main.py                       # CLI 入口 (ingest/query/infer/export/serve)
├── eval.py                       # 离线打分（EM/F1/ROUGE-L）
├── configs/                      # 7 份 yaml 预设
├── data/                         # 数据集（mock_wiki / hotpotqa / two_wiki / musique）
├── scripts/                      # 实用脚本（LLM-only baseline / ollama 重启 / 数据下载）
├── docs/                         # 设计文档 + 实验指南
├── paper/                        # 论文结果梳理 + 对比估算
├── storage/                      # 持久化图谱产物
│
├── src/chimera_rag/
│   ├── __init__.py               # ChimeraRAG 门面
│   ├── cli.py                    # 5 个子命令
│   ├── core/                     # types / config / registry / pipeline
│   ├── interfaces/               # 14 个 ABC
│   ├── defaults/                 # 9 个 vanilla 实现
│   ├── providers/                # LLM/Embedding: openai/deepseek/ollama/st/mock
│   ├── stores/                   # NetworkX + FAISS
│   ├── plugins/
│   │   ├── adagraph/             # 创新点 1
│   │   └── intent_agent/         # 创新点 2 + 2b + 2c
│   ├── datasets/                 # 数据集 loader
│   ├── evaluation/               # 推理运行器 + 打分器 + 报告生成器
│   └── web/                      # FastAPI + React 静态
│
└── tests/                        # 348 个测试
```

---

## 开发 & 测试

```bash
uv run pytest -q                  # 全量测试
uv run ruff check .               # lint
uv run ruff format .              # format
```

开发约定：TDD 驱动，细粒度提交，三条硬红线（core 不导 plugins、两 plugin 不互导、关闭插件仍可端到端跑通）。

---

## 文档索引

| 文档 | 说明 |
|---|---|
| [docs/experiment_guide.md](docs/experiment_guide.md) | **实验指南**：三阶段工作流、数据集格式、操作手册 |
| [docs/architecture_overview.md](docs/architecture_overview.md) | 架构：注册表、插槽、依赖注入、管线数据流 |
| [docs/innovation_1_adagraph.md](docs/innovation_1_adagraph.md) | AdaGraph 3 个子模块实装细节 |
| [docs/innovation_intent_agent.md](docs/innovation_intent_agent.md) | IntentAgent：静态路由 + LangGraph 动态编排 + 经验记忆 |
| [docs/experiments/](docs/experiments/) | 实验设计文档目录 |
| [configs/README.md](configs/README.md) | yaml 预设的用途与切换方式 |
| [paper/relink_experiments_summary.md](paper/relink_experiments_summary.md) | Relink 论文实验结果梳理 |
| [paper/relink_qwen3_8b_estimate.md](docs/result/主实验.md) | Qwen3-8B 各方法 EM/F1 估算 + 实测 |

---

## License

[MIT](./LICENSE)
