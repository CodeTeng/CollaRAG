# Chimera-RAG 🧩

> 🏛️ **基于知识图谱的可插拔 RAG 框架**，以希腊神话融合体 *Chimera* 为名，象征两大创新点的协同：**AdaGraph**（自适应图谱构建）+ **IntentAgent**（意图感知智能体）。通过 `config.yaml` 一键开关，可完全降级为 vanilla 基线，也可完全激活双创新点。

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11+-blue.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-315%20passing-brightgreen.svg">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
  <img alt="uv" src="https://img.shields.io/badge/pkg-uv-orange.svg">
</p>

---

## 📖 目录

- [✨ 这个项目是什么](#-这个项目是什么)
- [🚀 快速开始](#-快速开始)
- [🧠 两大创新点](#-两大创新点)
- [🏗️ 架构一览](#️-架构一览)
- [⚙️ 配置文件](#️-配置文件)
- [🧪 推理 & 评测（两阶段工作流）](#-推理--评测两阶段工作流)
- [🌐 Web 系统](#-web-系统)
- [📂 项目结构](#-项目结构)
- [🧭 开发 & 测试](#-开发--测试)
- [📄 文档索引](#-文档索引)

---

## ✨ 这个项目是什么

**一句话**：一个自研的、**双创新点可独立消融**的知识图谱 RAG 系统，覆盖从图谱构建、意图感知检索、到在线 Web 演示的完整链路。

### 它能做什么

| 场景 | 能力 |
|---|---|
| 📚 **语料 → 图谱** | 文本文档自动 chunk、LLM 抽三元组、图谱 + 向量双路存储 |
| 💬 **问题 → 答案** | 6 类意图识别 → 6 种检索策略路由 → 基于证据的生成 |
| 🧭 **消融对比** | 同一数据集下 `vanilla / adagraph_only / intent_only / full` 四组对比 |
| 📊 **离线评测** | EM / F1 / ROUGE-L + 耗时分布 + Token 分布 + 多配置对比报告（含图表） |
| 🖥️ **单端口部署** | FastAPI + React 构建在一起，`serve` 起一条命令即可 |

### 为什么要写这个

实验室常见做法是"挑一个 RAG 框架，改改提示词就叫创新"。本项目反过来：
1. **先定义清楚两个正交的创新点** 各改造哪一层；
2. **再搭一个注册表驱动的骨架**，让两个创新点作为可插拔 plugin 单独上/下线；
3. **最后用 TDD 把它实现出来**，确保每次改动不破坏"关闭插件即降级到可运行基线"这条硬契约。

这样消融实验天然严谨：同数据集、同评测逻辑、同管线代码路径，**差异只由 yaml 的插槽 active 字段驱动**。

---

## 🚀 快速开始

### 1️⃣ 安装

```bash
# 需要 uv: https://github.com/astral-sh/uv
uv sync --all-extras --dev
```

### 2️⃣ 跑测试（零 API Key 依赖，纯 Mock Provider）

```bash
uv run pytest -q
# 315 passed ✅
```

### 3️⃣ 配 LLM（仅真实推理时需要）

在仓库根目录建 `.env`：

```bash
# DeepSeek（推荐，国内可直连，价格便宜）
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-xxxxxxxx
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
```

或使用 OpenAI / Ollama —— 见 `configs/default.yaml` 里的 `llm:` 节注释。

### 4️⃣ 端到端跑一条命令

```bash
# 构图 + 推理（首次调用该 config 会自动构图一次）
uv run python main.py infer \
    --config configs/deepseek_intent_only.yaml \
    --dataset mock_wiki \
    --limit 10

# 打分 + 报告
uv run python eval.py \
    --predictions output/mock_wiki/deepseek_intent_only/ \
    --metrics em,f1,rouge_l
```

打开 `output/mock_wiki/deepseek_intent_only/report.md` —— 里面包含：
- 推理阶段 + 构图阶段 + 总计三栏的耗时与 Token 消耗
- EM / F1 / ROUGE-L
- 耗时分布（avg / min / max / P50 / P95）
- Token 分布（prompt / completion / total 各自 sum / avg / P50 / P95）
- 按意图分组、策略分布、Top-10 最差 / 最好样本
- 全部样本折叠表

---

## 🧠 两大创新点

### 创新点 1️⃣ · AdaGraph — 让图谱"长得更聪明" <sub>作用于构建阶段</sub>

通过 `plugins.adagraph.enabled: true` 启用。它取代 3 个默认插槽：

| 默认 (`defaults.*`) | AdaGraph (`adagraph.*`) | 解决的问题 |
|---|---|---|
| `fixed` Chunker（定长 300） | 🔄 `dynamic` Chunker | 根据文本**复杂度 + 实体密度**自适应分块大小 |
| `simple_llm` Extractor（L1 显式三元组） | 🧱 `layered` Extractor | 4 层抽取：L1 显式 / L2 隐式 / L3 模式 / L4 跨句 |
| `noop` Pruner（不裁剪） | ✂️ `dual_redundancy` Pruner | 相似性 + 传递性双重冗余剪枝 |

核心自适应公式：

```
adaptive_size = base − (score − 0.5) × (max_size − min_size)
       score = w_c · complexity + w_d · density      (默认 w_c = w_d = 0.5)
```

📎 详见 [`docs/innovation_1_adagraph.md`](docs/innovation_1_adagraph.md)

### 创新点 2️⃣ · IntentAgent — 让检索"按意图分道" <sub>作用于查询阶段</sub>

通过 `plugins.intent_agent.enabled: true` 启用。四阶段：**Memory → Reasoning → Decision → Execution**。

| 意图类别 (6 类) | 路由到的策略 | 典型问法 |
|---|---|---|
| 🎯 `factual` | `direct` | "X 是什么？" |
| 🔬 `analytical` | `iterative` | "为什么 X 会导致 Y？" |
| ⚖️ `comparative` | `parallel` | "比较 A 和 B" |
| 🧩 `multi_hop` | `multi_hop` | "收购了 X 雇主的是哪家？" |
| 🔭 `exploratory` | `expansion` | "关于 X 你能告诉我什么？" |
| 🗣️ `follow_up` | `context_aware` | "那它和前面的 Y 有关系吗？" |

3 个可插拔 Hooks：

- 🪞 **HeuristicReflectionHook** — 答案质量启发式评估（LLM-free）
- 🔀 **TokenOverlapRerankerHook** — Jaccard 重排（cross-encoder 接口就绪）
- 🧵 **ParallelRetrievalPool** — asyncio `Semaphore` 限流子检索池

📎 详见 [`docs/innovation_2_intent_agent.md`](docs/innovation_2_intent_agent.md)

---

## 🏗️ 架构一览

```
                    ┌────────────────────────────────────────┐
                    │          ChimeraRAG (facade)           │
                    └──────────┬─────────────────────────────┘
                               │
      ┌────────────────────────┼───────────────────────────┐
      │                        │                           │
┌─────▼──────┐         ┌───────▼───────┐          ┌────────▼────────┐
│IngestionPL │         │ QueryPipeline │          │  PipelineState  │
│chunk→extr  │         │intent→retr→gen│          │ graph + vector  │
│→ prune     │         │               │          │ + chunk_lookup  │
└─────┬──────┘         └───────┬───────┘          └─────────────────┘
      │                        │
 ┌────▼──────────────┐    ┌────▼────────────────┐
 │ defaults.*  (baseline) │ defaults.*    │
 │ adagraph.*  (plug-in1) │ intent_agent.*│
 └────────────────────────┴───────────────┘
```

**依赖方向硬约束**（由 `tests/test_plugin_toggle_regression.py` 守护）：

1. `core/` ❌ 不允许 import `defaults/` 或 `plugins/`
2. `plugins/adagraph/` 与 `plugins/intent_agent/` ❌ 互不依赖
3. 所有跨边界调用 ✅ 只走 `interfaces/*` ABC + `registry`

📎 骨架细节（注册表、插槽、依赖注入）详见 [`docs/architecture_overview.md`](docs/architecture_overview.md)

---

## ⚙️ 配置文件

`configs/` 提供 **5 份 yaml 预设**：

| 文件 | AdaGraph | IntentAgent | 用途 |
|---|:---:|:---:|---|
| `default.yaml` | — | — | 🗒️ 全字段注释模板，不直接运行 |
| `deepseek_vanilla.yaml` | ❌ | ❌ | 📉 裸 KG-RAG 基线 |
| `deepseek_adagraph_only.yaml` | ✅ | ❌ | 🔬 仅测创新点 1 |
| `deepseek_intent_only.yaml` | ❌ | ✅ | 🔬 仅测创新点 2 |
| `deepseek_full.yaml` | ✅ | ✅ | 🚀 全启用 |

四组消融配置共享同一份管线代码，差异**只由插槽 active 字段驱动**。

---

## 🧪 推理 & 评测（两阶段工作流）

评测被拆成**完全解耦**的两步：

```
┌──────────────────────┐                ┌──────────────────────┐
│   main.py infer      │ ──落盘──►      │      eval.py         │
│   跑 pipeline        │ JSONL + meta   │   纯打分 + 报告       │
│   消耗 LLM Token     │                │   0 LLM 调用          │
└──────────────────────┘                └──────────────────────┘
```

### 好处

- 💰 推理贵、打分便宜，**推理只需跑一次**，指标迭代 / 报告重生成零 Token 成本
- 🔍 **meta.json 自描述**：config 路径、数据集、起止时间、构图消耗、推理消耗都分开记录
- 📊 **报告清晰区分**「构图一次性消耗」vs「推理消耗」

### 阶段 1️⃣ · 推理（`main.py infer`）

```bash
uv run python main.py infer \
    --config configs/deepseek_intent_only.yaml \
    --dataset mock_wiki \
    --limit 50
# → output/mock_wiki/deepseek_intent_only/
#     ├── predictions.jsonl   每行一条 QA 预测
#     └── meta.json            运行元信息 + 构图/推理双份 usage
```

**首次**使用某 config + dataset 组合时会自动构图一次（会在 stderr 打印提示）；再次 infer 会复用已持久化的图与向量，跳过构图。

### 阶段 2️⃣ · 打分 + 报告（`eval.py`）

**单目录模式**：

```bash
uv run python eval.py \
    --predictions output/mock_wiki/deepseek_intent_only/ \
    --metrics em,f1,rouge_l
# → report.md / report.json / report.csv
```

指标来源优先级：`--metrics` > `--config.evaluation.metrics` > 默认 `em,f1,rouge_l`。

**多目录对比模式**（自动触发）：

```bash
uv run python eval.py \
    --predictions output/mock_wiki/deepseek_vanilla/ \
                  output/mock_wiki/deepseek_intent_only/ \
                  output/mock_wiki/deepseek_full/ \
    --out output/mock_wiki/compare/ \
    --metrics em,f1,rouge_l
# → compare.md / compare.json / compare.csv
#   + compare_bar.png       并排柱图
#   + compare_radar.png     指标雷达图
```

### 报告里有什么 📝

| 章节 | 内容 |
|---|---|
| 一、总览 | 推理耗时/Token/调用 + **构图耗时/Token/调用**（一次性） + 合计三栏对比 |
| 二、耗时分布 | avg / min / max / **P50 / P95** |
| 三、Token 分布 | prompt / completion / total 各自 sum / avg / P50 / P95 |
| 四、按意图分组 | 每类意图下的 EM/F1/ROUGE-L + 平均延迟 + 平均 Token |
| 五、策略分布 | IntentAgent 实际命中各策略的次数 |
| 六/七、Top-10 最差/最好 | 按 F1 排序，快速定位 bad case |
| 八、全部样本 | 折叠表格，含每条 QA 的预测、参考、指标、耗时、Token |

---

## 🌐 Web 系统

```bash
uv run python main.py serve --config configs/deepseek_full.yaml
# http://localhost:8000
```

后端 FastAPI + 前端 React 打包到 `src/chimera_rag/web/static/`，单端口部署。

### 7 个路由组

| 方法 | 路径 | 用途 |
|---|---|---|
| `GET` | `/api/health` | 💓 健康检查 + 插件状态 |
| `POST` | `/api/ingest` | 📥 摄取数据集或内联文档 |
| `POST` | `/api/query` | 💬 单次问答（带 intent / strategy / evidence） |
| `GET` | `/api/graph` + `/stats` | 🕸️ 图谱可视化与分层统计 |
| `GET`/`POST` | `/api/plugins(/toggle)` | 🔌 插件热切换（无需重启） |
| `GET`/`PUT` | `/api/config` | 📝 在线读/写配置（api_key 自动脱敏） |
| `POST` | `/api/evaluate` | 📊 Web UI 的批量评测入口 |

### 前端开发

```bash
cd web-ui
pnpm install
pnpm dev              # Vite 5173，/api 自动代理到 FastAPI 8000

# 构建到 FastAPI 静态目录
pnpm build
rm -rf ../src/chimera_rag/web/static
mkdir -p ../src/chimera_rag/web/static
cp -r dist/* ../src/chimera_rag/web/static/
```

---

## 📂 项目结构

```
chimera-rag/
├── main.py                       # CLI 入口 (ingest/query/infer/export/serve)
├── eval.py                       # 离线打分脚本
├── pyproject.toml                # uv + 全部 extras
├── uv.lock                       # 锁定依赖
│
├── configs/                      # 5 份 yaml 预设
├── data/mock_wiki/               # 配套小语料（25 chunk + 10 QA）
├── docs/                         # 架构 + 双创新点设计文档 + experiments/ 实验设计
├── web-ui/                       # React + Vite + Tailwind
│
├── src/chimera_rag/
│   ├── __init__.py               # ChimeraRAG 门面
│   ├── cli.py                    # 5 个子命令
│   ├── core/                     # types / config / registry / pipeline / ...
│   ├── interfaces/               # 14 个 ABC
│   ├── defaults/                 # 9 个 vanilla 实现
│   ├── providers/                # LLM/Embedding: openai/deepseek/ollama/st/mock
│   ├── stores/                   # NetworkX + FAISS (+ Neo4j 占位)
│   ├── plugins/
│   │   ├── adagraph/             # 🧱 创新点 1
│   │   └── intent_agent/         # 🧠 创新点 2 (6 strategies + 3 hooks)
│   ├── datasets/                 # plain_text/jsonl/hotpotqa/2wiki/musique/mock_wiki
│   ├── evaluation/               # InferenceRunner + Scorer + EvalReporter + CompareReporter
│   └── web/                      # FastAPI app + 7 routers + static/
│
└── tests/                        # 315 个测试（TDD 驱动）
```

---

## 🧭 开发 & 测试

### 测试矩阵

```bash
uv run pytest tests/test_core_*.py          -q   # 🧬 核心层
uv run pytest tests/test_interfaces_*.py    -q   # 📐 抽象接口合约
uv run pytest tests/test_providers_*.py     -q   # 🔌 LLM / Embedding
uv run pytest tests/test_stores_*.py        -q   # 💾 NetworkX / FAISS
uv run pytest tests/test_defaults_*.py      -q   # 📦 vanilla 实现
uv run pytest tests/test_adagraph_*.py      -q   # 🧱 创新点 1
uv run pytest tests/test_intent_agent_*.py  -q   # 🧠 创新点 2
uv run pytest tests/test_pipeline_*.py      -q   # 🔁 4 种部署组合端到端
uv run pytest tests/test_web_*.py           -q   # 🌐 FastAPI 路由
uv run pytest tests/test_datasets*.py       -q   # 📚 数据集加载
uv run pytest tests/test_evaluation_*.py    -q   # 📊 评测与报告
uv run pytest tests/test_cli_*.py           -q   # 🖥️ CLI 子命令
uv run pytest tests/test_eval_script.py     -q   # 🧪 eval.py
```

### 代码规范

```bash
uv run ruff check .              # lint
uv run ruff format .             # format
```

### 开发约定

- **TDD**：RED（失败测试） → GREEN（最小实现） → REFACTOR → COMMIT
- **细粒度提交**：每个 TDD 循环一次 commit
- **uv 依赖**：`uv add <pkg>` / `uv add --optional <extra> <pkg>` / `uv add --dev <pkg>`
- **三条硬红线**（由 CI 守护）：
  1. `core/` 不 import `defaults/` / `plugins/`
  2. 两个 plugin 不互相 import
  3. 关闭插件时系统仍能端到端跑通（`test_plugin_toggle_regression`）

---

## 📄 文档索引

| 文档 | 说明 |
|---|---|
| [`docs/architecture_overview.md`](docs/architecture_overview.md) | 🏗️ 骨架：注册表、插槽、依赖注入、管线数据流 |
| [`docs/innovation_1_adagraph.md`](docs/innovation_1_adagraph.md) | 🧱 AdaGraph 3 个子模块实装细节 |
| [`docs/innovation_2_intent_agent.md`](docs/innovation_2_intent_agent.md) | 🧠 IntentAgent M/R/D/E + 6 策略 + 3 hook |
| [`docs/experiments/`](docs/experiments/) | 🧪 实验设计文档目录（含创新点 2 实验：消融对比 / 按意图分组 / SOTA 横向对比） |
| [`docs/README.md`](docs/README.md) | 📑 文档目录 |
| [`configs/README.md`](configs/README.md) | ⚙️ 5 份 yaml 的用途与切换方式 |

---

## 📜 License

[MIT](./LICENSE)

---

<p align="center">
  <sub>Built with 🦀 <a href="https://github.com/astral-sh/uv">uv</a> · 🧪 pytest · ⚡ FastAPI · ⚛️ React · 🎨 Tailwind</sub>
</p>
