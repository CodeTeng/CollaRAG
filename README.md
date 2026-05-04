# Chimera-RAG

> 基于知识图谱的可插拔 RAG 框架，以希腊神话融合体为名，象征两大创新点的协同：**AdaGraph**（自适应知识图谱构建）+ **IntentAgent**（意图感知智能体）。通过 `config.yaml` 一键开关，可完全降级为基线也可完全激活。

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-242%20passing-brightgreen.svg)](./tests)

---

## 亮点

- **完全可插拔**：两大创新点作为独立 plugins，关闭后框架降级为可运行的 vanilla KG-RAG 基线。
- **注册表驱动**：`@register(slot, name)` 装饰器 + `config.<slot>.active` 查表，核心层不 import 任何插件。
- **14 个抽象接口**：chunker / extractor / pruner / intent_classifier / retriever / strategy / reranker / reflector / memory / generator / llm_provider / embedding_provider / graph_store / vector_store。
- **端到端可运行**：CLI 六子命令 + FastAPI 后端 + React 前端，单端口部署。
- **TDD**：242 个测试，所有生产代码由失败测试驱动；零真实 API 依赖（MockProviders）。
- **uv 包管理**：`uv.lock` 提交，可复现构建。

## 命名映射

| 当前命名 | 原设计材料 |
|---------|-----------|
| Chimera-RAG | KG-RAG / Agent-Graph-RAG |
| **AdaGraph** | DEKER（知识构建双重增强） |
| **IntentAgent** | MRDE（Memory-Reasoning-Decision-Execution） |

## 架构

```
                     ┌──────────────────────────────────────────────┐
                     │              ChimeraRAG (facade)             │
                     └───────────────┬──────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼────────┐          ┌────────▼────────┐         ┌─────────▼────────┐
│ IngestionPL    │          │ QueryPipeline   │         │  PipelineState   │
│ chunker→extr→  │          │ intent→retr→gen │         │  (graph+vector)  │
│ pruner         │          │                 │         │                  │
└───────┬────────┘          └────────┬────────┘         └──────────────────┘
        │                            │
   ┌────▼───────────────────┐   ┌────▼──────────────────┐
   │  defaults.* (vanilla)  │   │  intent_agent.* (opt) │
   │  adagraph.*    (opt)   │   │                       │
   └────────────────────────┘   └───────────────────────┘
```

## 快速开始

### 1. 安装

```bash
# 使用 uv：https://github.com/astral-sh/uv
uv sync --all-extras --dev
```

### 2. 运行测试（MockProviders，无需 API key）

```bash
uv run pytest -q
# 242 passed
```

### 3. CLI 使用

```bash
# 任一配置都能端到端跑通
uv run python main.py ingest --config configs/vanilla.yaml        --dataset sample
uv run python main.py ingest --config configs/adagraph_only.yaml  --dataset sample
uv run python main.py ingest --config configs/intent_agent_only.yaml --dataset sample
uv run python main.py ingest --config configs/full.yaml           --dataset sample

# 单次问答（Mock 响应）
uv run python main.py query --config configs/full.yaml "Compare Nobel and Einstein"

# 启动 Web 系统（FastAPI + React）
uv run python main.py serve --config configs/full.yaml
# 浏览器访问 http://localhost:8000
```

### 4. 运行 demo 脚本

```bash
uv run python examples/demo_vanilla.py
uv run python examples/demo_adagraph.py
uv run python examples/demo_intent_agent.py  # 演示 5 种意图 → 5 种策略路由
uv run python examples/demo_full.py
```

### 5. 前端开发（可选）

```bash
cd web-ui
pnpm install
pnpm dev            # Vite 5173，/api 自动代理到 FastAPI 8000
```

```bash
# 构建到 FastAPI 静态目录，实现单端口部署
pnpm build
rm -rf ../src/chimera_rag/web/static
mkdir -p ../src/chimera_rag/web/static
cp -r dist/* ../src/chimera_rag/web/static/
```

## 创新点一：AdaGraph

作用于**知识构建阶段**。通过 `plugins.adagraph.enabled: true` 启用。

| 子模块 | 角色 | 实装状态 |
|--------|------|--------|
| `DynamicChunker` | 自适应粒度分块（复杂度 + 实体密度） | ✅ 完整 |
| `LayeredTripleExtractor` | 4 层三元组提取（L1-L4） | L1 ✅；L2/L3/L4 接口就绪 |
| `DualRedundancyPruner` | 相似性 + 传递性冗余剪枝 | 相似性 ✅；传递性接口就绪 |

核心公式：

```
adaptive_size = base - (score - 0.5) × (max_size - min_size)
   score = w_c × complexity + w_d × density      （默认 w_c = w_d = 0.5）
```

## 创新点二：IntentAgent

作用于**查询阶段**。通过 `plugins.intent_agent.enabled: true` 启用。

四阶段：**Memory → Reasoning → Decision → Execution**。

| 意图（6 类） | 映射的策略 |
|-------------|----------|
| `factual` | `direct` |
| `analytical` | `iterative` |
| `comparative` | `parallel` |
| `multi_hop` | `multi_hop` |
| `exploratory` | `expansion` |
| `follow_up` | `context_aware` |

3 个可插拔 Hooks：

- `HeuristicReflectionHook` — 启发式答案质量评估
- `TokenOverlapRerankerHook` — Jaccard 重排（cross-encoder 接口占位）
- `ParallelRetrievalPool` — asyncio Semaphore 限流子检索池

## FastAPI 后端（7 个路由组）

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/health` | 健康检查 + 插件状态 |
| POST | `/api/ingest` | 摄取数据集或内联文档 |
| POST | `/api/query` | 单次问答，返回 answer + intent + strategy + 证据 |
| GET | `/api/graph` | 图谱可视化（可按 entity 筛选） |
| GET | `/api/graph/stats` | 图谱统计（按 layer 分层） |
| GET/POST | `/api/plugins(/toggle)` | 列出 / 热切换插件 |
| GET/PUT | `/api/config` | 读取 / 深合并更新配置（敏感字段脱敏） |
| POST | `/api/evaluate` | 批量评测数据集 |

## 数据集

内置 5 个 loader：

| 名称 | 描述 |
|-----|------|
| `plain_text` | 纯文本文件，按空行分段 |
| `generic_jsonl` | 通用 JSONL，支持字段映射 |
| `hotpotqa` | [HotpotQA](https://hotpotqa.github.io/) 原生格式 |
| `two_wiki` | [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop)（与 HotpotQA 同 schema） |
| `musique` | [MusiqueQA](https://github.com/StonyBrookNLP/musique) JSONL |

数据下载见 `data/README.md`（该目录被 gitignore，大文件不入库）。

## 评测

`Evaluator` 支持 EM / F1（SQuAD 归一化）。命令行：

```bash
uv run python main.py evaluate --config configs/full.yaml --dataset hotpotqa --limit 100
uv run python main.py benchmark --configs configs/vanilla.yaml,configs/full.yaml --dataset hotpotqa
```

Web UI 的 `/evaluate` 页面提供可视化对比。

## 项目结构

```
chimera-rag/
├── main.py                        # CLI 入口
├── pyproject.toml                 # uv + 所有 extras
├── uv.lock                        # 锁定依赖
├── configs/                       # 5 份 yaml 预设
├── docs/                          # 计划文档 + 名称映射
├── my/                            # 原始研究材料（vendored）
├── src/chimera_rag/
│   ├── __init__.py                # ChimeraRAG 门面
│   ├── cli.py                     # 六子命令实现
│   ├── core/                      # types / config / registry / pipeline / ...
│   ├── interfaces/                # 14 个 ABC
│   ├── defaults/                  # 9 个 vanilla 实现
│   ├── providers/                 # MockLLM + MockEmbedding + factories
│   ├── stores/                    # NetworkX + FAISS + Neo4j(占位)
│   ├── plugins/
│   │   ├── adagraph/              # 创新点 1
│   │   └── intent_agent/          # 创新点 2（含 6 strategies + 3 hooks）
│   ├── datasets/                  # 5 个 loader + registry
│   ├── evaluation/                # metrics + Evaluator
│   └── web/                       # FastAPI app + 7 routers
├── web-ui/                        # React + Vite + Tailwind
├── examples/                      # 4 份 demo
└── tests/                         # 242 个测试
```

## 依赖方向约束（严格）

1. `core/` **不**允许 import `defaults/` 或 `plugins/`（单向依赖）
2. `plugins/adagraph/` 与 `plugins/intent_agent/` 相互**不**依赖
3. 所有跨边界调用通过 `interfaces/*` ABC + `registry` 完成
4. `ChimeraRAG` 门面是唯一允许 import 所有层的节点

## 测试矩阵

```bash
# 按功能分类
uv run pytest tests/test_core_*.py         -q  # 核心层
uv run pytest tests/test_interfaces_*.py   -q  # 抽象接口合约
uv run pytest tests/test_providers_*.py    -q  # LLM/Embedding mock
uv run pytest tests/test_stores_*.py       -q  # NetworkX/FAISS
uv run pytest tests/test_defaults_*.py     -q  # vanilla 实现
uv run pytest tests/test_adagraph_*.py     -q  # 创新点 1
uv run pytest tests/test_intent_agent_*.py -q  # 创新点 2
uv run pytest tests/test_pipeline_*.py     -q  # 4 种部署端到端
uv run pytest tests/test_web_*.py          -q  # 7 个路由组
uv run pytest tests/test_datasets*.py      -q  # 数据集加载
uv run pytest tests/test_evaluation_*.py   -q  # 评测
uv run pytest tests/test_examples_run.py   -q  # 4 个 demo 脚本
```

## 开发约定

- **TDD**：RED（失败测试） → GREEN（最小实现） → REFACTOR → COMMIT
- **细粒度提交**：每个 TDD 循环一次 commit，总计 45+ 个细分 commits
- **uv**：新增依赖用 `uv add <pkg>` / `uv add --optional <extra> <pkg>` / `uv add --dev <pkg>`
- **ruff + black**：配置在 pyproject.toml

## 许可

MIT License.

## 致谢

参考文献见 `docs/plan.md` 与 `my/` 目录下的原始设计材料。
