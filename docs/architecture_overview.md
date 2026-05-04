# 架构总览：Chimera-RAG 如何承载双创新点

> 本文档根据 `docs/plan.md` 与 `src/chimera_rag/` 实际实装整理，不依赖历史设计材料。目的是让读者在进入两个创新点的细节文档之前，先看清"**注册表 + 插槽 + yaml 开关**"这套骨架是如何设计的，以及两个创新点如何以**纯插件形态**挂载其上。

---

## 1. 项目定位

**Chimera-RAG** 是一个从零自研的、基于知识图谱的可插拔 RAG 框架，由 `docs/plan.md` 定义为以下五件事的集合：

1. **一个可独立运行的裸 KG-RAG 基线**：固定分块 + LLM 显式三元组提取 + NetworkX 图存储 + FAISS 向量索引 + 直接检索 + LLM 生成。
2. **两个研究创新点作为可选插件**：
   - **AdaGraph**（知识构建阶段增强，plan.md 中标注为原 DEKER）
   - **IntentAgent**（查询阶段智能体，plan.md 中标注为原 MRDE）
3. **一套注册表驱动的依赖注入机制**：创新点开/关只改一行 yaml，`core/` 代码不动。
4. **一个 FastAPI + React 的演示系统**：单端口部署，包含 ingest/查询/图谱/评测/插件开关/配置编辑/追踪七个页面。
5. **一个 TDD 体系**：全部实现由失败测试驱动，所有测试默认 `MockProvider`，CI 零 API key 依赖。

项目名取自希腊神话融合体 Chimera，对应"知识构建 × 智能检索"两个头并存（plan.md §实施策略）。

---

## 2. 分层架构

`docs/plan.md` 的架构图（Mermaid）在实际代码中落地为如下六层：

| 层 | 目录 | 职责 | 允许 import |
|---|---|---|---|
| **入口层** | `main.py` + `src/chimera_rag/cli.py` | 六个子命令的 argparse 分发 | 全部 |
| **门面层** | `src/chimera_rag/__init__.py` 的 `ChimeraRAG` | 组装所有插槽、暴露 `ingest/query/stats/toggle_plugin` | 全部 |
| **核心层** | `src/chimera_rag/core/` | `types / config / registry / pipeline / logging / exceptions / env_loader` | 仅 `interfaces/` |
| **接口层** | `src/chimera_rag/interfaces/` | 14 个 ABC（见下表） | 仅 `core/types` |
| **默认层** | `src/chimera_rag/defaults/` | 9 个裸实现，都以 `defaults.<name>` 注册 | `core/` + `interfaces/` |
| **插件层** | `src/chimera_rag/plugins/adagraph/` 与 `plugins/intent_agent/` | 两个创新点，均以 `<plugin>.<name>` 注册 | `core/` + `interfaces/` + `defaults/`（仅复用，不反向引用） |

**依赖方向（plan.md §实施注意事项 第 1 条）**：`core/` **绝不** import `defaults/` 和 `plugins/`，这是保证"关闭插件即降级"契约的前提，由 `tests/test_plugin_toggle_regression.py` 守护。

---

## 3. 14 个抽象接口

`src/chimera_rag/interfaces/` 下每个 ABC 对应 `registry.py` 中的一个 slot 名（plan.md §关键代码结构）：

| 接口文件 | slot 名 | 典型默认实现（`defaults.*`） | 两创新点覆盖 |
|---|---|---|---|
| `chunker.py` | `chunker` | `defaults.fixed` | **AdaGraph** → `adagraph.dynamic` |
| `extractor.py` | `extractor` | `defaults.simple_llm` | **AdaGraph** → `adagraph.layered` |
| `pruner.py` | `pruner` | `defaults.noop` | **AdaGraph** → `adagraph.dual_redundancy` |
| `intent_classifier.py` | `intent_classifier` | `defaults.rule` | **IntentAgent** → `intent_agent.llm` |
| `retriever.py` | `retriever` | `defaults.direct` | **IntentAgent** → `intent_agent.strategy_selector` |
| `strategy.py`（async） | `strategy` | — | **IntentAgent** → `intent_agent.{direct/multi_hop/parallel/iterative/expansion/context_aware}` |
| `reflector.py`（async） | `reflector` | `defaults.noop` | **IntentAgent** → `intent_agent.heuristic` |
| `reranker.py`（async） | `reranker` | `defaults.noop` | **IntentAgent** → `intent_agent.token_overlap` |
| `memory.py` | `memory` | `defaults.none` | **IntentAgent** → `intent_agent.session` |
| `generator.py` | `generator` | `defaults.prompt` | — |
| `llm_provider.py`（async） | `llm_provider` | `mock / openai / deepseek / ollama` | — |
| `embedding_provider.py` | `embedding_provider` | `mock / openai / ollama / sentence_transformer` | — |
| `graph_store.py` | `graph_store` | `networkx`（`neo4j` 占位） | — |
| `vector_store.py` | `vector_store` | `faiss`（flat_ip / flat_l2 / hnsw / ivf_flat） | — |

> 表中标出的"两创新点覆盖"列即本文档的切入点：**AdaGraph 占 3 个插槽，IntentAgent 占 6 个插槽**，彼此不重叠，这就是两者天然正交的结构性原因。

---

## 4. 注册表如何工作

### 4.1 装饰器写法（`src/chimera_rag/core/registry.py`）

```python
@register("chunker", "adagraph.dynamic")
class DynamicChunker(BaseChunker):
    ...

@register("retriever", "intent_agent.strategy_selector")
class IntentAgentRetriever(BaseRetriever):
    ...
```

`Registry` 是一个 `slot → name → class` 的二级字典，提供 `register / get / get_or_default / list_slot` 四个方法。`register` 对同名重复注册**主动抛 `ValueError`**，而非静默覆盖——这个防御性选择让调试更直接。

### 4.2 动态 import 触发注册

`ChimeraRAG.__init__` 在解析 yaml 之后，按 `config.plugins.<name>.enabled` 决定**是否** `importlib.import_module("chimera_rag.plugins.<name>")`（源码：`src/chimera_rag/__init__.py::_load_enabled_plugins`）。这一步只有成功 import，`@register` 装饰器的副作用才会把插件类注册进表。

这样得到三条硬性保障（plan.md §实施注意事项 第 2 条）：

1. **关闭插件 = 不 import** → 插件所需的可选依赖缺失也不会报错；
2. **关闭插件 + yaml 仍指向插件名** → `Registry.get_or_default` 打 warning 后回退 `defaults.*`；
3. **开启插件 = 装饰器副作用跑完** → yaml 里的 `adagraph.*` / `intent_agent.*` 名字都能被解析。

### 4.3 门面装配

`ChimeraRAG._build()`（facade 内部）顺序做六件事：

1. 触发插件动态 import（上一步）；
2. 构建 LLM / Embedding Provider（`providers/__init__.py::build_*`）；
3. 构建 GraphStore / VectorStore（NetworkX / FAISS）；
4. 对六个 slot（chunker / extractor / pruner / intent_classifier / retriever / generator）分别 `registry.get(slot, config.<...>.active)` 拿到类；
5. 用 `_filter_params` 根据每个类的 `__init__` 签名过滤参数，避免 "fixed → dynamic" 切换时 yaml 里多余字段报 `TypeError`；
6. 组装 `IngestionPipeline` 和 `QueryPipeline` 两条管线。

热切换（`toggle_plugin(name, enabled)`）复用了同样的 `_build()`，但**保留 `PipelineState`（graph + vector + chunk_lookup）**——这正是 plan.md §实施注意事项 第 4 条要求的"切换不需要重新 ingest"。

---

## 5. 两条管线的数据流

源码：`src/chimera_rag/core/pipeline.py`

### 5.1 IngestionPipeline

```
documents
  → chunker.chunk()          ← 默认 defaults.fixed；AdaGraph 改为 adagraph.dynamic
  → embedder.encode(...)     → vector_store.add()
  → extractor.extract()      ← 默认 defaults.simple_llm；AdaGraph 改为 adagraph.layered
  → pruner.prune()           ← 默认 defaults.noop；AdaGraph 改为 adagraph.dual_redundancy
  → graph_store.add_triple()
```

返回统计：`{"documents": N, "chunks": M, "triples_raw": R, "triples_pruned": P}`。

### 5.2 QueryPipeline

```
query
  → intent_classifier.classify()   ← 默认 defaults.rule；IntentAgent 改为 intent_agent.llm
  → retriever.retrieve(top_k)      ← 默认 defaults.direct；IntentAgent 改为 intent_agent.strategy_selector
                                     （后者内部再串联 Memory / Reasoning / Decision / Execution + hooks）
  → generator.generate()           ← defaults.prompt
  → Answer（带 intent_label / strategy_name / trace.retrieval_metadata）
```

两条管线共享 `PipelineState`（`graph_store + vector_store + chunk_lookup`），但**方法签名彼此无依赖**——ingest 完全可以用配置 A 跑完，再把 query 切换成配置 B（热切换场景）。

---

## 6. 四种部署组合

`configs/` 目录下提供了 5 份 yaml：

| 文件 | 对应 plan.md 规划 | 插件开关 |
|---|---|---|
| `default.yaml` | 全字段注释参考模板 | 不直接用于运行 |
| `deepseek_vanilla.yaml` | 裸 KG-RAG 基线 | AdaGraph=off，IntentAgent=off |
| `deepseek_adagraph_only.yaml` | 仅创新点一 | AdaGraph=on，IntentAgent=off |
| `deepseek_intent_only.yaml` | 仅创新点二 | AdaGraph=off，IntentAgent=on |
| `deepseek_full.yaml` | 全启用 | 两个都 on |

四种组合共用同一份 `QueryPipeline.run()` 代码路径，差异**只由 yaml 的 `active` 字段驱动**。这是消融实验可信度的工程基础：同数据集、同评测逻辑，只比较配置差异带来的 EM / F1 / ROUGE-L 变化。

`main.py benchmark --configs a.yaml,b.yaml,c.yaml --dataset <name>` 子命令会顺序跑所有配置（源码：`cli.py::cmd_benchmark`），产出对比数据。

---

## 7. 数据契约（`src/chimera_rag/core/types.py`）

所有跨层/跨模块传递的结构都用 Pydantic v2 锁定（plan.md §目录结构 第 169 行）：

```
Document / Chunk / Triple / Entity                        ← 构建阶段
Query / Intent / ReasoningPlan / RetrievalResult / Answer ← 查询阶段
QualityFeedback                                            ← 反馈循环
QAExample / EvalResult                                    ← 评测
```

关键字段对两创新点特别重要：

- `Triple.layer: Literal["L1","L2","L3","L4"]` — 标注来自 AdaGraph 的哪一层抽取；
- `Intent.label: Literal[6 项]` — 严格约束 IntentAgent 的 6 类意图；
- `Answer.trace: dict` — 承载检索元数据，供 `/trace` 页面还原决策过程。

---

## 8. Web 系统（单端口部署）

后端（`src/chimera_rag/web/`）由 FastAPI + `create_app(crag)` 工厂生成，路由组与 plan.md §核心功能 第 7 条要求一致：

| 路由 | 用途 |
|---|---|
| `POST /api/ingest` | 触发 IngestionPipeline（AdaGraph 作用点） |
| `POST /api/query` | 触发 QueryPipeline（IntentAgent 作用点） |
| `GET /api/graph` + `/api/graph/stats` | 图谱可视化数据源 |
| `GET/POST /api/plugins(/toggle)` | 列出 / 热切换两个插件 |
| `GET/PUT /api/config` | 在线读/写 yaml，api_key 字段脱敏 |
| `POST /api/evaluate` | 触发 Evaluator |
| `GET /api/health` | 健康检查 + 激活插件状态 |

前端（`web-ui/`）共 7 个页面与上述路由一一对应：`IngestPage / QueryPage / GraphPage / EvaluatePage / PluginsPage / ConfigPage / TracePage`。构建产物 `pnpm build` 后拷贝到 `src/chimera_rag/web/static/`，FastAPI 通过 `StaticFiles(html=True)` 挂载到 `/`，dev 模式下 Vite 起 5173 端口并 proxy `/api` 到 8000（plan.md §实施注意事项 第 3 条）。

---

## 9. 创新点在整个系统里的位置

一张图概括两个创新点如何挂到骨架上：

```
             ┌──────────── ChimeraRAG (facade) ────────────┐
             │                                             │
             │   IngestionPipeline        QueryPipeline    │
             │        ▲                        ▲           │
             │        │                        │           │
  chunker ──►│   chunker slot           intent_classifier ─│◄── IntentAgent
  extractor ►│   extractor slot         retriever slot ───│◄── IntentAgent
  pruner   ─►│   pruner slot            generator slot     │
             │        ▲                                    │
             │        │                                    │
             │        └─── AdaGraph 占据 3 个构建插槽       │
             │                                             │
             │   PipelineState                             │
             │   ├─ NetworkXGraphStore                     │
             │   ├─ FAISSVectorStore                       │
             │   └─ chunk_lookup                           │
             └─────────────────────────────────────────────┘
```

- **AdaGraph** 只改变"图长什么样"；
- **IntentAgent** 只改变"在图上怎么查"。

它们通过共享 `PipelineState` 产生**正交但可叠加**的效果，这是下一章两份细节文档的共同前提。

---

## 10. 相关文档

- `docs/plan.md` — 最初的实施计划，本文档的上游"事实来源"；
- `docs/innovation_1_adagraph.md` — 创新点一的 3 个子模块实装细节；
- `docs/innovation_2_intent_agent.md` — 创新点二的 M/R/D/E 四阶段 + 6 策略 + 3 hook 实装细节；
- `README.md` — 仓库根目录的用户级入门文档，不在 `docs/` 下。
