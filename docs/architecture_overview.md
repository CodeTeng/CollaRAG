# 架构总览：Chimera-RAG 如何承载创新点

> 本文档基于 `src/chimera_rag/` 实际实装整理。目的是让读者先看清"**注册表 + 插槽 + yaml 开关**"这套骨架是如何设计的，以及创新点与构建侧基线如何以**纯插件形态**挂载其上。

---

## 1. 项目定位

**Chimera-RAG** 是一个从零自研的、基于知识图谱的可插拔 RAG 框架：

1. **一个可独立运行的裸 KG-RAG 基线**：固定分块 + LLM 显式三元组提取 + NetworkX 图存储 + FAISS 向量索引 + 直接检索 + LLM 生成。
2. **两个研究创新点**：
   - **创新点一：面向知识构建的新型分块方法** —— 🚧 **规划中，尚未实装**。
   - **创新点二：CollaRAG（查询阶段多Agent智能检索）** —— ✅ 已实装，内部含 4 个子贡献（C1 Query Triage / C2 G-PER / C3 图接地双层记忆 / C4 图条件化能力特化）。
3. **一个承载插槽的构建侧前置基线**：`plugins/adagraph/` 实现了自适应分块、分层提取、双重冗余剪枝的**部分**能力，作为创新点一的落点，**本身不是本仓库宣称的创新点**。
4. **一套注册表驱动的依赖注入机制**：创新点开/关只改一行 yaml，`core/` 代码不动。
5. **一个 FastAPI + React 的演示系统**：单端口部署。
6. **一个 TDD 体系**：全部实现由失败测试驱动，所有测试默认 `FakeProvider`，CI 零 API key 依赖。

---

## 2. 分层架构

| 层 | 目录 | 职责 | 允许 import |
|---|---|---|---|
| **入口层** | `main.py` + `src/chimera_rag/cli.py` | 子命令的 argparse 分发 | 全部 |
| **门面层** | `src/chimera_rag/__init__.py` 的 `ChimeraRAG` | 组装所有插槽、暴露 `ingest/query/stats/toggle_plugin` | 全部 |
| **核心层** | `src/chimera_rag/core/` | `types / config / registry / pipeline / logging / exceptions / env_loader` | 仅 `interfaces/` |
| **接口层** | `src/chimera_rag/interfaces/` | ABC 接口 | 仅 `core/types` |
| **默认层** | `src/chimera_rag/defaults/` | 裸实现，都以 `defaults.<name>` 注册 | `core/` + `interfaces/` |
| **插件层** | `src/chimera_rag/plugins/adagraph/` 与 `plugins/colla_rag/` | 构建侧前置基线 + 创新点二，均以 `<plugin>.*` 注册 | `core/` + `interfaces/` + `defaults/`（仅复用，不反向引用） |

**依赖方向**：`core/` **绝不** import `defaults/` 和 `plugins/`，由 `tests/test_plugin_toggle_regression.py` 守护。

---

## 3. 核心接口

`src/chimera_rag/interfaces/` 下每个 ABC 对应 `registry.py` 中的一个 slot 名：

| 接口文件 | slot 名 | 默认实现 | 基线覆盖 | CollaRAG 覆盖 |
|---|---|---|---|---|
| `chunker.py` | `chunker` | `defaults.fixed` | `adagraph.dynamic` | — |
| `extractor.py` | `extractor` | `defaults.simple_llm` | `adagraph.layered` | — |
| `pruner.py` | `pruner` | `defaults.noop` | `adagraph.dual_redundancy` | — |
| `intent_classifier.py` | `intent_classifier` | `defaults.rule` | — | `colla_rag.tree` |
| `retriever.py` | `retriever` | `defaults.direct` | — | `colla_rag.multi_agent` |
| `generator.py` | `generator` | `defaults.prompt` | — | — |
| `llm_provider.py` | `llm_provider` | `providers/llm.py` (OpenAI-compatible) | — | — |
| `embedding_provider.py` | `embedding_provider` | `providers/embedding.py` (OpenAI-compatible) | — | — |
| `rerank_provider.py` | `rerank_provider` | `providers/rerank.py` (OpenAI-compatible) | — | — |
| `graph_store.py` | `graph_store` | `stores/graph.py` (NetworkX) | — | — |
| `vector_store.py` | `vector_store` | `stores/vector.py` (FAISS) | — | — |
| `base_agent.py` | — | — | — | `BaseAgent` ABC + `AgentContext` |

构建侧基线占 3 个 ingestion 插槽，CollaRAG 占 2 个 query 插槽，彼此不重叠。（创新点一落地后会替换掉 `chunker` 这一插槽。）

---

## 4. 注册表如何工作

### 4.1 装饰器写法

```python
@register("chunker", "adagraph.dynamic")
class DynamicChunker(BaseChunker): ...

@register("retriever", "colla_rag.multi_agent")
class MultiAgentRetriever(BaseRetriever): ...
```

### 4.2 动态 import 触发注册

`ChimeraRAG.__init__` 在解析 yaml 之后，按 `config.plugins.<name>.enabled` 决定**是否** `importlib.import_module("chimera_rag.plugins.<name>")`。只有成功 import，`@register` 装饰器的副作用才会把插件类注册进表。

### 4.3 门面装配

`ChimeraRAG._build()` 顺序做：触发插件 import → 构建 Provider → 构建 Store → 解析 slot → 实例化 → 组装 Pipeline。

热切换（`toggle_plugin(name, enabled)`）复用 `_build()`，但**保留 `PipelineState`**——切换不需要重新 ingest。

---

## 5. 两条管线的数据流

### 5.1 IngestionPipeline

```
documents
  → chunker.chunk()          ← defaults.fixed / adagraph.dynamic
  → embedder.encode(...)     → vector_store.add()
  → extractor.extract()      ← defaults.simple_llm / adagraph.layered
  → pruner.prune()           ← defaults.noop / adagraph.dual_redundancy
  → graph_store.add_triple()
```

### 5.2 QueryPipeline

```
query
  → intent_classifier.classify()   ← defaults.rule / colla_rag.tree
  → retriever.retrieve(top_k)      ← defaults.direct / colla_rag.multi_agent
  → generator.generate()           ← defaults.prompt
  → Answer
```

当 `colla_rag.multi_agent` 启用时，`MultiAgentRetriever.retrieve()` 内部展开为：

```
query
  → Tree-Intent Classifier          ← 先分类（对原始 query）
      ├─ greeting → LLM 直答（短路返回，不进缓存/预处理）
      └─ 其余 4 类 ↓
  → QA 缓存检查        SharedMemory 文本哈希命中 → 直接返回
  → 证据子图缓存检查    C3：实体邻域重叠命中且 quality ≥ 0.6 → 直接返回
  → Preprocessor Agent（查询改写 / 指代消解 / 追问拼接）→ rewritten_query
  → 计算 GraphFootprint（C4）→ 条件化 Agent 的工具子集
  → 按意图分派
      ├─ single_hop    → SingleHop Agent（混合检索 → 重排 → 质量检查 → 生成）
      ├─ multi_hop     → MultiHop Agent（G-PER：Plan → 前沿式 Execute → 结构化 Reflect）
      ├─ summarization → Summarization Agent（检索 → Map 摘要 → Reduce 综合）
      └─ other         → Other Agent（联网搜索 → 生成）
  → 后处理（写路由统计 / QA 缓存 / 私有经验 / 图接地记忆、trace 元数据）
  → RetrievalResult
```

> ⚠️ **注意顺序**：分类发生在**预处理之前**，且分类用的是**原始 query**；只有分派给 Agent 时用的是改写后的 query（`Query(text=rewritten, ...)`）。这与"先预处理再分类"的直觉相反，改动前请先看 `orchestrator.py::retrieve()`。

---

## 6. CollaRAG 多Agent内部架构

### 6.1 共享工具层（34 个工具）

`orchestrator.py::_build_tools()` 依次调用 10 个 `build_*_tools()` 模块，**共产出 34 个工具**，**全部进入 `TOOL_MATRIX` 暴露给 Agent**。记忆类工具提供 read/write 变体；write 变体授予给实际产生对应数据的 specialist（shared：SH/MH/SM；agent-private：SH/MH/SM/OT；session：PP/MH）。

| 分类 | 模块 | 工具 |
|---|---|---|
| **检索** | `retrieval_tools.py` | `vector_search` · `bm25_search` · `hybrid_search` · `graph_neighbors` · `graph_path_search` · `graph_community_search` |
| **实体** | `entity_tools.py` | `entity_extract` · `relation_extract` · `entity_link` |
| **推理** | `reasoning_tools.py` | `sub_query_decompose` · `assess_evidence` · `rerank` |
| **预处理** | `preprocess_tools.py` | `query_rewrite` · `coreference_resolve` · `followup_merge` |
| **记忆** | `memory_tools.py` | `read_session_context` · `write_session_context` · `read_shared_memory` · `write_shared_memory` · `read_agent_memory` · `write_agent_memory` |
| **Web** | `web_tools.py` | `web_search` · `web_fetch` |
| **IO** | `io_tools.py` | `save_to_file` · `load_from_file` |
| **验证** | `verification_tools.py` | `answer_verify` · `claim_decompose` · `source_attribution` |
| **图分析** | `graph_advanced_tools.py` | `graph_subgraph_extract` · `graph_statistics` |
| **证据处理** | `evidence_tools.py` | `temporal_filter` · `evidence_dedup` · `chunk_summarize` · `confidence_calibrate` |

另有 `quality_evaluator.py` 提供 `HeuristicQualityEvaluator`（不注册为工具，作为 `reasoning_tools` 里 `assess_evidence` 的默认 reflector）。

每个 Agent 只能使用 `TOOL_MATRIX` 中指定的工具子集（由 `ToolSetBuilder` 在创建时过滤）。

### 6.2 Agent ↔ 工具可用矩阵

| 工具 | Preprocessor | SingleHop | MultiHop | Summarization | Other |
|---|:---:|:---:|:---:|:---:|:---:|
| `vector_search` | | ✅ | ✅ | ✅ | |
| `bm25_search` | | ✅ | ✅ | ✅ | |
| `hybrid_search` | | ✅ | ✅ | ✅ | ✅ |
| `graph_neighbors` | | ✅ | ✅ | | |
| `graph_path_search` | | | ✅ | | |
| `graph_community_search` | | | ✅ | ✅ | |
| `entity_extract` | | ✅ | ✅ | ✅ | ✅ |
| `relation_extract` | | | ✅ | | |
| `entity_link` | | ✅ | ✅ | | |
| `sub_query_decompose` | | | ✅ | | |
| `assess_evidence` | | ✅ | ✅ | | |
| `rerank` | | ✅ | ✅ | ✅ | |
| `query_rewrite` | ✅ | | | | |
| `coreference_resolve` | ✅ | | | | |
| `followup_merge` | ✅ | | | | |
| `read_session_context` | ✅ | | ✅ | | |
| `write_session_context` | ✅ | | ✅ | | |
| `web_search` | | | | | ✅ |
| `web_fetch` | | | | | ✅ |
| `save/load_from_file` | | ✅ | ✅ | ✅ | ✅ |
| `read_shared_memory` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `write_shared_memory` | | ✅ | ✅ | ✅ | |
| `read_agent_memory` | | ✅ | ✅ | ✅ | ✅ |
| `write_agent_memory` | | ✅ | ✅ | ✅ | ✅ |
| `answer_verify` | | ✅ | ✅ | | |
| `claim_decompose` | | | ✅ | | |
| `source_attribution` | | | ✅ | ✅ | |
| `graph_subgraph_extract` | | | ✅ | | |
| `graph_statistics` | | | ✅ | | |
| `temporal_filter` | | | ✅ | | |
| `evidence_dedup` | | ✅ | ✅ | ✅ | |
| `chunk_summarize` | | | ✅ | ✅ | |
| `confidence_calibrate` | | ✅ | ✅ | ✅ | ✅ |

**矩阵是"静态上界"，运行时会按图条件化收缩（C4）**：`ToolSetBuilder.for_agent(..., footprint)` 在传入 `GraphFootprint` 时会：

- `multi_hop` 且图贫乏（`entity_hit_rate == 0` 或 `sparse && !has_path`）→ **撤回** 5 个图遍历工具（`graph_path_search` / `graph_subgraph_extract` / `graph_neighbors` / `graph_community_search` / `graph_statistics`），**补入** `web_search` / `web_fetch`；
- `single_hop` 且实体完全不在图中（`entity_hit_rate == 0 && n_entities > 0`）→ 撤回 `graph_neighbors`；
- `footprint is None` → 完全回退到上表静态矩阵。

此外每个 Agent 有一个 KG 投影声明 `GRAPH_VIEWS`：`single_hop → attribute`、`multi_hop → path`、`summarization → community`、`preprocessor` / `other → none`。

### 6.3 双层记忆架构

基目录由 `MultiAgentRetriever(memory_path=...)` 决定，默认 `.chimera_memory/`：

```
.chimera_memory/
├── shared/                          # SharedMemory（JSON-backed）
│   ├── routing_stats.json           #   query→intent→agent 路由统计
│   ├── qa_cache.json                #   高质量 QA 对缓存（按 query 文本哈希）
│   ├── entity_aliases.json          #   实体别名表
│   ├── rewrite_templates.json       #   成功的改写 pattern
│   └── evidence_subgraphs.json      # ★ C3-3B 证据子图缓存（按实体邻域重叠检索）
├── private/                         # ★ C3 私有层（图接地）
│   ├── plan_templates.json          #   C3-3A 按 compositionality + 实体签名归档的图规划模板
│   └── gap_lessons.json             #   C3-3C G-PER Reflect 产出的类型化 gap 教训
├── multi_hop/experience/            # AgentPrivateMemory（每个 Agent × 意图一个 JSONL 桶）
│   └── multi_hop.jsonl
├── single_hop/experience/
│   └── single_hop.jsonl
├── summarization/experience/
│   └── summarization.jsonl
└── other/experience/
    └── other.jsonl
```

★ 标的三层是 C3「图接地双层记忆」新增的，与原有的文本哈希缓存**并存但不混同**：
`EvidenceSubgraphStore` 命中反映的是**同一实体邻域被结构性复用**（在非重复问答的基准上也合法），
而 `qa_cache` 命中反映的是**问题重复**。把两者分开，正是为了避免"记忆收益其实是缓存命中"的质疑。

每个 JSONL 桶内的条目带 `outcome ∈ {success, lesson}`：高质量轨迹作为**模仿样例**，低质量轨迹转成**结构化失败教训**（`orchestrator._derive_lesson()`，不再额外调 LLM），供 MultiHop 在 Plan 阶段召回、避免重蹈覆辙。

### 6.4 设计模式

| 模式 | 应用位置 |
|---|---|
| **Strategy** | 每个 Agent 实现 `BaseAgent.do_execute()` |
| **Factory** | `AgentFactory.create(intent, all_tools, ...)` 映射意图到 Agent 类 |
| **Builder** | `ToolSetBuilder.for_agent(agent_type, all_tools)` 按矩阵过滤工具 |
| **Template Method** | `BaseAgent.execute()` 定义 `_pre → do → _post` 骨架 |
| **Facade** | `MultiAgentRetriever` 对外暴露 `retrieve()`，内部完成编排 |
| **Registry** | `@register` 装饰器 + 全局 Registry 单例 |

### 6.5 G-PER：多跳 Agent 的图谱锚定推理

`agents/multi_hop.py` 不是普通的 ReAct 循环，而是 **G-PER（Graph-Grounded Plan-Execute-Reflect）**，三个阶段各自做一件 text-only 方案做不到的事：

| 阶段 | 图谱锚定的做法 | 无图时退化为 |
|---|---|---|
| **Plan** | 产出**图操作**步骤 `link / traverse / attribute / aggregate`，先过 KG schema 校验再花预算；非法步骤降级为自由子查询（不阻塞） | 纯文本子问题分解 |
| **Execute** | **前沿式共享预算**：全局迭代预算在步骤间流动，先完成的简单步骤把富余迭代捐给下游困难步骤 | 逐步固定迭代上限 |
| **Reflect** | 用 `gper/completeness.py::StructuralCompletenessChecker` 做**结构化完整性检查**——先按查询措辞推断所需证据形状（comparison / bridge / attribute），再与已收集的三元组做差，输出**类型化 gap**（`missing_attribute` / `missing_path` / `missing_relation`），每个 gap 自带定向 `repair_query`；随后做对抗性矛盾探针 | 标量质量打分（Reflexion 式） |

触发退化的开关是 `graph_store is None`（由 `plugins.colla_rag.multi_agent.<agent>.gper.graph_grounded: false` 或图缺失导致），**这个退化路径本身就是论文里的 graph-poor 消融设置**。

---

## 7. 四种部署组合

| 文件 | 插件开关 |
|---|---|
| `vanilla.yaml` | AdaGraph=off，CollaRAG=off |
| `adagraph_only.yaml` | AdaGraph=on，CollaRAG=off |
| `colla_rag_only.yaml` | AdaGraph=off，CollaRAG=on |
| `full.yaml` | 两个都 on |

四种组合共用同一份 `QueryPipeline.run()` 代码路径，差异**只由 yaml 的 `active` 字段驱动**。

---

## 8. 数据契约（`src/chimera_rag/core/types.py`）

```
Document / Chunk / Triple / Entity                                ← 构建阶段
Query / Intent / ReasoningPlan / RetrievalResult / Answer         ← 查询阶段
PreprocessResult / AgentExperienceEntry                           ← 多Agent架构
QualityFeedback                                                    ← 质量评估
MultiAgentIntentLabel = ["greeting","single_hop","multi_hop","summarization","other"]
QAExample / EvalResult                                            ← 评测
```

---

## 9. 创新点在整个系统里的位置

```
             ┌──────────── ChimeraRAG (facade) ────────────┐
             │                                             │
             │   IngestionPipeline        QueryPipeline    │
             │        ▲                        ▲           │
             │        │                        │           │
  chunker ──►│   chunker slot           intent_classifier ─│◄── TreeIntentClassifier
  extractor ►│   extractor slot         retriever slot ───│◄── MultiAgentRetriever
  pruner   ─►│   pruner slot            generator slot     │        │
             │        ▲                                    │   ┌────┴────┐
             │        │                                    │   │ 5 Agents│
             │        └─── adagraph 基线占 3 个构建插槽      │   │ 31 Tools│
             │                                             │   │ 2-Layer │
             │   PipelineState                             │   │ Memory  │
             │   ├─ NetworkXGraphStore                     │   └─────────┘
             │   ├─ FAISSVectorStore                       │
             │   └─ chunk_lookup                           │
             └─────────────────────────────────────────────┘
```

- **创新点一（规划中）** 将改变"图长什么样"；
- **CollaRAG（创新点二）** 改变"在图上怎么查"——它不只换检索器，还通过 G-PER、图接地记忆、图条件化能力特化三处**回读** KG 的结构。

它们通过共享 `PipelineState` 产生**正交但可叠加**的效果。

---

## 10. 相关文档

- `docs/innovation-1/adagraph.md` — 构建侧前置基线的 3 个子模块实装细节与 TODO 清单（**注意：它不是创新点一**）；
- `docs/innovation-2/colla-rag.md` — 创新点二的多Agent架构实装细节（含 C1–C4 四大子贡献）；
- `docs/innovation-2/design.md` — 创新点二多Agent架构设计文档；
- `docs/experiments/` — 实验结果与实验设计文档；
- `README.md` — 仓库根目录的用户级入门文档。
