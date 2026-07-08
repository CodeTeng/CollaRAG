# 架构总览：Chimera-RAG 如何承载双创新点

> 本文档基于 `src/chimera_rag/` 实际实装整理。目的是让读者先看清"**注册表 + 插槽 + yaml 开关**"这套骨架是如何设计的，以及两个创新点如何以**纯插件形态**挂载其上。

---

## 1. 项目定位

**Chimera-RAG** 是一个从零自研的、基于知识图谱的可插拔 RAG 框架：

1. **一个可独立运行的裸 KG-RAG 基线**：固定分块 + LLM 显式三元组提取 + NetworkX 图存储 + FAISS 向量索引 + 直接检索 + LLM 生成。
2. **两个研究创新点作为可选插件**：
   - **AdaGraph**（知识构建阶段增强）
   - **CollaRAG**（查询阶段多Agent智能检索）
3. **一套注册表驱动的依赖注入机制**：创新点开/关只改一行 yaml，`core/` 代码不动。
4. **一个 FastAPI + React 的演示系统**：单端口部署。
5. **一个 TDD 体系**：全部实现由失败测试驱动，所有测试默认 `MockProvider`，CI 零 API key 依赖。

---

## 2. 分层架构

| 层 | 目录 | 职责 | 允许 import |
|---|---|---|---|
| **入口层** | `main.py` + `src/chimera_rag/cli.py` | 子命令的 argparse 分发 | 全部 |
| **门面层** | `src/chimera_rag/__init__.py` 的 `ChimeraRAG` | 组装所有插槽、暴露 `ingest/query/stats/toggle_plugin` | 全部 |
| **核心层** | `src/chimera_rag/core/` | `types / config / registry / pipeline / logging / exceptions / env_loader` | 仅 `interfaces/` |
| **接口层** | `src/chimera_rag/interfaces/` | ABC 接口 | 仅 `core/types` |
| **默认层** | `src/chimera_rag/defaults/` | 裸实现，都以 `defaults.<name>` 注册 | `core/` + `interfaces/` |
| **插件层** | `src/chimera_rag/plugins/adagraph/` 与 `plugins/colla_rag/` | 两个创新点，均以 `<plugin>.*` 注册 | `core/` + `interfaces/` + `defaults/`（仅复用，不反向引用） |

**依赖方向**：`core/` **绝不** import `defaults/` 和 `plugins/`，由 `tests/test_plugin_toggle_regression.py` 守护。

---

## 3. 核心接口

`src/chimera_rag/interfaces/` 下每个 ABC 对应 `registry.py` 中的一个 slot 名：

| 接口文件 | slot 名 | 默认实现 | AdaGraph 覆盖 | CollaRAG 覆盖 |
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

**AdaGraph 占 3 个构建插槽，CollaRAG 占 2 个查询插槽**，彼此不重叠。

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
  → Preprocessor Agent（查询改写/指代消解/追问拼接）
  → Tree-Intent Classifier
      ├─ greeting      → LLM 直答（短路）
      ├─ single_hop    → SingleHop Agent（混合检索 → 重排 → 质量检查 → 生成）
      ├─ multi_hop     → MultiHop Agent（Plan-Execute-Reflect：规划 → 逐步 ReAct → 反思综合）
      ├─ summarization → Summarization Agent（检索 → Map 摘要 → Reduce 综合）
      └─ other         → Other Agent（联网搜索 → 生成）
  → 后处理（写入记忆、trace 元数据、QA 缓存）
  → RetrievalResult
```

---

## 6. CollaRAG 多Agent内部架构

### 6.1 共享工具层（22 个工具）

| 分类 | 工具 |
|---|---|
| **检索** | `vector_search` · `bm25_search` · `hybrid_search` · `graph_neighbors` · `graph_path_search` · `graph_community_search` |
| **实体** | `entity_extract` · `relation_extract` · `entity_link` |
| **推理** | `sub_query_decompose` · `assess_evidence` · `rerank` |
| **预处理** | `query_rewrite` · `coreference_resolve` · `followup_merge` |
| **记忆** | `read_session_context` · `write_session_context` · `read_shared_memory` · `write_shared_memory` · `read_agent_memory` · `write_agent_memory` |
| **Web** | `web_search` · `web_fetch` |
| **IO** | `save_to_file` · `load_from_file` |

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
| `web_search` | | | | | ✅ |
| `web_fetch` | | | | | ✅ |
| `save/load_from_file` | | ✅ | ✅ | ✅ | ✅ |
| `read/write_*_memory` | ✅ | ✅ | ✅ | ✅ | ✅ |

### 6.3 双层记忆架构

```
.chimera_memory/
├── shared/                         # SharedMemory（JSON-backed）
│   ├── routing_stats.json          #   query→intent→agent→quality 路由统计
│   ├── qa_cache.json               #   高质量 QA 对缓存（同 query 直接返回）
│   ├── entity_aliases.json         #   实体别名表
│   └── rewrite_templates.json      #   成功的改写 pattern
├── multi_hop/experience/            # MultiHop Agent 私有记忆
│   └── multi_hop.jsonl             #   按意图分桶的 JSONL 经验
├── single_hop/experience/          # SingleHop Agent 私有记忆
│   └── single_hop.jsonl
├── summarization/experience/       # Summarization Agent 私有记忆
│   └── summarization.jsonl
└── other/experience/               # Other Agent 私有记忆
    └── other.jsonl
```

### 6.4 设计模式

| 模式 | 应用位置 |
|---|---|
| **Strategy** | 每个 Agent 实现 `BaseAgent.do_execute()` |
| **Factory** | `AgentFactory.create(intent, all_tools, ...)` 映射意图到 Agent 类 |
| **Builder** | `ToolSetBuilder.for_agent(agent_type, all_tools)` 按矩阵过滤工具 |
| **Template Method** | `BaseAgent.execute()` 定义 `_pre → do → _post` 骨架 |
| **Facade** | `MultiAgentRetriever` 对外暴露 `retrieve()`，内部完成编排 |
| **Registry** | `@register` 装饰器 + 全局 Registry 单例 |

---

## 7. 四种部署组合

| 文件 | 插件开关 |
|---|---|
| `deepseek_vanilla.yaml` | AdaGraph=off，CollaRAG=off |
| `deepseek_adagraph_only.yaml` | AdaGraph=on，CollaRAG=off |
| `deepseek_colla_rag_only.yaml` | AdaGraph=off，CollaRAG=on |
| `deepseek_full.yaml` | 两个都 on |

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
             │        └─── AdaGraph 占据 3 个构建插槽       │   │ 22 Tools│
             │                                             │   │ 2-Layer │
             │   PipelineState                             │   │ Memory  │
             │   ├─ NetworkXGraphStore                     │   └─────────┘
             │   ├─ FAISSVectorStore                       │
             │   └─ chunk_lookup                           │
             └─────────────────────────────────────────────┘
```

- **AdaGraph** 只改变"图长什么样"；
- **CollaRAG** 只改变"在图上怎么查"。

它们通过共享 `PipelineState` 产生**正交但可叠加**的效果。

---

## 10. 相关文档

- `docs/innovation-1/adagraph.md` — 创新点一的 3 个子模块实装细节；
- `docs/innovation-2/colla-rag.md` — 创新点二的多Agent架构实装细节；
- `docs/innovation-2/design.md` — 创新点二多Agent架构设计文档；
- `docs/experiments/` — 实验结果与实验设计文档；
- `README.md` — 仓库根目录的用户级入门文档。
