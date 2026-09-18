# 创新点二：CollaRAG —— 图谱锚定的多 Agent 检索分工

> 本文是创新点二的**完整实装文档**，对应代码 `src/chimera_rag/plugins/colla_rag/`。
>
> **作用阶段**：Query（查询）
> **启用方式**：`configs/*.yaml` 中 `plugins.colla_rag.enabled: true`，`query.retriever.active` 设为 `colla_rag.multi_agent`，`query.intent_classifier.active` 设为 `colla_rag.tree`
> **论文**：*CollaRAG: Graph-Grounded Division of Labor for Multi-Agent Retrieval*（`paper/my/current/main.tex`）

## 四个子贡献（C1–C4）与代码落点

| # | 子贡献 | 核心机制 | 代码落点 |
|---|---|---|---|
| **C1** | Intent-Aware Query Triage | 两级级联分类 → 5 类专职 Agent 一次性分派 | `classifier.py` · `orchestrator.py` |
| **C2** | **G-PER**（Graph-Grounded Plan-Execute-Reflect） | Plan 出图操作并过 KG schema 校验；Execute 前沿式共享预算；Reflect 做结构化完整性检查 + 对抗性矛盾探针 | `agents/multi_hop.py` · `gper/completeness.py` |
| **C3** | Graph-Grounded Dual-Layer Memory | 共享层按**实体邻域重叠**检索证据子图；私有层存图规划模板 + 类型化 gap lessons | `memory/graph_memory.py` · `memory/memory_manager.py` |
| **C4** | Graph-Conditioned Capability Specialization | 按 `GraphFootprint`（hitRate / hasPath / density）动态收放 Agent 工具权限 | `agent_factory.py` |

**优雅退化**：C2/C3/C4 的图接地机制在 `graph_store` 缺席时各自退化为 text-only 版本（`gper.graph_grounded: false`），这条退化路径**本身就是论文的 graph-poor 消融设置**。

---

## 目录

- [1. 架构总览](#1-架构总览)
- [2. MultiAgentRetriever 编排器](#2-multiagentretriever-编排器)
- [3. Preprocessor Agent](#3-preprocessor-agent)
- [4. TreeIntentClassifier 树形分类器](#4-treeintentclassifier-树形分类器)
- [5. SingleHop Agent](#5-singlehop-agent)
- [6. MultiHop Agent（C2：G-PER）](#6-multihop-agentc2g-per)
- [7. Summarization Agent](#7-summarization-agent)
- [8. Other Agent](#8-other-agent)
- [9. Greeting 短路](#9-greeting-短路)
- [10. 共享工具层（34 个工具）](#10-共享工具层34-个工具)
- [11. 双层记忆架构（C3：图接地）](#11-双层记忆架构c3图接地)
- [12. 设计模式](#12-设计模式)
- [13. 配置开关](#13-配置开关)
- [14. 在 QueryPipeline 中的接入](#14-在-querypipeline-中的接入)
- [15. 对应测试](#15-对应测试)
- [16. 与构建侧基线的正交性](#16-与构建侧基线的正交性)
- [17. C4：图条件化能力特化](#17-c4图条件化能力特化)

---

## 1. 架构总览

CollaRAG 采用**多Agent协作架构**：一个 Preprocessor Agent 做预处理 → 一个 TreeIntentClassifier 做意图分类 → 根据意图分派到 5 个专职 Agent 之一。

```
用户 query + session_id
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   MultiAgentRetriever（顶层编排器）                │
│                                                                  │
│  ① Tree-Intent Classifier（对原始 query 分类）                     │
│     Level-1: greeting?                                           │
│     Level-2: single_hop | multi_hop | summarization | other       │
│               │                                                  │
│      greeting ┤ 其余 4 类                                         │
│        │      │                                                  │
│        ▼      ▼                                                  │
│   LLM 直答   ② QA 缓存（文本哈希）──命中──► 直接返回               │
│   (短路)     ③ 证据子图缓存（实体邻域重叠，C3-3B）──命中──► 返回    │
│               │                                                  │
│               ▼                                                  │
│  ④ Preprocessor Agent（query_rewrite / coreference_resolve /     │
│      followup_merge / read_session_context）→ rewritten_query     │
│               │                                                  │
│               ▼                                                  │
│  ⑤ 计算 GraphFootprint（C4）→ 条件化工具子集                       │
│               │                                                  │
│  ┌──┬──────┬──┴───┬────────┬─┐                                   │
│  │  │      │      │        │ │                                   │
│  ▼  ▼      ▼      ▼        ▼                                     │
│ SingleHop MultiHop Summa-  Other                                 │
│ Agent     Agent    rization Agent                                │
│           (G-PER)  Agent                                         │
│     │      │       │        │                                    │
│     └──────┴───────┴────────┘                                    │
│               │                                                  │
│               ▼                                                  │
│  ┌────────────────────────────┐                                  │
│  │  Post-Process（后处理）      │                                  │
│  │  统一 RetrievalResult       │                                  │
│  │  写路由统计 / QA 缓存 /      │                                  │
│  │  私有经验 / 图接地记忆(C3)    │                                  │
│  └────────────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────┘
```

> ⚠️ **顺序要点**：分类发生在**预处理之前**，且用的是**原始 query**；只有分派给 Agent 时传入的是改写后的 query（`Query(text=rewritten, ...)`）。改动前请先看 `orchestrator.py::retrieve()`。

**路由规则：一次性分派，不允许中途升级/转投。** 分类器决定后直接走对应 Agent，即使质量不达标也由该 Agent 自行在 max_iterations 内处理。

---

## 2. MultiAgentRetriever 编排器

`orchestrator.py::MultiAgentRetriever` 是 `BaseRetriever` 的实现，对 `QueryPipeline` 暴露单一 `retrieve()` 接口，内部完成完整编排流程。

### 注册方式

```python
@register("retriever", "colla_rag.multi_agent")
class MultiAgentRetriever(BaseRetriever):
```

### retrieve() 流程

1. **分类**：`TreeIntentClassifier.classify(query)` → `intent.label`（对**原始** query）
2. **Greeting 短路**：若 `greeting`，直接 LLM 直答返回
3. **QA 缓存检查**：`SharedMemory.get_qa_cache(md5(query))` 命中则直接返回
4. **证据子图缓存检查（C3-3B）**：`_probe_entities()` 抽大写 token 探针 → `shared_evidence.lookup()` 按实体邻域重叠命中且 `quality ≥ 0.6` 则直接返回
5. **预处理**：`PreprocessorAgent.execute(query)` → 改写后的 query
6. **计算 GraphFootprint（C4）**：`_footprint(probe_entities)` → 传入 `AgentFactory.create(..., footprint=...)` 条件化工具子集
7. **分派**：`AgentFactory.create(label, ...)` 创建对应 Agent，以 `Query(text=rewritten, ...)` 执行
8. **后处理**：合并结果、写路由统计、写 QA 缓存（`quality ≥ 0.6`）、写 Agent 私有经验、写图接地记忆（C3）、写会话历史

> 第 4/6 步是 C3/C4 引入的，都在 `orchestrator.py` 内，不改变 `BaseRetriever` 接口。

### 依赖注入

```python
MultiAgentRetriever(
    llm=...,                # LLM provider
    vector_store=...,       # FAISS 向量索引
    graph_store=...,        # NetworkX 图谱
    embedder=...,           # Embedding provider
    chunk_lookup=...,        # chunk_id → Chunk 映射
    memory_path=".chimera_memory",
    classifier_config={...},
    agent_configs={...},
)
```

构建时自动创建 `TreeIntentClassifier`、`MemoryManager`（双层记忆）、`BM25Index`，并触发 5 个 Agent 的 `@AgentFactory.register` 副作用。

---

## 3. Preprocessor Agent

**注册**：`@AgentFactory.register("preprocessor")`  
**文件**：`agents/preprocessor.py`

> ⚠️ **执行时机**：Preprocessor 在**分类之后**运行（见 §2 流程第 5 步），且它的改写结果只影响分派给专职 Agent 的 query，**不影响**意图分类与两个缓存的判据——那两步都用原始 query。

**职责**：对原始 query 做预处理，产出适合下游分类和检索的干净 query。

**可用工具**（6 个）：
- `query_rewrite`：LLM 改写 query（扩展/规范/消歧）
- `coreference_resolve`：指代消解
- `followup_merge`：追问拼接为自包含 query
- `read_session_context`：读取会话历史
- `write_session_context`：写入 QA 轮次到会话历史
- `read_shared_memory`：读取共享记忆

**流程**：`do_execute()` 中先调 `query_rewrite` 工具改写 query，结果写入 `context.trace` 的 `rewritten_query` / `original_query` 字段。

**无检索工具**，纯 LLM + 上下文工具。

---

## 4. TreeIntentClassifier 树形分类器

**注册**：`@register("intent_classifier", "colla_rag.tree")`  
**文件**：`classifier.py`

### 两级分类

```
Level-1: greeting 正则过滤（快速短路）
   ↓ 非 greeting
Level-2: LLM 分类 → single_hop | multi_hop | summarization | other
   ↓ LLM 失败
规则兜底（正则启发式）
```

### 5 类意图标签

| label | 路由到的 Agent | 典型问法 |
|---|---|---|
| `greeting` | LLM 直答（短路） | "你好" / "Hello" |
| `single_hop` | SingleHop Agent | "爱因斯坦的出生地？" |
| `multi_hop` | MultiHop Agent（Plan-Execute-Reflect） | "比较 A 和 B 的贡献差异" |
| `summarization` | Summarization Agent | "总结这篇文档的主要观点" |
| `other` | Other Agent | "今天东京天气怎样？" |

### Greeting 检测

正则匹配：`^(hi|hello|hey|你好|嗨|哈喽|good morning|...)` 或包含问候语 + 寒暄词。

### LLM 分类

用 `_CLASSIFY_PROMPT` 向 LLM 发一次 completion，要求返回单个标签。解析结果时做模糊匹配（只要包含有效标签名即可）。

### 规则兜底

- `summarization`：包含 summarize/总结/概括/归纳/overview 等关键词
- `multi_hop`：包含 compare/contrast/differ/vs/relationship 等关键词
- 默认 → `single_hop`

---

## 5. SingleHop Agent

**注册**：`@AgentFactory.register("single_hop")`  
**文件**：`agents/single_hop.py`

**适用场景**：事实性单跳查询。

**流程**（单轮、无 ReAct 循环、无反思重试）：

```
hybrid_search(query, top_k=10) → assess_evidence（一次性质量检查）→ LLM 生成
```

**可用工具**（17 个）：
- 检索：`vector_search`、`bm25_search`、`hybrid_search`、`graph_neighbors`
- 实体：`entity_extract`、`entity_link`
- 推理：`assess_evidence`、`rerank`
- 记忆：`read_shared_memory`、`write_shared_memory`、`read_agent_memory`、`write_agent_memory`
- IO：`save_to_file`、`load_from_file`
- 验证：`answer_verify`
- 证据处理：`evidence_dedup`、`confidence_calibrate`

**质量检查**：`assess_evidence` 仅做**一次性**轻量检查，不触发循环。若 `quality < quality_threshold`（默认 0.4），在 trace 中标记 `low_confidence`。

**特点**：最快、成本最低的路径。

---

## 6. MultiHop Agent（C2：G-PER）

**注册**：`@AgentFactory.register("multi_hop")`  
**文件**：`agents/multi_hop.py`

**适用场景**：多跳推理、比较、探索性查询。

**架构**（G-PER：Graph-Grounded Plan-Execute-Reflect 三阶段 + 前置准备）：

```
Phase 0: 准备（图无关）
   抽实体 → checker.infer_shape() 推断 compositionality(comparison/bridge/attribute)
   召回历史失败教训（避免重蹈覆辙）
   召回图规划模板（C3-3A）——命中则直接复用，跳过 LLM Plan

Phase 1: Plan（schema 约束的「图操作」计划）
   LLM 产出步骤，每步是一个图操作：link / traverse / attribute / aggregate
       │
       ▼
   逐个过 KG schema 校验（graph_store.schema_predicates()）
   非法步骤降级为自由子查询（计入 plan_stats["degraded"]），不阻塞

Phase 2: Execute（前沿式共享预算）
   全局预算 total_budget（默认 max_plan_steps × max_step_iterations）
   在各步骤间流动：先完成的简单步骤把富余迭代「捐」给下游困难步骤
   每步内部仍是迷你 ReAct：Thought → Action → Observation → Step Result
       │
       ▼
   （C3-3C）抢占式缺口检索：先把历史 gap lesson 的 repair edge 补进来

Phase 3: Reflect & Synthesize（结构化 + 对抗式）
   checker.check(query, entities, evidence_triples)
     ├── 按推断出的证据形状与已收集三元组做差
     ├── 输出类型化 gap：missing_attribute / missing_path /
     │                    missing_relation / missing_entity
     └── 每个 gap 自带定向 repair_query → 执行定向补检
         （最多 reflect_retries 轮，默认 2）
       │
       ▼
   标量兜底：结构化检查无 gap 但 scalar quality 仍低 → 跑一轮文本式反思
       │
       ▼
   对抗性矛盾探针：在 KG 中搜与草稿答案矛盾的关系
     └── verdict == "contested" → 带争议重综合
       │
       ▼
   Synthesize: 综合所有步骤证据 → 生成最终答案
```

**可用工具**（29 个，见 `TOOL_MATRIX["multi_hop"]`，运行时会按 `GraphFootprint` 收缩，见 §17）：
- 检索：`vector_search`、`bm25_search`、`hybrid_search`、`graph_neighbors`、`graph_path_search`、`graph_community_search`
- 图分析：`graph_subgraph_extract`、`graph_statistics`
- 实体：`entity_extract`、`relation_extract`、`entity_link`
- 推理：`sub_query_decompose`、`assess_evidence`、`rerank`
- 记忆：`read_session_context`、`write_session_context`、`read_shared_memory`、`write_shared_memory`、`read_agent_memory`、`write_agent_memory`
- IO：`save_to_file`、`load_from_file`
- 验证：`answer_verify`、`claim_decompose`、`source_attribution`
- 证据处理：`temporal_filter`、`evidence_dedup`、`chunk_summarize`、`confidence_calibrate`

**Plan 阶段**：使用 `GPER_PLAN_PROMPT`（含 KG schema 与 `{avoid_lessons}` 槽）让 LLM 产出**图操作**步骤。计划来源有三种（trace 里的 `plan_source`）：`template`（命中 C3-3A 图规划模板，直接复用、不调 LLM）→ `llm`（正常生成）→ 解析失败时降级为单步计划 `[{"step": original_query}]`。步骤数上限由 `max_plan_steps`（默认 5）控制。

**Execute 阶段**：前沿式执行，使用 `STEP_REACT_PROMPT`。每步通过正则解析 `Action: tool_name\nAction Input: {json}` 调用工具，解析 `Step Result: ...` 收集结果；前序步骤结果注入后续上下文。关键差异是**全局迭代预算 `total_budget`（默认 `max_plan_steps × max_step_iterations`）在步骤间共享**——先完成的步骤释放的迭代数可供后续困难步骤使用。

**Reflect 阶段**：主判据不再是标量质量分，而是 `gper/completeness.py::StructuralCompletenessChecker.check()` 的**结构化 gap 列表**。有 gap 且未超 `reflect_retries`（默认 2）时，用 `GPER_REFLECT_PROMPT` 为每个 gap 生成定向 `repair_query` 并补检。结构化检查为空但 scalar quality 仍低于 `quality_threshold`（默认 0.6）时，才回落到文本式反思（保留原 PER 的安全网）。最后用 `GPER_ADVERSARIAL_PROMPT` 做矛盾探针，再用 `SYNTHESIZE_PROMPT` 综合。

**Plan 阶段的失败教训规避（text-lesson fallback）**：Plan 之前，MultiHop 通过 `_recall_avoid_lessons(context.agent_memory)` 召回本桶（`multi_hop/multi_hop`）里**复发最多的失败经验**（`outcome="lesson"` 条目，按 `hit_count` 排序），渲染进 `GPER_PLAN_PROMPT` 的 `{avoid_lessons}` 槽，提示规划器**不要重复**已证实低质量的工具序列模式。这是图谱锚定 gap-lesson（3C）的**纯文本降级对应物**：graph-poor 设定下图谱缺口为空，它成为唯一可用的学习信号；graph-rich 设定下与结构化 gap-lesson 互补。召回失败或无记忆时优雅返回 `[]`，规划器无种子运行。trace 记录 `avoided_lessons` 计数。

**trace 记录**：`plan`、`plan_source`、`plan_stats`、`compositionality`、`step_results`、`iterations`（工具调用数）、`tool_call_log`、`reflect_count`、`quality`、`evidence_triples`、`adversarial_verdict`、`structural_gaps`（未解 gap，供 C3-3C 落盘）、`pre_emptive_gaps`、`avoided_lessons`。

**优雅退化**：`graph_store is None`（图缺失，或 `...gper.graph_grounded: false`）时——Plan 的 schema 变成 `"(unknown — graph store absent)"` 不做校验；`StructuralCompletenessChecker` 因 `compositionality="unknown"` 直接返回空 gap；整条链路坍缩为 text-only PER。**这条退化路径本身就是论文的 graph-poor 消融设置。**

**与 text-only ReAct / Reflexion 的关键差异**：

| 方面 | text-only ReAct / Reflexion | G-PER |
|---|---|---|
| 规划 | 隐式决策，或纯文本子问题 | 显式**图操作**列表（link/traverse/attribute/aggregate），先过 KG schema 校验 |
| 预算 | 全局单循环固定上限 | **前沿式共享预算**，步骤间可捐赠迭代 |
| 反思 | 标量质量打分（口头自省） | **结构化完整性检查** + 类型化 gap + 定向修复查询 |
| 矛盾检验 | 无 | **对抗性探针**在 KG 中搜矛盾关系 |
| 图缺失时 | 不变 | 优雅退化到 text-only 版本（即 graph-poor 消融） |

---

## 7. Summarization Agent

**注册**：`@AgentFactory.register("summarization")`  
**文件**：`agents/summarization.py`

**适用场景**：归纳总结、综述类查询。

**流程**（确定性流水线，非 ReAct 循环）：

```
hybrid_search(query, top_k=50) → Map（每个 chunk 独立 LLM 摘要）→ Reduce（合并为最终综述）
```

**可用工具**（16 个）：
- 检索：`vector_search`、`bm25_search`、`hybrid_search`、`graph_community_search`
- 实体：`entity_extract`
- 推理：`rerank`
- 记忆：`read_shared_memory`、`write_shared_memory`、`read_agent_memory`、`write_agent_memory`
- IO：`save_to_file`、`load_from_file`
- 验证：`source_attribution`
- 证据处理：`chunk_summarize`、`evidence_dedup`、`confidence_calibrate`

**Map 阶段**：对每个检索到的 chunk 独立调用 LLM 提取与 query 相关的关键信息。

**Reduce 阶段**：将所有局部摘要合并，LLM 生成最终综述答案。

**成本控制**：Map 阶段的 chunk 数量由 `retrieve_top_k` 控制（默认 50）。

---

## 8. Other Agent

**注册**：`@AgentFactory.register("other")`  
**文件**：`agents/other.py`

**适用场景**：超出知识库范围的查询。

**流程**：

```
web_search(query, max_results=5) → hybrid_search（可选补充本地）→ LLM 综合生成
```

**可用工具**（10 个）：
- 检索：`hybrid_search`
- Web：`web_search`、`web_fetch`
- 实体：`entity_extract`
- 记忆：`read_shared_memory`、`read_agent_memory`、`write_agent_memory`
- IO：`save_to_file`、`load_from_file`
- 证据处理：`confidence_calibrate`

**搜索后端**：DuckDuckGo（免费无 key），通过 `web_tools.py` 封装。

---

## 9. Greeting 短路

**不是 Agent**，而是分类器判定 `greeting` 后 Orchestrator 直接调 LLM 返回。不进入任何 Agent、不读写记忆、不产生 trace。

```python
if label == "greeting":
    greeting_text = await llm.complete(f"Respond warmly to this greeting: {query.text}")
    return RetrievalResult(metadata={"agent": "greeting", "answer_text": greeting_text})
```

---

## 10. 共享工具层（34 个工具）

`orchestrator.py::_build_tools()` 依次调用 10 个 `build_*_tools()` 模块，**共产出 34 个工具**，**全部进入 `TOOL_MATRIX` 暴露给 Agent**。记忆类工具提供 read/write 变体；write 变体授予给实际产生对应数据的 specialist（shared：SH/MH/SM；agent-private：SH/MH/SM/OT；session：PP/MH）。

所有工具统一实现为 `langchain_core.tools.StructuredTool`，参数由 Pydantic schema 校验。每个 Agent 通过 `ToolSetBuilder` 按 `TOOL_MATRIX` 过滤可用工具子集。

### 10.1 检索与实体类工具（`tools/retrieval_tools.py` + `tools/entity_tools.py`）

| 工具名 | 模块 | 描述 |
|---|---|---|
| `vector_search` | retrieval | FAISS 语义向量检索 |
| `bm25_search` | retrieval | BM25 关键词检索（rank-bm25） |
| `hybrid_search` | retrieval | RRF 融合检索（vector + BM25） |
| `graph_neighbors` | retrieval | 知识图谱 N-hop 邻居展开 |
| `graph_path_search` | retrieval | 两实体间最短路径搜索 |
| `graph_community_search` | retrieval | 基于图社区发现的主题检索 |
| `entity_extract` | entity | 从文本抽取实体（LLM） |
| `relation_extract` | entity | 从文本抽取关系三元组（LLM） |
| `entity_link` | entity | 实体链接 / 别名归一（对齐到 KG 中的规范实体） |

此外还有 `BM25Index`（从 chunk_lookup 构建）和 `ToolAccumulator`（跨工具累积 chunk 并去重合并）。

### 10.2 推理类工具（`tools/reasoning_tools.py`）

| 工具名 | 描述 |
|---|---|
| `sub_query_decompose` | LLM 将复合 query 拆解为子查询列表 |
| `assess_evidence` | 评估当前证据质量/相关性/完整性 |
| `rerank` | CrossEncoder 语义重排序 |

### 10.3 预处理类工具（`tools/preprocess_tools.py`）

| 工具名 | 描述 |
|---|---|
| `query_rewrite` | LLM 改写 query（扩展/规范/消歧） |
| `coreference_resolve` | 指代消解（代词 → 显式实体） |
| `followup_merge` | 追问与上一轮 QA 拼接为自包含 query |

### 10.4 记忆/上下文工具（`tools/memory_tools.py`）

| 工具名 | 描述 |
|---|---|
| `read_session_context` | 读取当前会话对话历史 |
| `write_session_context` | 写入当前轮 QA 到会话历史 |
| `read_shared_memory` | 从共享记忆读取（QA 缓存/路由统计/实体别名） |
| `write_shared_memory` | 写入共享记忆 |
| `read_agent_memory` | 读取当前 Agent 的私有经验记忆 |
| `write_agent_memory` | 写入当前 Agent 的私有经验记忆 |

> ⚠️ 三个 `write_*` 工具**不在 `TOOL_MATRIX` 中**——记忆写入由 `MultiAgentRetriever` 的后处理统一完成（见 §2 第 8 步），不走 Agent 的工具调用。

### 10.5 联网搜索工具（`tools/web_tools.py`）

| 工具名 | 描述 |
|---|---|
| `web_search` | DuckDuckGo 联网搜索 |
| `web_fetch` | 抓取网页内容并提取正文 |

### 10.6 通用工具（`tools/io_tools.py`）

| 工具名 | 描述 |
|---|---|
| `save_to_file` | 结果/中间数据保存到文件 |
| `load_from_file` | 从文件加载数据 |

### 10.7 验证与溯源工具（`tools/verification_tools.py`）

| 工具名 | 描述 |
|---|---|
| `answer_verify` | LLM 验证 draft answer 是否被 evidence 支持（verified/partially_verified/rejected） |
| `claim_decompose` | 将文本拆解为原子可验证命题列表 |
| `source_attribution` | 为 answer 中每个 claim 标注来源 chunk_id，生成可追溯引用链 |

### 10.8 图谱分析工具（`tools/graph_advanced_tools.py`）

| 工具名 | 描述 |
|---|---|
| `graph_subgraph_extract` | 围绕种子实体提取 k-hop 子图，LLM 生成结构化描述 |
| `graph_statistics` | 返回图谱统计信息（节点/边数、top hubs、谓词分布、密度、孤立叶节点数） |

### 10.9 证据处理工具（`tools/evidence_tools.py`）

| 工具名 | 描述 |
|---|---|
| `temporal_filter` | 按年份范围过滤 evidence chunks（正则提取年份，无日期的 chunk 保留） |
| `evidence_dedup` | Jaccard token 相似度去重，减少冗余证据 |
| `chunk_summarize` | LLM 压缩单个 chunk（保留与 query 相关的关键事实） |
| `confidence_calibrate` | 多信号加权置信度校准（证据覆盖度、query/evidence overlap、refusal 检测、长度） |

### 10.10 Agent ↔ 工具可用矩阵

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
| `read_shared_memory` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `write_shared_memory` | | ✅ | ✅ | ✅ | |
| `read_agent_memory` | | ✅ | ✅ | ✅ | ✅ |
| `write_agent_memory` | | ✅ | ✅ | ✅ | ✅ |
| `web_search` | | | | | ✅ |
| `web_fetch` | | | | | ✅ |
| `save_to_file` | | ✅ | ✅ | ✅ | ✅ |
| `load_from_file` | | ✅ | ✅ | ✅ | ✅ |
| `answer_verify` | | ✅ | ✅ | | |
| `claim_decompose` | | | ✅ | | |
| `source_attribution` | | | ✅ | ✅ | |
| `graph_subgraph_extract` | | | ✅ | | |
| `graph_statistics` | | | ✅ | | |
| `temporal_filter` | | | ✅ | | |
| `evidence_dedup` | | ✅ | ✅ | ✅ | |
| `chunk_summarize` | | | ✅ | ✅ | |
| `confidence_calibrate` | | ✅ | ✅ | ✅ | ✅ |

> 上表是**静态上界**。运行时 `ToolSetBuilder` 会按 `GraphFootprint` 动态收缩（C4，见 §17）：
> 图贫乏时 multi_hop 的 5 个图遍历工具会被撤回、并补入 `web_search` / `web_fetch`。

---

## 11. 双层记忆架构（C3：图接地）

### 11.1 共享记忆（SharedMemory）

`memory/shared_memory.py` — JSON 文件持久化，存储在 `.chimera_memory/shared/` 下。

| 存储项 | 数据结构 | 作用 |
|---|---|---|
| **路由统计** | `{query_hash → {intent, agent, quality, latency}}` | 记录每次查询的路由历史 |
| **QA 缓存** | `{query_hash → {answer, quality}}` | 同 query 命中时直接返回 |
| **实体别名表** | `{alias → canonical_name}` | 预处理 Agent 指代消解时参考 |
| **查询改写模板** | `{pattern → rewrite_template, hit_count}` | 成功的改写 pattern 沉淀 |

### 11.2 Agent 私有记忆（AgentPrivateMemory）

`memory/agent_memory.py` — 按 Agent × 意图分桶的 JSONL 文件，存储在 `.chimera_memory/<agent>/experience/<intent>.jsonl`。

```
.chimera_memory/
├── shared/
│   ├── routing_stats.json
│   ├── qa_cache.json
│   ├── entity_aliases.json
│   ├── rewrite_templates.json
│   └── evidence_subgraphs.json      # ★ C3-3B 证据子图缓存
├── private/                         # ★ C3 私有层（图接地）
│   ├── plan_templates.json          #   C3-3A 图规划模板
│   └── gap_lessons.json             #   C3-3C 类型化 gap 教训
├── single_hop/experience/
│   └── single_hop.jsonl
├── multi_hop/experience/
│   └── multi_hop.jsonl
├── summarization/experience/
│   └── summarization.jsonl
└── other/experience/
    └── other.jsonl
```

★ 三层由 `memory/graph_memory.py` 提供，详见 §11.5。

**沉淀条目**（`AgentExperienceEntry`）：

```python
{
    intent: str,              # 意图标签（分桶键）
    query_text: str,          # 原始查询
    tool_sequence: list[str], # 工具调用序列
    final_quality: float,     # 最终质量分数
    outcome: "success" | "lesson",
    lesson: str | None,       # 失败教训（仅 outcome="lesson" 时）
    hit_count: int,           # 命中计数（用于热度排序/淘汰）
    created_at: str,          # ISO 时间戳
}
```

**失败教训的写入与召回**（`outcome="lesson"`）：

- **写入**：`MultiAgentRetriever` 在每轮结束后按 `quality` 判定 `outcome`（`quality < 0.5` 记为 `lesson`），并调用 `_derive_lesson()` 从 trace 派生**结构化 lesson 文本**（不额外调用 LLM）：对 MultiHop 记录 `low_quality q=…; reflect=N; unresolved_gaps=[…]`（含 G-PER 反思产出的类型化缺口摘要），对其他 Agent 记录 `low_quality q=…; query='…'`。同一 `tool_sequence + outcome` 的条目去重合并时，`final_quality` 取最大、`hit_count` 自增、`lesson` 文本**刷新为最新一次的反思摘要**。
- **召回**：`recall()` 返回「先 success（按质量降序）后 lesson（按 hit_count 降序）」的混合列表——失败条目不再被质量排序压到底，能真正浮出。另提供 `recall_lessons()` 只返回失败条目，按 **复发次数**（hit_count）排序，作为粗粒度相似度代理：在某 (agent, intent) 桶里失败得越多的工具模式，越可能在类似 query 上重演，故优先召回以规避。

**关键修正**：旧版 `recall()` 仅按 `final_quality` 降序取 top_n，`outcome="lesson"` 的低质量条目几乎永远排不上，导致失败教训「记了不用」。新版同时浮现成功（模仿）与失败（规避）两类经验。

**容量淘汰**：每桶上限 `max_per_bucket`（默认 200），超限时按 `hit_count` + `final_quality` 升序淘汰冷旧条目。

**跨 Agent 不互看** — SingleHop 的经验不会被 MultiHop Agent 召回，保证消融实验中 "w/o Memory" 的粒度可以精确到单个 Agent。

**记忆闭环**：MultiHop 的 Plan 阶段从本桶召回 `outcome="lesson"` 条目作为规避指引（见 §6），G-PER Reflect 产出的类型化缺口则沉淀进 `GapLessonStore`（3C）供未来预取——**普通文本 lesson（工具序列层面规避）与图谱 gap-lesson（结构层面预取修复）分工互补**，前者是后者的 graph-poor 降级路径。

### 11.3 会话记忆（SessionMemory）

`memory/memory_manager.py::SessionMemory` — 进程内 `dict[str, list[str]]`，按 `session_id` 隔离，用于 `follow_up` 指代消解。与双层记忆职责不同但可共存。

### 11.4 MemoryManager 统一门面

```python
class MemoryManager:
    shared: SharedMemory                # 跨 Agent 共享（文本哈希）
    agent_private: AgentPrivateMemory   # 按 Agent × 意图隔离
    session: SessionMemory              # 会话级
    # —— C3 图接地三层 ——
    shared_evidence: EvidenceSubgraphStore   # 共享：证据子图（实体邻域重叠检索）
    plan_templates: PlanTemplateStore        # 私有：图规划模板
    gap_lessons: GapLessonStore              # 私有：类型化 gap 教训
```

### 11.5 C3 实装：图接地双层记忆

`memory/graph_memory.py` 为原来的"文本哈希缓存 + 自由文本 lesson"补上图谱锚定的对应物。动机是一个**审稿意见**：在**不重复提问**的 QA 基准上，文本哈希缓存的命中率难以解释；而且私有记忆的"学习"贡献会和缓存命中混为一谈。三层机制如下：

| 机制 | 类 | 层 | 键 / 检索方式 | 解决的问题 |
|---|---|---|---|---|
| **3B** | `EvidenceSubgraphStore` | 共享 | 按**实体邻域重叠（Jaccard）**检索，而非文本哈希 | 命中反映的是"**同一实体邻域被结构性复用**"，而非问题重复——在非重复基准上也站得住 |
| **3A** | `PlanTemplateStore` | 私有 | 按 `compositionality + 实体签名` 召回**图规划模板** | 新查询可以拿到一个**结构层面的 plan prior**，命中即跳过 LLM Plan（`plan_source="template"`） |
| **3C** | `GapLessonStore` | 私有 | 按实体召回 G-PER Reflect 产出的**类型化 gap** | 把"上次缺哪条边"变成**抢占式检索**：推理开始前就把缺失边取回来（`pre_emptive_gaps`） |

**三者都优雅退化**：没有实体或没有图证据时返回空，系统回落到文本哈希缓存与文本 lesson。所以

- **graph-rich** 下：3A/3B/3C + 文本机制同时生效；
- **graph-poor** 下：3A/3B/3C 静默失效，只剩文本哈希缓存 + 文本 lesson ——这正是 §2 流程里第 4 步（证据子图缓存）会"跳过"、而第 3 步（QA 缓存）仍然工作的原因。

`MultiAgentRetriever._persist_graph_memory()` 是写侧：每轮结束后把 `(entities, evidence triples, answer, quality)` 写进 `shared_evidence`；仅当 `label == "multi_hop"` 时，再把 trace 里的 `plan + compositionality` 写进 `plan_templates`、把 `structural_gaps` 写进 `gap_lessons`。

---

## 12. 设计模式

| 模式 | 应用位置 |
|---|---|
| **Strategy** | 每个 Agent 实现 `BaseAgent.do_execute()` |
| **Factory** | `AgentFactory.create(intent, all_tools, ...)` 映射意图到 Agent 类 |
| **Builder** | `ToolSetBuilder.for_agent(agent_type, all_tools)` 按矩阵过滤工具 |
| **Template Method** | `BaseAgent.execute()` 定义 `_pre → do → _post` 骨架 |
| **Facade** | `MultiAgentRetriever` 对外暴露 `retrieve()`，内部完成编排 |
| **Registry** | `@register` 装饰器 + 全局 Registry 单例 |

### BaseAgent 模板方法

```python
class BaseAgent(ABC):
    def execute(self, query, context) -> Answer:
        self._pre_execute(query, context)    # 读记忆
        answer = self.do_execute(query, context)  # 子类实现
        self._post_execute(query, answer, context)  # 写记忆 + trace
        return answer

    @abstractmethod
    def do_execute(self, query, context) -> Answer: ...
```

### AgentFactory 工厂

```python
class AgentFactory:
    _registry: dict[str, type[BaseAgent]] = {}

    @classmethod
    def register(cls, intent: str):
        """装饰器，将 Agent 类注册到意图标签。"""

    @classmethod
    def create(cls, intent, all_tools, memory_manager, config) -> BaseAgent:
        agent_cls = cls._registry[intent]
        tools = ToolSetBuilder.for_agent(intent, all_tools)
        return agent_cls(agent_type=intent, tools=tools, config=config)
```

---

## 13. 配置开关

```yaml
query:
  intent_classifier:
    active: colla_rag.tree
  retriever:
    active: colla_rag.multi_agent

plugins:
  colla_rag:
    enabled: true

    multi_agent:
      classifier:
        confidence_threshold: 0.7
        rule_fallback: true

      native_rag:
        default_top_k: 10
        rerank_top_k: 5
        quality_threshold: 0.4

      react:
        max_iterations: 3
        quality_threshold: 0.6

      map_reduce:
        retrieve_top_k: 50

      web_search:
        search_backend: "duckduckgo"
        max_results: 5

      memory:
        path: ".chimera_memory"
        max_per_bucket: 200
```

关闭时只需 `enabled: false`，并把 `intent_classifier.active` 回到 `defaults.rule`、`retriever.active` 回到 `defaults.direct`。

---

## 14. 在 QueryPipeline 中的接入

`QueryPipeline.run()` 对底层的多Agent编排**完全透明**——它看见的只是一个实现了 `BaseRetriever.retrieve(query, top_k) -> RetrievalResult` 的 `MultiAgentRetriever`。

```
query
  → self.intent_classifier.classify(query)   ← colla_rag.tree
  → self.retriever.retrieve(query, top_k)    ← colla_rag.multi_agent
                                               （内部完成预处理→分类→分派→后处理）
  → self.generator.generate(query, retrieval)
  → Answer
```

`QueryPipeline` 自己也会跑一次 `intent_classifier.classify()`，当 CollaRAG 启用时，这次分类与 `MultiAgentRetriever` 内部的分类**都走 `TreeIntentClassifier`**——结果一致。

---

## 15. 对应测试

| 测试文件 | 主要覆盖点 |
|---|---|
| `tests/test_colla_rag_memory.py` | SessionMemory append / 窗口截断 |
| `tests/test_colla_rag_classifier.py` | 5 类 LLM mock 分类；greeting 过滤；规则兜底 |
| `tests/test_colla_rag_strategies.py` | 每个 Agent 的独立行为 |
| `tests/test_colla_rag_agent.py` | AgentFactory 创建；ToolSetBuilder 过滤 |
| `tests/test_multi_hop_agent.py` | MultiHop Agent Plan-Execute-Reflect 流程 |
| `tests/test_single_hop_agent.py` | SingleHop Agent 单轮检索 |
| `tests/test_summarization_agent.py` | Summarization Agent Map-Reduce 流程 |
| `tests/test_other_agent.py` | Other Agent 联网搜索 |
| `tests/test_pipeline_colla_rag.py` | 端到端：`colla_rag_only` 配置的 `ingest → query` |
| `tests/test_pipeline_full.py` | AdaGraph × CollaRAG 同时启用 |

---

## 16. 与构建侧基线的正交性

CollaRAG **只读** `PipelineState.graph_store / vector_store / chunk_lookup`——不关心这些存储是怎么被构建出来的。因此：

- 开启 `adagraph_only`：CollaRAG 关闭，只检查"图好了之后裸检索效果"
- 开启 `colla_rag_only`：构建侧基线关闭，只检查"固定图上多Agent检索的效果"
- 开启 `full`：构建侧基线 + CollaRAG 叠加，观察超加性增益

**四种配置**共用同一份 `QueryPipeline.run()` 代码路径，差异**只由 yaml 的 `active` 字段驱动**：

| 文件 | 构建侧基线 | CollaRAG |
|---|:---:|:---:|
| `vanilla.yaml` | off | off |
| `adagraph_only.yaml` | on | off |
| `colla_rag_only.yaml` | off | on |
| `full.yaml` | on | on |

---

## 17. C4：图条件化能力特化

**文件**：`agent_factory.py`（`GraphFootprint` + `ToolSetBuilder` + `GRAPH_VIEWS`）

这是把"静态工具权限矩阵"升级为"**按知识图谱能不能撑住这个查询动态收放**"。

### 17.1 GraphFootprint

分派前由 `MultiAgentRetriever._footprint()` 做一次**廉价探针**（无 LLM）：

```python
@dataclass
class GraphFootprint:
    entity_hit_rate: float   # 探针实体中存在于 KG 的比例
    has_path: bool           # 两个实体之间在 KG 中是否存在路径
    density: "dense"|"sparse"|"unknown"   # 由 schema 谓词数粗判（≥10 → dense）
    n_entities: int

    @property
    def graph_poor(self) -> bool:
        return self.entity_hit_rate == 0.0 or (self.density == "sparse" and not self.has_path)
```

`graph_store is None` 或探针实体为空时返回 `None`，`ToolSetBuilder` 随即回退到静态矩阵。

### 17.2 动态规则（机制 4A）

| 条件 | 动作 |
|---|---|
| `agent == "multi_hop"` 且 `footprint.graph_poor` | **撤回** 5 个图遍历工具（`graph_path_search` / `graph_subgraph_extract` / `graph_neighbors` / `graph_community_search` / `graph_statistics`），**补入** `web_search` / `web_fetch` |
| `agent == "single_hop"` 且 `entity_hit_rate == 0 且 n_entities > 0` | 撤回 `graph_neighbors`（返回必为空），改用稠密 / BM25 检索 |
| 其他 Agent / `footprint is None` | 不动，用静态矩阵 |

设计意图：图贫乏区域里，图遍历工具会白白烧掉迭代次数去探一个空图；撤回它并给文本回退，等于**把"工具权限"从静态配置变成了 KG 条件的函数**。

### 17.3 KG 投影声明（机制 4B）

`GRAPH_VIEWS` 把"每个 Agent 该看到 KG 的哪一片"显式化——静态矩阵其实已经实现了这些投影，这个映射只是把它**写明**：

```python
GRAPH_VIEWS = {
    "preprocessor":  "none",
    "single_hop":    "attribute",   # 实体 → 属性
    "multi_hop":     "path",        # 实体 → 关系 → 实体 路径
    "summarization": "community",   # 社区级子图
    "other":         "none",
}
```

### 17.4 Agent 可用矩阵（C1–C4 的具体落点）

完整的 `TOOL_MATRIX` 见 §10.10；Web 端可在 `GET /api/agents/matrix` 拿到运行时视图。

---

## 相关文档

- [`architecture_overview.md`](../architecture_overview.md) — 架构总览：注册表、插槽、依赖注入
- [`design.md`](design.md) — 多Agent架构设计文档（含消融实验设计）
- [`../innovation-1/adagraph.md`](../innovation-1/adagraph.md) — 构建侧前置基线（**注意：不是创新点一**）
- [`../experiments/`](../experiments/) — 实验结果与实验设计文档
- [`../../README.md`](../../README.md) — 仓库入门与创新点口径说明