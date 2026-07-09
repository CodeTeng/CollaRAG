# 创新点二：CollaRAG —— 多Agent意图感知智能检索

> 本文是创新点二的**完整实装文档**，对应代码 `src/chimera_rag/plugins/colla_rag/`。
>
> **作用阶段**：Query（查询）  
> **启用方式**：`configs/*.yaml` 中 `plugins.colla_rag.enabled: true`，`query.retriever.active` 设为 `colla_rag.multi_agent`，`query.intent_classifier.active` 设为 `colla_rag.tree`

---

## 目录

- [1. 架构总览](#1-架构总览)
- [2. MultiAgentRetriever 编排器](#2-multiagentretriever-编排器)
- [3. Preprocessor Agent](#3-preprocessor-agent)
- [4. TreeIntentClassifier 树形分类器](#4-treeintentclassifier-树形分类器)
- [5. SingleHop Agent](#5-singlehop-agent)
- [6. MultiHop Agent](#6-multihop-agent)
- [7. Summarization Agent](#7-summarization-agent)
- [8. Other Agent](#8-other-agent)
- [9. Greeting 短路](#9-greeting-短路)
- [10. 共享工具层（31 个工具）](#10-共享工具层31-个工具)
- [11. 双层记忆架构](#11-双层记忆架构)
- [12. 设计模式](#12-设计模式)
- [13. 配置开关](#13-配置开关)
- [14. 在 QueryPipeline 中的接入](#14-在-querypipeline-中的接入)
- [15. 对应测试](#15-对应测试)
- [16. 与 AdaGraph 的正交性](#16-与-adagraph-的正交性)

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
│  ┌────────────────────────────┐                                  │
│  │  Preprocessor Agent（预处理）│                                  │
│  │  工具：query_rewrite        │                                  │
│  │       coreference_resolve  │                                  │
│  │       followup_merge       │                                  │
│  │       read_session_context │                                  │
│  └────────────┬───────────────┘                                  │
│               │ rewritten_query                                  │
│               ▼                                                  │
│  ┌────────────────────────────┐                                  │
│  │  Tree-Intent Classifier     │                                  │
│  │  Level-1: greeting?         │                                  │
│  │  Level-2: single_hop |      │                                  │
│  │    multi_hop | summarization │                                 │
│  │    | other                  │                                  │
│  └──┬──────┬───────┬────────┬─┘                                  │
│     │      │       │        │                                    │
│  greeting single   multi   summ   other                          │
│     │      hop     _hop      │      │                            │
│     ▼      ▼       ▼       ▼      ▼                              │
│  LLM直答 SingleHop MultiHop Summa-  Other                        │
│  (短路)  Agent    Agent    rization Agent                        │
│                  (Plan-     Agent                                │
│                  Execute-                                        │
│                  Reflect)                                          │
│     │      │       │       │      │                              │
│     └──────┴───────┴───────┴──────┘                              │
│               │                                                  │
│               ▼                                                  │
│  ┌────────────────────────────┐                                  │
│  │  Post-Process（后处理）      │                                  │
│  │  统一 RetrievalResult       │                                  │
│  │  写入记忆 + trace 元数据     │                                  │
│  └────────────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────┘
```

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

1. **分类**：`TreeIntentClassifier.classify(query)` → `intent.label`
2. **Greeting 短路**：若 `greeting`，直接 LLM 直答返回
3. **QA 缓存检查**：`SharedMemory.get_qa_cache()` 命中则直接返回
4. **预处理**：`PreprocessorAgent.execute(query)` → 改写后的 query
5. **分派**：`AgentFactory.create(label, ...)` 创建对应 Agent，执行 `agent.execute()`
6. **后处理**：合并结果、写 QA 缓存、写路由统计、写 Agent 私有记忆、写会话历史

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

**职责**：对原始 query 做预处理，产出适合下游分类和检索的干净 query。

**可用工具**（5 个）：
- `query_rewrite`：LLM 改写 query（扩展/规范/消歧）
- `coreference_resolve`：指代消解
- `followup_merge`：追问拼接为自包含 query
- `read_session_context`：读取会话历史
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

**可用工具**（15 个）：
- 检索：`vector_search`、`bm25_search`、`hybrid_search`、`graph_neighbors`
- 推理：`assess_evidence`、`rerank`
- 记忆：`read_shared_memory`、`read_agent_memory`
- IO：`save_to_file`、`load_from_file`
- 验证：`answer_verify`
- 证据处理：`evidence_dedup`、`confidence_calibrate`

**质量检查**：`assess_evidence` 仅做**一次性**轻量检查，不触发循环。若 `quality < quality_threshold`（默认 0.4），在 trace 中标记 `low_confidence`。

**特点**：最快、成本最低的路径。

---

## 6. MultiHop Agent

**注册**：`@AgentFactory.register("multi_hop")`  
**文件**：`agents/multi_hop.py`

**适用场景**：多跳推理、比较、探索性查询。

**架构**（Plan-Execute-Reflect 三阶段）：

```
Phase 1: Plan（规划）
   LLM 分析 query → 生成结构化步骤列表 [{"step": "...", "tools_hint": [...]}]
       │
       ▼
Phase 2: Execute（逐步执行）
   对每个步骤运行迷你 ReAct 循环：
   Thought → Action → Observation → ... → Step Result
   每步最多 max_step_iterations 轮
       │
       ▼
Phase 3: Reflect（反思综合）
   assess_evidence 评估整体质量
   ├── quality ≥ threshold → 直接综合
   └── quality < threshold → 反思 → 补充执行 → 再评估（最多 reflect_retries 次）
       │
       ▼
   Synthesize: 综合所有步骤证据 → 生成最终答案
```

**可用工具**（15 个）：
- 检索：`vector_search`、`bm25_search`、`hybrid_search`、`graph_neighbors`、`graph_path_search`、`graph_community_search`
- 图分析：`graph_subgraph_extract`、`graph_statistics`
- 实体：`entity_extract`、`relation_extract`、`entity_link`
- 推理：`sub_query_decompose`、`assess_evidence`、`rerank`
- 记忆：`read_session_context`、`read_shared_memory`、`read_agent_memory`
- IO：`save_to_file`、`load_from_file`
- 验证：`answer_verify`、`claim_decompose`、`source_attribution`
- 证据处理：`temporal_filter`、`evidence_dedup`、`chunk_summarize`、`confidence_calibrate`

**Plan 阶段**：使用 `PLAN_PROMPT` 让 LLM 分析 query 并输出 JSON 步骤列表。若 LLM 返回无法解析，降级为单步计划 `[{"step": original_query}]`。步骤数上限由 `max_plan_steps`（默认 5）控制。

**Execute 阶段**：对每个步骤运行迷你 ReAct 循环，使用 `STEP_REACT_PROMPT`。每步通过正则解析 `Action: tool_name\nAction Input: {json}` 调用工具，解析 `Step Result: ...` 收集本步结果。前序步骤的结果作为上下文注入后续步骤。

**Reflect 阶段**：所有步骤执行完毕后，调用 `assess_evidence` 评估整体证据质量。若 `quality < quality_threshold`（默认 0.6）且剩余反思预算（`reflect_retries`，默认 2），使用 `REFLECT_PROMPT` 生成反思文本，识别信息缺口，再执行一轮补充检索。最终用 `SYNTHESIZE_PROMPT` 综合所有证据生成最终答案。

**Plan 阶段的失败教训规避（text-lesson fallback）**：Plan 之前，MultiHop 通过 `_recall_avoid_lessons(context.agent_memory)` 召回本桶（`multi_hop/multi_hop`）里**复发最多的失败经验**（`outcome="lesson"` 条目，按 `hit_count` 排序），渲染进 `GPER_PLAN_PROMPT` 的 `{avoid_lessons}` 槽，提示规划器**不要重复**已证实低质量的工具序列模式。这是图谱锚定 gap-lesson（3C）的**纯文本降级对应物**：graph-poor 设定下图谱缺口为空，它成为唯一可用的学习信号；graph-rich 设定下与结构化 gap-lesson 互补。召回失败或无记忆时优雅返回 `[]`，规划器无种子运行。trace 记录 `avoided_lessons` 计数。

**trace 记录**：`plan`（执行计划）、`step_results`（每步结果）、`iterations`（总工具调用数）、`tool_call_log`（工具调用序列）、`reflect_count`（反思次数）、`quality`（最终质量分数）。

**与旧 ReAct 循环的关键差异**：

| 方面 | 旧 ReAct 循环 | Plan-Execute-Reflect |
|---|---|---|
| 结构 | 平铺 Thought→Action→Observation 循环 | 三阶段：Plan → Execute → Reflect |
| 规划 | 无显式规划，LLM 隐式决策 | 显式 JSON 步骤列表 |
| 执行 | 全局单循环，max_iterations=3 | 每步独立迷你循环，max_step_iterations=3 |
| 反思 | 循环内 assess_evidence 提示 | 独立反思阶段，可补充执行 |
| 综合 | 循环内直接 Final Answer | 独立 Synthesize 阶段，综合所有步骤证据 |

---

## 7. Summarization Agent

**注册**：`@AgentFactory.register("summarization")`  
**文件**：`agents/summarization.py`

**适用场景**：归纳总结、综述类查询。

**流程**（确定性流水线，非 ReAct 循环）：

```
hybrid_search(query, top_k=50) → Map（每个 chunk 独立 LLM 摘要）→ Reduce（合并为最终综述）
```

**可用工具**（9 个）：
- 检索：`vector_search`、`bm25_search`、`hybrid_search`、`graph_community_search`
- 推理：`rerank`
- 记忆：`read_shared_memory`、`read_agent_memory`
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

**可用工具**（7 个）：
- 检索：`hybrid_search`
- Web：`web_search`、`web_fetch`
- 记忆：`read_shared_memory`、`read_agent_memory`
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

## 10. 共享工具层（31 个工具）

所有工具统一实现为 `langchain_core.tools.StructuredTool`，参数由 Pydantic schema 校验。每个 Agent 通过 `ToolSetBuilder` 按 `TOOL_MATRIX` 过滤可用工具子集。

### 10.1 检索类工具（`tools/retrieval_tools.py`）

| 工具名 | 描述 |
|---|---|
| `vector_search` | FAISS 语义向量检索 |
| `bm25_search` | BM25 关键词检索（rank-bm25） |
| `hybrid_search` | RRF 融合检索（vector + BM25） |
| `graph_neighbors` | 知识图谱 N-hop 邻居展开 |
| `graph_path_search` | 两实体间最短路径搜索 |
| `graph_community_search` | 基于图社区发现的主题检索 |

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
| `sub_query_decompose` | | | ✅ | | |
| `assess_evidence` | | ✅ | ✅ | | |
| `rerank` | | ✅ | ✅ | ✅ | |
| `query_rewrite` | ✅ | | | | |
| `coreference_resolve` | ✅ | | | | |
| `followup_merge` | ✅ | | | | |
| `read_session_context` | ✅ | | ✅ | | |
| `write_session_context` | | 后处理 | 后处理 | 后处理 | 后处理 |
| `read_shared_memory` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `write_shared_memory` | | 后处理 | 后处理 | 后处理 | 后处理 |
| `read_agent_memory` | | ✅ | ✅ | ✅ | ✅ |
| `write_agent_memory` | | 后处理 | 后处理 | 后处理 | 后处理 |
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

---

## 11. 双层记忆架构

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
│   └── rewrite_templates.json
├── single_hop/experience/
│   └── single_hop.jsonl
├── multi_hop/experience/
│   └── multi_hop.jsonl
├── summarization/experience/
│   └── summarization.jsonl
└── other/experience/
    └── other.jsonl
```

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
    shared: SharedMemory              # 跨 Agent 共享
    agent_private: AgentPrivateMemory # 按 Agent × 意图隔离
    session: SessionMemory            # 会话级
```

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

## 16. 与 AdaGraph 的正交性

CollaRAG **只读** `PipelineState.graph_store / vector_store / chunk_lookup`——不关心这些存储是怎么被构建出来的。因此：

- 开启 `adagraph_only`：CollaRAG 关闭，只检查"图好了之后裸检索效果"
- 开启 `colla_rag_only`：AdaGraph 关闭，只检查"固定图上多Agent检索的效果"
- 开启 `full`：两创新点叠加，观察超加性增益

**四种配置**共用同一份 `QueryPipeline.run()` 代码路径，差异**只由 yaml 的 `active` 字段驱动**：

| 文件 | AdaGraph | CollaRAG |
|---|:---:|:---:|
| `vanilla.yaml` | off | off |
| `adagraph_only.yaml` | on | off |
| `colla_rag_only.yaml` | off | on |
| `full.yaml` | on | on |

---

## 相关文档

- [`architecture_overview.md`](../architecture_overview.md) — 架构总览：注册表、插槽、依赖注入
- [`design.md`](design.md) — 多Agent架构设计文档（含消融实验设计）
- [`../innovation-1/adagraph.md`](../innovation-1/adagraph.md) — 创新点一：AdaGraph 自适应图谱构建
- [`../experiments/`](../experiments/) — 实验结果与实验设计文档
- [`../experiments/`](../experiments/) — 实验结果