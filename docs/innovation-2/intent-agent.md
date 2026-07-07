# 创新点二：IntentAgent —— 多Agent意图感知智能检索

> 本文是创新点二的**完整实装文档**，对应代码 `src/chimera_rag/plugins/intent_agent/`。
>
> **作用阶段**：Query（查询）  
> **启用方式**：`configs/*.yaml` 中 `plugins.intent_agent.enabled: true`，`query.retriever.active` 设为 `intent_agent.multi_agent`，`query.intent_classifier.active` 设为 `intent_agent.tree`

---

## 目录

- [1. 架构总览](#1-架构总览)
- [2. MultiAgentRetriever 编排器](#2-multiagentretriever-编排器)
- [3. Preprocessor Agent](#3-preprocessor-agent)
- [4. TreeIntentClassifier 树形分类器](#4-treeintentclassifier-树形分类器)
- [5. Native RAG Agent](#5-native-rag-agent)
- [6. ReAct Agent](#6-react-agent)
- [7. Map-Reduce Agent](#7-map-reduce-agent)
- [8. WebSearch + LLM Agent](#8-websearch--llm-agent)
- [9. Greeting 短路](#9-greeting-短路)
- [10. 共享工具层（19 个工具）](#10-共享工具层19-个工具)
- [11. 双层记忆架构](#11-双层记忆架构)
- [12. 设计模式](#12-设计模式)
- [13. 配置开关](#13-配置开关)
- [14. 在 QueryPipeline 中的接入](#14-在-querypipeline-中的接入)
- [15. 对应测试](#15-对应测试)
- [16. 与 AdaGraph 的正交性](#16-与-adagraph-的正交性)

---

## 1. 架构总览

IntentAgent 采用**多Agent协作架构**：一个 Preprocessor Agent 做预处理 → 一个 TreeIntentClassifier 做意图分类 → 根据意图分派到 5 个专职 Agent 之一。

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
│  LLM直答  Native  ReAct   MapRed  WebSearch                       │
│  (短路)  RAG     Agent   Agent   +LLM Agent                     │
│          Agent  (+反思)                                          │
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
@register("retriever", "intent_agent.multi_agent")
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

**注册**：`@register("intent_classifier", "intent_agent.tree")`  
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
| `single_hop` | Native RAG Agent | "爱因斯坦的出生地？" |
| `multi_hop` | ReAct Agent（含反思循环） | "比较 A 和 B 的贡献差异" |
| `summarization` | Map-Reduce Agent | "总结这篇文档的主要观点" |
| `other` | WebSearch + LLM Agent | "今天东京天气怎样？" |

### Greeting 检测

正则匹配：`^(hi|hello|hey|你好|嗨|哈喽|good morning|...)` 或包含问候语 + 寒暄词。

### LLM 分类

用 `_CLASSIFY_PROMPT` 向 LLM 发一次 completion，要求返回单个标签。解析结果时做模糊匹配（只要包含有效标签名即可）。

### 规则兜底

- `summarization`：包含 summarize/总结/概括/归纳/overview 等关键词
- `multi_hop`：包含 compare/contrast/differ/vs/relationship 等关键词
- 默认 → `single_hop`

---

## 5. Native RAG Agent

**注册**：`@AgentFactory.register("single_hop")`  
**文件**：`agents/native_rag.py`

**适用场景**：事实性单跳查询。

**流程**（单轮、无 ReAct 循环、无反思重试）：

```
hybrid_search(query, top_k=10) → assess_evidence（一次性质量检查）→ LLM 生成
```

**可用工具**（10 个）：
- 检索：`vector_search`、`bm25_search`、`hybrid_search`、`graph_neighbors`
- 推理：`assess_evidence`、`rerank`
- 记忆：`read_shared_memory`、`read_agent_memory`
- IO：`save_to_file`、`load_from_file`

**质量检查**：`assess_evidence` 仅做**一次性**轻量检查，不触发循环。若 `quality < quality_threshold`（默认 0.4），在 trace 中标记 `low_confidence`。

**特点**：最快、成本最低的路径。

---

## 6. ReAct Agent

**注册**：`@AgentFactory.register("multi_hop")`  
**文件**：`agents/react_agent.py`

**适用场景**：多跳推理、比较、探索性查询。

**流程**（ReAct 循环 + 反思）：

```
Thought → Action → Observation → ... (循环 max_iterations 轮)
   ↓ 质量达标 或 出现 Final Answer
生成最终答案
```

**可用工具**（13 个）：
- 检索：`vector_search`、`bm25_search`、`hybrid_search`、`graph_neighbors`、`graph_path_search`、`graph_community_search`
- 推理：`sub_query_decompose`、`assess_evidence`、`rerank`
- 记忆：`read_session_context`、`read_shared_memory`、`read_agent_memory`
- IO：`save_to_file`、`load_from_file`

**ReAct 协议**：自研协议解析器，用正则 `Action: tool_name\nAction Input: {json}` 和 `Final Answer: ...` 匹配，不依赖特定 LLM 的 function calling。

**反思机制**：每轮迭代后调用 `assess_evidence` 评估证据质量。若 `quality ≥ quality_threshold`（默认 0.6），注入提示引导 LLM 给出最终答案。`max_iterations` 默认 3。

**trace 记录**：`iterations`（实际迭代轮数）、`tool_call_log`（工具调用序列）。

---

## 7. Map-Reduce Agent

**注册**：`@AgentFactory.register("summarization")`  
**文件**：`agents/map_reduce.py`

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

**Map 阶段**：对每个检索到的 chunk 独立调用 LLM 提取与 query 相关的关键信息。

**Reduce 阶段**：将所有局部摘要合并，LLM 生成最终综述答案。

**成本控制**：Map 阶段的 chunk 数量由 `retrieve_top_k` 控制（默认 50）。

---

## 8. WebSearch + LLM Agent

**注册**：`@AgentFactory.register("other")`  
**文件**：`agents/web_search.py`

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

## 10. 共享工具层（19 个工具）

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

### 10.7 Agent ↔ 工具可用矩阵

| 工具 | Preprocessor | Native RAG | ReAct | Map-Reduce | WebSearch |
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
├── native_rag/experience/
│   └── single_hop.jsonl
├── react_agent/experience/
│   └── multi_hop.jsonl
├── map_reduce/experience/
│   └── summarization.jsonl
└── web_search/experience/
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
    lesson: str | None,       # 失败教训
    hit_count: int,           # 命中计数（用于热度排序/淘汰）
    created_at: str,          # ISO 时间戳
}
```

**去重合并**：同 `tool_sequence` + 同 `outcome` 的条目不重复写入，只更新 `final_quality` 和 `hit_count`。

**容量淘汰**：每桶上限 `max_per_bucket`（默认 200），超限时按 `hit_count` + `final_quality` 升序淘汰冷旧条目。

**跨 Agent 不互看** — Native RAG 的经验不会被 ReAct Agent 召回，保证消融实验中 "w/o Memory" 的粒度可以精确到单个 Agent。

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
    active: intent_agent.tree
  retriever:
    active: intent_agent.multi_agent

plugins:
  intent_agent:
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
  → self.intent_classifier.classify(query)   ← intent_agent.tree
  → self.retriever.retrieve(query, top_k)    ← intent_agent.multi_agent
                                               （内部完成预处理→分类→分派→后处理）
  → self.generator.generate(query, retrieval)
  → Answer
```

`QueryPipeline` 自己也会跑一次 `intent_classifier.classify()`，当 IntentAgent 启用时，这次分类与 `MultiAgentRetriever` 内部的分类**都走 `TreeIntentClassifier`**——结果一致。

---

## 15. 对应测试

| 测试文件 | 主要覆盖点 |
|---|---|
| `tests/test_intent_agent_memory.py` | SessionMemory append / 窗口截断 |
| `tests/test_intent_agent_classifier.py` | 5 类 LLM mock 分类；greeting 过滤；规则兜底 |
| `tests/test_intent_agent_strategies.py` | 每个 Agent 的独立行为 |
| `tests/test_intent_agent_agent.py` | AgentFactory 创建；ToolSetBuilder 过滤 |
| `tests/test_pipeline_intent_agent.py` | 端到端：`intent_only` 配置的 `ingest → query` |
| `tests/test_pipeline_full.py` | AdaGraph × IntentAgent 同时启用 |

---

## 16. 与 AdaGraph 的正交性

IntentAgent **只读** `PipelineState.graph_store / vector_store / chunk_lookup`——不关心这些存储是怎么被构建出来的。因此：

- 开启 `adagraph_only`：IntentAgent 关闭，只检查"图好了之后裸检索效果"
- 开启 `intent_only`：AdaGraph 关闭，只检查"固定图上多Agent检索的效果"
- 开启 `full`：两创新点叠加，观察超加性增益

**四种配置**共用同一份 `QueryPipeline.run()` 代码路径，差异**只由 yaml 的 `active` 字段驱动**：

| 文件 | AdaGraph | IntentAgent |
|---|:---:|:---:|
| `deepseek_vanilla.yaml` | off | off |
| `deepseek_adagraph_only.yaml` | on | off |
| `deepseek_intent_only.yaml` | off | on |
| `deepseek_full.yaml` | on | on |

---

## 相关文档

- [`architecture_overview.md`](../architecture_overview.md) — 架构总览：注册表、插槽、依赖注入
- [`design.md`](design.md) — 多Agent架构设计文档（含消融实验设计）
- [`../innovation-1/adagraph.md`](../innovation-1/adagraph.md) — 创新点一：AdaGraph 自适应图谱构建
- [`../experiments/experiment-guide.md`](../experiments/experiment-guide.md) — 实验指南
- [`../experiments/`](../experiments/) — 实验结果