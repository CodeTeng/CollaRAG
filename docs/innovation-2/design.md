# Multi-Agent RAG 架构设计文档

> 日期：2026-07-07
> 作者：Teng Li
> 状态：设计中
> 范围：重构创新点二（CollaRAG），从"M-R-D-E + 6策略 + LangGraph动态编排"演化为"多Agent协作 + 共享工具层 + 双层记忆"架构。创新点一（AdaGraph）不受影响。

---

## 1. 设计动机

现有 CollaRAG 将所有检索逻辑集中在一个 Retriever 内部（6 种策略 + LangGraph 工具选择循环），存在三个问题：

1. **职责耦合**：单跳事实查询和多跳推理共用同一个 ReAct 循环，前者承受不必要的工具选择开销，后者的反思循环又被单跳场景拉低平均质量。
2. **缺乏预处理**：query 未经改写/指代消解直接进入分类和检索，导致追问型和模糊 query 的分类准确率下降。
3. **无归纳总结能力**：现有 6 种策略全部面向"找证据"，对"总结/综述"类需求只能退化为直接检索 + 生成，缺少 Map-Reduce 式的全局信息聚合。

新架构的核心思路：**按查询意图将检索分派给专职 Agent，每个 Agent 配备与其职责匹配的工具子集和私有记忆，共享工具层保证底层能力复用，双层记忆实现跨 Agent 的经验积累**。

---

## 2. 整体拓扑

```
用户 query + session_id
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Orchestrator（顶层编排器）                       │
│                                                                  │
│  ┌────────────────────────────┐                                  │
│  │  Preprocessor Agent（预处理）│                                  │
│  │  工具：query_rewrite        │                                  │
│  │       coreference_resolve  │                                  │
│  │       followup_merge       │                                  │
│  │       read_session_context │                                  │
│  │       read_shared_memory   │                                  │
│  └────────────┬───────────────┘                                  │
│               │ rewritten_query + metadata                       │
│               ▼                                                  │
│  ┌────────────────────────────┐                                  │
│  │  Tree-Intent Classifier     │                                  │
│  │  Level-1: greeting?         │                                  │
│  │  Level-2: single_hop |      │                                  │
│  │    multi_hop | summarization │                                 │
│  │    | other                  │                                  │
│  └──┬──────┬───────┬────────┬─┘                                  │
│     │      │       │        │                                    │
│  greeting single multi   summ   other                            │
│     │      │     _hop      │      │                              │
│     ▼      ▼       ▼       ▼      ▼                              │
│  LLM直答 Native  ReAct   MapRed  WebSearch                       │
│  (短路)  RAG     Agent   Agent   +LLM Agent                     │
│          Agent  (+反思)                                          │
│     │      │       │       │      │                              │
│     └──────┴───────┴───────┴──────┘                              │
│               │                                                  │
│               ▼                                                  │
│  ┌────────────────────────────┐                                  │
│  │  Post-Process（后处理）      │                                  │
│  │  统一 Answer schema         │                                  │
│  │  写入记忆 + trace 元数据     │                                  │
│  └────────────────────────────┘                                  │
│               │                                                  │
└───────────────┼──────────────────────────────────────────────────┘
                ▼
           Answer（统一输出）
```

**路由规则：一次性分派，不允许中途升级/转投。** 分类器决定后直接走对应 Agent，即使质量不达标也由该 Agent 自行在 max_iterations 内处理，不会 fallback 到其他 Agent。这保证了路由是纯函数，消融实验中每个 Agent 的贡献边界清晰。

---

## 3. 开源库选型

| 用途 | 选型 | 理由 |
|---|---|---|
| Agent 编排框架 | **LangGraph ≥1.0** | 项目已依赖，StateGraph API 稳定，支持条件边、循环、工具绑定 |
| 工具抽象 | **langchain-core ≥1.0** 的 `StructuredTool` | 已依赖，Pydantic schema 验证，与 LangGraph ToolNode 原生集成 |
| BM25 检索 | **rank-bm25 ≥0.2** | 零依赖纯 Python，够用于学术实验；不需要 Elasticsearch 等外部服务 |
| 联网搜索 | **duckduckgo-search ≥7.0** 或 **SerpAPI** | DDG 免费无 key；SerpAPI 质量更高但需 key，作为可选后端 |
| ReAct 协议解析 | 自研（复用现有 `llm_adapter.py` 的解析器） | 现有解析器已能处理 Thought/Action/Observation 格式，不依赖特定 LLM 的 function calling |
| Cross-Encoder 重排序 | **OpenAI-compatible `/rerank` 端点**（vLLM / Xinference / SiliconFlow / Jina） | 已实装，通过 `config.rerank.enabled: true` 开启 |

**不选 CrewAI/AutoGen 的理由**：它们强假设多 Agent 之间需要"对话协商"，而本方案是"一次性分派到专职 Agent"，不存在 Agent 间对话，LangGraph 的 StateGraph 更匹配这种确定性编排拓扑。

---

## 4. 设计模式

| 模式 | 应用位置 | 目的 |
|---|---|---|
| **Strategy** | 每个 Agent 实现 `BaseAgent` 接口的 `execute()` 方法 | Agent 可互换，Orchestrator 不感知具体实现 |
| **Factory** | `AgentFactory.create(agent_type, tools, memory, config)` | 集中创建 Agent 实例，注入依赖 |
| **Registry** | 扩展现有 `core/registry.py`，新增 `agent` 和 `tool` 两个 slot | 工具和 Agent 都用装饰器注册，yaml 开关控制 |
| **Template Method** | `BaseAgent` 定义 `pre_execute → do_execute → post_execute` 骨架 | 公共逻辑（记忆读写、trace 记录）在基类，差异逻辑在子类 |
| **Observer** | 记忆写入采用事件通知模式 | Agent 执行完毕后发出 `AgentCompleteEvent`，SharedMemory 和 AgentMemory 各自订阅并写入 |
| **Builder** | `ToolSetBuilder.for_agent(agent_type).with_vector().with_bm25().with_graph().build()` | 为每个 Agent 组装工具子集，避免手动拼列表 |
| **Facade** | `MultiAgentRetriever` 作为 `BaseRetriever` 的实现 | 对 `QueryPipeline` 暴露单一 `retrieve()` 接口，内部完成编排 |

---

## 5. 共享工具层（Tool Registry）

所有工具统一注册，每个 Agent 声明自己可用的工具子集。工具全部实现为 `langchain_core.tools.StructuredTool`，参数由 Pydantic schema 严格校验。

### 5.1 检索类工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `vector_search` | 复用 | FAISS 语义向量检索 | `query: str, top_k: int=5` |
| `bm25_search` | **新增** | BM25 关键词检索（rank-bm25） | `query: str, top_k: int=5` |
| `hybrid_search` | **新增** | RRF(vector + BM25) 融合检索 | `query: str, top_k: int=5, vector_weight: float=0.5` |
| `graph_neighbors` | 复用 | 知识图谱 N-hop 邻居展开 | `entity: str, hops: int=1` |
| `graph_path_search` | **新增** | 两实体间最短路径搜索（NetworkX） | `entity_a: str, entity_b: str, max_depth: int=4` |
| `graph_community_search` | **新增** | 基于图社区发现的主题检索 | `query: str, top_k: int=3` |

### 5.2 推理类工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `sub_query_decompose` | **新增** | LLM 将复合 query 拆解为子查询列表 | `query: str, max_sub: int=4` |
| `assess_evidence` | 复用 | 评估当前证据的质量/相关性/完整性 | `query: str` |
| `rerank` | **新增** | CrossEncoder 对 chunk 列表做语义重排序 | `query: str, chunk_ids: list[str], top_k: int=5` |

### 5.3 预处理类工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `query_rewrite` | **新增** | LLM 改写 query（扩展/规范/消歧） | `query: str, context: str=""` |
| `coreference_resolve` | **新增** | LLM 指代消解（将代词替换为显式实体） | `query: str, history: list[str]` |
| `followup_merge` | **新增** | 将追问与上一轮 QA 拼接为自包含 query | `query: str, prev_qa: dict` |

### 5.4 记忆/上下文工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `read_session_context` | **新增** | 读取当前会话的对话历史 | `session_id: str, last_n: int=5` |
| `write_session_context` | **新增** | 将当前轮 QA 写入会话历史 | `session_id: str, query: str, answer: str` |
| `read_shared_memory` | **新增** | 从共享记忆读取（QA 缓存/路由统计/实体别名） | `memory_type: str, key: str` |
| `write_shared_memory` | **新增** | 写入共享记忆 | `memory_type: str, key: str, value: Any` |
| `read_agent_memory` | **新增** | 读取当前 Agent 的私有经验记忆 | `agent_type: str, intent: str, top_n: int=3` |
| `write_agent_memory` | **新增** | 写入当前 Agent 的私有经验记忆 | `agent_type: str, entry: dict` |

### 5.5 联网搜索工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `web_search` | **新增** | 联网搜索（DuckDuckGo/SerpAPI） | `query: str, max_results: int=5` |
| `web_fetch` | **新增** | 抓取网页内容并提取正文 | `url: str, max_chars: int=3000` |

### 5.6 通用工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `save_to_file` | **新增** | 将结果/中间数据保存到指定路径 | `path: str, content: str, format: str="json"` |
| `load_from_file` | **新增** | 从文件加载数据 | `path: str` |

### 5.7 验证与溯源工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `answer_verify` | **新增** | LLM 验证 answer 是否被 evidence 支持 | `query: str, draft_answer: str` |
| `claim_decompose` | **新增** | 将文本拆为原子可验证命题 | `text: str` |
| `source_attribution` | **新增** | 为 answer 每个 claim 标注来源 chunk_id | `query: str, answer: str` |

### 5.8 图谱分析工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `graph_subgraph_extract` | **新增** | 围绕种子实体提取 k-hop 子图 + LLM 描述 | `entities: list[str], hops: int=2, max_triples: int=50` |
| `graph_statistics` | **新增** | 图谱统计（节点/边数、hub、密度） | 无参数 |

### 5.9 证据处理工具

| 工具名 | 新增/复用 | 描述 | Pydantic 参数 |
|---|---|---|---|
| `temporal_filter` | **新增** | 按年份范围过滤 evidence chunks | `year_min: int=0, year_max: int=2100` |
| `evidence_dedup` | **新增** | Jaccard 相似度语义去重 | `similarity_threshold: float=0.8` |
| `chunk_summarize` | **新增** | LLM 压缩单个 chunk（保留关键事实） | `query: str, chunk_id: str, max_length: int=300` |
| `confidence_calibrate` | **新增** | 多信号加权置信度校准 | `query: str, answer: str` |

### 5.10 Agent ↔ 工具可用矩阵

| 工具 | Preprocessor | SingleHop | MultiHop | Summarization | Other |
|---|:---:|:---:|:---:|:---:|:---:|
| `vector_search` | | ✅ | ✅ | ✅ | |
| `bm25_search` | | ✅ | ✅ | ✅ | |
| `hybrid_search` | | ✅ | ✅ | ✅ | ✅ |
| `graph_neighbors` | | ✅ | ✅ | | |
| `graph_path_search` | | | ✅ | | |
| `graph_community_search` | | | ✅ | ✅ | |
| `sub_query_decompose` | | | ✅ | | |
| `assess_evidence` | | ✅* | ✅ | | |
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

> \* Native RAG 的 `assess_evidence` 仅在最终结果上做**一次性**轻量质量检查（不触发循环），在 trace 中标记 `low_confidence`。

---

## 6. 各 Agent 详细设计

### 6.1 Preprocessor Agent

**职责**：对原始 query 做预处理，产出适合下游分类和检索的干净 query。

**LangGraph 状态图**：

```
START → read_context → process → END

process 节点：
  1. read_session_context → 获取会话历史
  2. read_shared_memory(entity_aliases) → 获取已知别名
  3. 判断是否追问 → followup_merge
  4. coreference_resolve → 指代消解
  5. query_rewrite → 改写/扩展/消歧
```

**输出**：`PreprocessResult(rewritten_query, original_query, is_followup, entities_mentioned, language)`

**无检索工具**，纯 LLM + 上下文工具。

### 6.2 Tree-Intent Classifier

**不是独立 Agent**，而是 Orchestrator 内部的一个分类函数。

**实现**：

```python
def classify(query: str) -> IntentLabel:
    # Level-1: 快速判断
    if _is_greeting(query):        # 关键词/正则
        return "greeting"

    # Level-2: LLM 分类 + 规则回退
    llm_label = _llm_classify(query)   # LLM 输出 single_hop|multi_hop|summarization|other
    if llm_label == "unknown":
        return _rule_fallback(query)   # 正则启发式兜底
    return llm_label
```

**分类标签集**：`greeting | single_hop | multi_hop | summarization | other`

其中 `multi_hop` 涵盖了原来的 `multi_hop + comparative + exploratory` 三类。

### 6.3 SingleHop Agent

**适用场景**：事实性单跳查询（"爱因斯坦的出生地是哪里？"）

**LangGraph 状态图**：

```
START → retrieve → rerank → quality_check → generate → END
```

**流程**：
1. `hybrid_search(query, top_k=10)` — 混合检索
2. `rerank(query, chunks, top_k=5)` — CrossEncoder 重排序
3. `assess_evidence(query)` — **一次性**质量检查，不触发循环，仅标记 `low_confidence`
4. LLM 生成答案

**特点**：单轮、无 ReAct 循环、无反思重试。最快、成本最低的路径。

### 6.4 MultiHop Agent（Plan-Execute-Reflect）

**适用场景**：多跳推理、比较、探索性查询

**三阶段架构**：

```
Phase 1: Plan
   LLM → JSON 步骤列表 [{"step": "...", "tools_hint": [...]}]
       ↓
Phase 2: Execute（逐步）
   for each step:
     迷你 ReAct 循环（max_step_iterations 轮）
     Thought → Action → Observation → Step Result
       ↓
Phase 3: Reflect + Synthesize
   assess_evidence → quality check
   ├── quality ≥ τ → synthesize → Answer
   └── quality < τ → reflect → 补充执行 → 再评估（最多 reflect_retries 次）
```

**与旧 ReAct 循环的关键差异**：

| 方面 | 旧 ReAct 循环 | Plan-Execute-Reflect |
|---|---|---|
| 结构 | 平铺 Thought→Action→Observation 循环 | 三阶段：Plan → Execute → Reflect |
| 规划 | 无显式规划，LLM 隐式决策 | 显式 JSON 步骤列表 |
| 执行 | 全局单循环，max_iterations=3 | 每步独立迷你循环，max_step_iterations=2 |
| 反思 | 循环内 assess_evidence 提示 | 独立反思阶段，可补充执行 |
| 综合 | 循环内直接 Final Answer | 独立 Synthesize 阶段，综合所有步骤证据 |

**max_plan_steps**：默认 5（可配置），控制计划步骤数上限。  
**max_step_iterations**：默认 2，每步 ReAct 循环上限。  
**reflect_retries**：默认 1，反思重试次数。

### 6.5 Summarization Agent (Map-Reduce)

**适用场景**：归纳总结、综述类查询（"总结这篇文档的主要观点"）

**流程**（非 ReAct 循环，确定性流水线）：

```
START → retrieve → map → reduce → END
```

1. **Retrieve**：`hybrid_search(query, top_k=50)` 拉取候选 chunk 集合
2. **Map**：对每个 chunk 独立调用 LLM 做局部摘要/信息提取（可并发）
   - Map prompt：`"从以下段落中提取与查询 '{query}' 相关的关键信息，简要概括。"`
3. **Reduce**：将所有局部摘要合并，LLM 生成最终综述答案
   - Reduce prompt：`"基于以下多个信息摘要，为查询 '{query}' 生成一个全面、连贯的综述。"`
4. `rerank` 可选——在 Retrieve 后、Map 前对 chunk 做一次重排序以减少无关 chunk 进入 Map 阶段

**成本控制**：Map 阶段的 chunk 数量由 `top_k` 控制（默认 50），不会对整个知识库做 Map。

### 6.6 Other Agent (WebSearch + LLM)

**适用场景**：超出知识库范围的查询

**流程**：

```
START → web_search → web_fetch(可选) → hybrid_search(可选补充) → generate → END
```

1. `web_search(query, max_results=5)` — 联网搜索
2. 对 top 结果可选 `web_fetch` 抓取全文
3. 可选 `hybrid_search` 补充本地知识库中的相关信息
4. LLM 综合网络结果 + 本地知识生成答案

### 6.7 Greeting 短路

**不是 Agent**，而是分类器判定 `greeting` 后 Orchestrator 直接调 LLM 返回。不进入任何 Agent、不读写记忆、不产生 trace。

---

## 7. 双层记忆架构

### 7.1 共享记忆（Shared Memory）

| 存储项 | 数据结构 | 读取时机 | 写入时机 |
|---|---|---|---|
| **路由统计** | `{query_hash → {intent, agent, quality, latency}}` | 分类器可参考历史路由模式 | Agent 执行完毕后 |
| **QA 缓存** | `{query_hash → {answer, quality, ttl}}` | Orchestrator 入口处检查，命中则直接返回 | 高质量 (Q≥τ) 的 QA 对 |
| **实体别名表** | `{alias → canonical_name}` | 预处理 Agent 指代消解时 | 预处理 Agent 发现新别名时 |
| **查询改写模板** | `{pattern → rewrite_template, hit_count}` | 预处理 Agent | 成功的改写 pattern 沉淀 |

**持久化**：JSON 文件，存储在 `.chimera_memory/shared/` 下。

### 7.2 Agent 私有记忆（Agent-Private Memory）

继承现有 `ExperientialMemory` 的核心设计（意图分桶、JSONL 追加、冷热淘汰、去重合并），但按 Agent 类型隔离存储目录：

```
.chimera_memory/
├── shared/
│   ├── routing_stats.json
│   ├── qa_cache.json
│   ├── entity_aliases.json
│   └── rewrite_templates.json
├── native_rag/
│   └── experience/
│       ├── single_hop.jsonl
│       └── ...
├── react_agent/
│   └── experience/
│       ├── multi_hop.jsonl
│       └── ...
├── map_reduce/
│   └── experience/
│       └── summarization.jsonl
└── web_search/
    └── experience/
        └── other.jsonl
```

**Agent 私有记忆条目（扩展自现有 ExperienceEntry）**：

```python
class AgentExperienceEntry(BaseModel):
    agent_type: str                    # 所属 Agent
    intent: str                        # 意图标签（分桶键）
    query_text: str                    # 原始查询
    tool_sequence: list[str]           # 工具调用序列
    reflection_count: int              # 反思迭代次数
    final_quality: float               # 最终质量分数
    outcome: Literal["success", "lesson"]
    lesson: str | None                 # 失败教训（Orchestrator 从 trace 派生的结构化文本，见下）
    answer_text: str | None            # 成功时保存答案
    evidence_chunk_ids: list[str]      # 证据 chunk ID
    hit_count: int = 0                 # 命中计数
    created_at: str                    # ISO 时间戳
```

**失败教训的写入、去重与召回**（让 `outcome="lesson"` 不再"记了不用"）：

- **写入**：Orchestrator 每轮按 `quality < 0.5` 判定 `outcome="lesson"`，并用 `_derive_lesson()` 从 trace 派生**结构化 lesson 文本**（不额外调用 LLM）——MultiHop 记 `low_quality q=…; reflect=N; unresolved_gaps=[…]`（含 G-PER 类型化缺口摘要），其他 Agent 记 `low_quality q=…; query='…'`。
- **去重合并**：同 `tool_sequence + outcome` 的条目不重复写入，合并时 `final_quality` 取最大、`hit_count` 自增、`lesson` 文本**刷新为最新一次的反思摘要**（旧版只更新 quality/hit_count，丢失最新教训）。
- **召回**：`recall()` 返回「先 success（按质量降序）后 lesson（按 hit_count 降序）」的混合列表，失败条目得以浮出；另提供 `recall_lessons()` 只返回失败条目，按 **复发次数** 排序——复发越多的工具模式越可能在类似 query 上重演，故优先召回以规避。**旧版 `recall()` 仅按 `final_quality` 降序，lesson 条目永远排不上**，这是本设计的核心修正。

**记忆闭环**：MultiHop 的 Plan 阶段从本桶召回 `outcome="lesson"` 条目作为工具序列层面的**规避指引**（注入 `GPER_PLAN_PROMPT` 的 `{avoid_lessons}` 槽）；G-PER Reflect 产出的类型化缺口则沉淀进 `GapLessonStore`（3C）供未来**结构层面预取修复**。两者分工互补：普通文本 lesson 是图谱 gap-lesson 的 graph-poor 降级路径——graph-poor 设定下图谱缺口为空，文本 lesson 是唯一可用的学习信号。

**跨 Agent 不互看**——SingleHop 的经验不会被 MultiHop Agent 召回，反之亦然。这保证了消融实验中"w/o Memory"的粒度可以精确到单个 Agent。

---

## 8. 核心接口设计

### 8.1 BaseAgent（模板方法模式）

```python
from abc import ABC, abstractmethod
from chimera_rag.core.types import Query, Answer

class BaseAgent(ABC):
    """所有专职 Agent 的基类，定义 pre → do → post 三段式骨架。"""

    def __init__(self, agent_type: str, tools: list, config: dict):
        self.agent_type = agent_type
        self.tools = tools
        self.config = config

    def execute(self, query: Query, context: AgentContext) -> Answer:
        """模板方法：子类不覆盖此方法，只覆盖 do_execute。"""
        # pre: 读取记忆，准备上下文
        self._pre_execute(query, context)
        # do: 核心逻辑（子类实现）
        answer = self.do_execute(query, context)
        # post: 写入记忆，记录 trace
        self._post_execute(query, answer, context)
        return answer

    def _pre_execute(self, query: Query, context: AgentContext) -> None:
        """读取 agent 私有记忆 + 共享记忆，注入到 context。"""
        ...

    @abstractmethod
    def do_execute(self, query: Query, context: AgentContext) -> Answer:
        """子类实现的核心检索/生成逻辑。"""
        ...

    def _post_execute(self, query: Query, answer: Answer, context: AgentContext) -> None:
        """写入 agent 私有记忆 + 共享记忆，填充 trace metadata。"""
        ...
```

### 8.2 AgentFactory（工厂模式）

```python
class AgentFactory:
    """根据意图标签创建对应的 Agent 实例。"""

    _registry: dict[str, type[BaseAgent]] = {}

    @classmethod
    def register(cls, intent: str):
        """装饰器，将 Agent 类注册到意图标签。"""
        def decorator(agent_cls):
            cls._registry[intent] = agent_cls
            return agent_cls
        return decorator

    @classmethod
    def create(cls, intent: str, tool_registry, memory_manager, config) -> BaseAgent:
        agent_cls = cls._registry[intent]
        tools = ToolSetBuilder.for_agent(intent, tool_registry).build()
        return agent_cls(agent_type=intent, tools=tools, config=config)
```

### 8.3 ToolSetBuilder（构建器模式）

```python
class ToolSetBuilder:
    """根据 Agent 类型组装工具子集。"""

    # Agent → 工具名 的可用矩阵（见 §5.7）
    _TOOL_MATRIX: dict[str, list[str]] = {
        "preprocessor": ["query_rewrite", "coreference_resolve", "followup_merge",
                         "read_session_context", "read_shared_memory"],
        "single_hop":   ["vector_search", "bm25_search", "hybrid_search",
                         "graph_neighbors", "assess_evidence", "rerank",
                         "read_shared_memory", "read_agent_memory",
                         "save_to_file", "load_from_file"],
        "multi_hop":    ["vector_search", "bm25_search", "hybrid_search",
                         "graph_neighbors", "graph_path_search",
                         "graph_community_search", "sub_query_decompose",
                         "assess_evidence", "rerank",
                         "read_session_context", "read_shared_memory",
                         "read_agent_memory", "save_to_file", "load_from_file"],
        "summarization":["vector_search", "bm25_search", "hybrid_search",
                         "graph_community_search", "rerank",
                         "read_shared_memory", "read_agent_memory",
                         "save_to_file", "load_from_file"],
        "other":        ["hybrid_search", "web_search", "web_fetch",
                         "read_shared_memory", "read_agent_memory",
                         "save_to_file", "load_from_file"],
    }

    @classmethod
    def for_agent(cls, agent_type: str, tool_registry) -> "ToolSetBuilder":
        ...

    def build(self) -> list[StructuredTool]:
        ...
```

### 8.4 MultiAgentRetriever（门面模式）

```python
@register("retriever", "colla_rag.multi_agent")
class MultiAgentRetriever(BaseRetriever):
    """对 QueryPipeline 暴露单一 retrieve() 接口，内部完成：
       预处理 → 分类 → 分派 → Agent 执行 → 后处理
    """

    def retrieve(self, query: Query, top_k: int = 5) -> RetrievalResult:
        # 1. 预处理
        preprocessed = self.preprocessor.execute(query, self.context)

        # 2. 分类
        intent = self.classifier.classify(preprocessed.rewritten_query)

        # 3. Greeting 短路
        if intent == "greeting":
            return self._greeting_shortcircuit(query)

        # 4. QA 缓存命中检查
        cached = self.shared_memory.get_qa_cache(preprocessed.rewritten_query)
        if cached:
            return cached

        # 5. 分派到专职 Agent
        agent = self.agent_factory.create(intent, ...)
        answer = agent.execute(preprocessed, self.context)

        # 6. 后处理：写缓存、记录路由统计
        self._post_process(query, intent, answer)
        return answer.to_retrieval_result()
```

---

## 9. 与现有系统的对接

| 现有组件 | 处理方式 |
|---|---|
| **AdaGraph（创新点1）** | 不动，继续占 IngestionPipeline 的 3 个插槽 |
| **`core/registry.py`** | 扩展，新增 `agent` 和 `tool` 两个 slot |
| **`core/pipeline.py` QueryPipeline** | 不改结构，`MultiAgentRetriever` 注册为 `retriever` slot 的新实现 |
| **`core/types.py`** | 新增 `AgentType` 枚举、`PreprocessResult`、`AgentContext`、`AgentExperienceEntry` |
| **`interfaces/`** | 新增 `base_agent.py`（BaseAgent ABC） |
| **`configs/*.yaml`** | 新增 `multi_agent` 配置段（见 §10） |
| **旧 `strategies/` 目录** | **删除**，策略逻辑分散到各 Agent 内部 |
| **旧 `langgraph/` 目录** | **删除**，MultiHop Agent 重写为 Plan-Execute-Reflect 架构 |
| **旧 `intent_classifier.py`** | **重构**为树形分类器，标签集从 6 类改为 5 类 |
| **旧 `experiential_memory.py`** | **重构**为双层架构（Shared + AgentPrivate），保留核心 JSONL/去重/淘汰逻辑 |

---

## 10. 配置示例（`configs/colla_rag_only.yaml`）

```yaml
query:
  retriever:
    active: colla_rag.multi_agent

plugins:
  colla_rag:
    enabled: true

    multi_agent:
      # 树形分类器
      classifier:
        llm_model: "primary"               # 使用主 LLM
        rule_fallback: true                 # 开启规则回退
        confidence_threshold: 0.7

      # 预处理 Agent
      preprocessor:
        enable_rewrite: true
        enable_coreference: true
        enable_followup_merge: true

      # SingleHop Agent
      native_rag:
        default_top_k: 10
        rerank_top_k: 5
        quality_check: true                 # 一次性轻量质量检查
        quality_threshold: 0.4              # 低于此值标记 low_confidence

      # MultiHop Agent (Plan-Execute-Reflect)
      react:
        max_iterations: 3
        quality_threshold: 0.6
        reflection_mode: "llm"              # "heuristic" | "llm"
        enable_sub_query: true

      # Summarization Agent (Map-Reduce)
      map_reduce:
        retrieve_top_k: 50
        map_concurrency: 5                  # Map 阶段并发数
        enable_rerank_before_map: true

      # Other Agent (WebSearch + LLM)
      web_search:
        search_backend: "duckduckgo"        # "duckduckgo" | "serpapi"
        max_results: 5
        supplement_local: true              # 是否同时查本地知识库

      # 双层记忆
      memory:
        shared:
          path: ".chimera_memory/shared"
          qa_cache_ttl: 3600                # QA 缓存 TTL（秒）
          max_qa_cache: 1000
        agent_private:
          path: ".chimera_memory"
          max_per_bucket: 200
```

---

## 11. 消融实验设计

| 消融变体 | 具体做法 | 验证目标 |
|---|---|---|
| **w/o Preprocess** | 跳过预处理 Agent，原始 query 直接分类 | 预处理改写的价值 |
| **w/o TreeIntent** | 所有 query 统一走 MultiHop Agent | 意图路由的必要性 |
| **w/o Reflect** | ReAct 只跑单轮，不做反思循环 | 反思机制的贡献 |
| **w/o Memory** | 禁用双层记忆（shared + private 都关） | 经验记忆的累积价值 |
| **w/o SharedMem** | 只禁用共享记忆，保留 Agent 私有记忆 | 共享 vs 私有记忆的贡献分离 |
| **w/o BM25** | 去掉 BM25 和混合检索，只用向量 | BM25/混合检索的增益 |
| **w/o Graph** | 去掉图相关工具，只用文本检索 | 知识图谱工具的贡献 |
| **w/o MapReduce** | 归纳总结查询退化为 Native RAG | Map-Reduce 的增量价值 |

---

## 12. 新增目录结构

```
src/chimera_rag/plugins/colla_rag/
├── __init__.py                          # 插件入口
├── orchestrator.py                      # Orchestrator + MultiAgentRetriever (Facade)
├── classifier.py                        # 树形意图分类器
├── base_agent.py                        # BaseAgent ABC (Template Method)
├── agent_factory.py                     # AgentFactory + ToolSetBuilder
├── agents/
│   ├── __init__.py
│   ├── preprocessor.py                  # Preprocessor Agent
│   ├── single_hop.py                    # SingleHop Agent
│   ├── multi_hop.py                     # MultiHop Agent (Plan-Execute-Reflect)
│   ├── summarization.py                 # Summarization Agent (Map-Reduce)
│   └── other.py                         # Other Agent (WebSearch + LLM)
├── tools/
│   ├── __init__.py                      # build_all_tools() + TOOL_MATRIX
│   ├── retrieval_tools.py               # vector/bm25/hybrid/graph 系列
│   ├── reasoning_tools.py               # sub_query_decompose/assess/rerank
│   ├── preprocess_tools.py              # query_rewrite/coreference/followup
│   ├── memory_tools.py                  # read/write session/shared/agent memory
│   ├── web_tools.py                     # web_search/web_fetch
│   ├── io_tools.py                      # save_to_file/load_from_file
│   ├── verification_tools.py            # answer_verify/claim_decompose/source_attribution
│   ├── graph_advanced_tools.py          # graph_subgraph_extract/graph_statistics
│   └── evidence_tools.py               # temporal_filter/evidence_dedup/chunk_summarize/confidence_calibrate
├── memory/
│   ├── __init__.py
│   ├── shared_memory.py                 # 共享记忆（路由统计/QA缓存/实体别名/改写模板）
│   ├── agent_memory.py                  # Agent 私有记忆（继承 ExperientialMemory 核心逻辑）
│   └── memory_manager.py                # 双层记忆统一管理器
└── hooks/
    ├── reflection_hook.py               # 反思 hook（复用）
    └── reranking_hook.py                # 重排序 hook（复用）
```

**删除的目录/文件**：
- `strategies/` — 整个目录（6 种策略文件）
- `langgraph/` — 整个目录（graph.py, tools.py, llm_adapter.py, retriever.py, experience.py）
- `agent.py` — 旧的 CollaRAGRetriever
- `reasoning.py` — 旧的推理阶段

**保留并复用的文件**：
- `hooks/reflection_hook.py` — 反思逻辑
- `hooks/reranking_hook.py` — 重排序逻辑
- `langgraph/llm_adapter.py` 的 ReAct 协议解析逻辑 — 提取到 `agents/multi_hop.py` 内部

---

## 13. 新增依赖

```toml
# pyproject.toml [project.optional-dependencies]
agent = [
    "langgraph>=1.0,<2",
    "langchain-core>=1.0,<2",
    "rank-bm25>=0.2.2",                    # 新增：BM25 检索
    "duckduckgo-search>=7.0",              # 新增：联网搜索
]
```

---

## 14. 数据流完整示例

**示例 query**："对比爱因斯坦和玻尔在量子力学上的贡献差异"

```
1. Preprocessor Agent:
   - read_session_context → 无历史
   - query_rewrite → "比较 Albert Einstein 和 Niels Bohr 在量子力学发展中的各自贡献及差异"

2. Tree-Intent Classifier:
   - Level-1: is_greeting? → No
   - Level-2: LLM → "multi_hop"（含比较语义 + 多实体跨文档推理）

3. 分派 → MultiHop Agent

4. MultiHop Agent 执行:
   - read_agent_memory(react, multi_hop) → 召回历史经验
   - Iteration 1:
     - Thought: "需要分别检索两人的量子力学贡献"
     - Action: sub_query_decompose → ["爱因斯坦 量子力学 贡献", "玻尔 量子力学 贡献"]
     - Action: hybrid_search("爱因斯坦 量子力学 贡献") → 5 chunks
     - Action: hybrid_search("玻尔 量子力学 贡献") → 5 chunks
     - Action: graph_neighbors("Einstein", hops=2) → 12 triples
     - Action: graph_neighbors("Bohr", hops=2) → 9 triples
     - Action: assess_evidence → quality=0.52 < τ=0.6
   - Reflect: "缺少两人之间的直接交互关系（如波尔-爱因斯坦论战）"
     → 注入反思文本到下一轮
   - Iteration 2:
     - Thought: "补充两人的图谱关联路径"
     - Action: graph_path_search("Einstein", "Bohr") → 3 paths
     - Action: hybrid_search("玻尔 爱因斯坦 论战 量子力学") → 4 chunks
     - Action: rerank(query, all_chunks, top_k=10) → 重排序
     - Action: assess_evidence → quality=0.78 ≥ τ
   - 合并证据 → LLM 生成对比分析

5. 后处理:
   - write_agent_memory(react, success entry)
   - write_shared_memory(routing_stats, ...)
   - write_shared_memory(qa_cache, ...)
   - 填充 trace metadata

6. 返回 Answer
```
