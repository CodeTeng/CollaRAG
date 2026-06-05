# 创新点二：IntentAgent —— 意图感知的智能体

> 本文档基于 `docs/plan.md` §核心功能 第 3 条与 `src/chimera_rag/plugins/intent_agent/` 实际实装整理。
> **作用阶段**：Query（查询）
> **代号对应**（plan.md §产品概述）：`IntentAgent` = 原设计材料中的 `MRDE`（Memory-Reasoning-Decision-Execution）
> **启用方式**：`configs/*.yaml` 中 `plugins.intent_agent.enabled: true`，并把 `query.intent_classifier.active` 切到 `intent_agent.llm`、`query.retriever.active` 切到 `intent_agent.strategy_selector`

---

## 1. 插件定位

IntentAgent 在 `QueryPipeline` 中同时占据两个主要插槽，并在其内部再挂接 4 个子插槽：

| 对外插槽 | 注册名 | 作用 |
|---|---|---|
| `intent_classifier` | `intent_agent.llm` | 6 类意图分类 |
| `retriever` | `intent_agent.strategy_selector` | 四阶段 M-R-D-E 编排器 |

`intent_agent.strategy_selector` 这个 retriever 内部再从注册表中拉取：

| 内部插槽 | 注册名 | 实现 |
|---|---|---|
| `memory` | `intent_agent.session` | 会话级历史 |
| `strategy`（6 个） | `intent_agent.{direct / multi_hop / parallel / iterative / expansion / context_aware}` | 6 种检索策略 |
| `reflector` | `intent_agent.heuristic` | 启发式答案质量评估（hook） |
| `reranker` | `intent_agent.token_overlap` | Jaccard 重排（hook） |
| — 非 slot，工具类 — | `ParallelRetrievalPool` | asyncio Semaphore 限流池（hook） |

开关方式（摘自 `configs/deepseek_full.yaml`）：

```yaml
query:
  intent_classifier:
    active: intent_agent.llm
    params: { confidence_threshold: 0.55 }
  retriever:
    active: intent_agent.strategy_selector
    params: { top_k: 5, neighbor_hops: 1 }

plugins:
  intent_agent:
    enabled: true
    memory:            { max_history: 10 }
    intent_classifier: { confidence_threshold: 0.55 }
    strategies:
      enabled: [direct, multi_hop, parallel, iterative, expansion, context_aware]
    hooks:
      reflection:    { enabled: false }
      reranking:     { enabled: false }
      parallel_pool: { enabled: true, max_workers: 4 }
```

关闭时只需 `enabled: false`，并把 `intent_classifier.active` 回到 `defaults.rule`、`retriever.active` 回到 `defaults.direct`——`ChimeraRAG.toggle_plugin("intent_agent", False)` 已经自动完成这一切（见 `src/chimera_rag/__init__.py::toggle_plugin`）。

---

## 2. 四阶段 M-R-D-E 总览

plan.md §核心功能 第 3 条规定的四个模块，在代码中分别落到 4 个文件：

```
Memory         → memory.py            SessionMemory
  │
  ▼
Reasoning      → intent_classifier.py LLMIntentClassifier  + rule_based_classify
                 reasoning.py         IntentReasoner
  │
  ▼
Decision       → agent.py             StrategySelector
  │
  ▼
Execution      → strategies/*.py      6 种策略（均 async）
  │
  └──► Hooks   → hooks/*.py           Reflection / Reranking / ParallelPool
```

整个四阶段**对 `QueryPipeline` 透明**：`QueryPipeline` 看见的只是一个实现了 `BaseRetriever.retrieve(query, top_k) -> RetrievalResult` 的对象——这个对象叫 `IntentAgentRetriever`（`agent.py`），它才是四阶段真正的装配车间。

---

## 3. 阶段 1 — Memory（会话级记忆）

### 3.1 类与接口

```python
# src/chimera_rag/plugins/intent_agent/memory.py
@register("memory", "intent_agent.session")
class SessionMemory(BaseMemory):
    def __init__(self, max_history: int | None = 10): ...
    def append(self, session_id: str, turn: str): ...
    def get_history(self, session_id: str) -> list[str]: ...
    def clear(self, session_id: str): ...
```

底层用 `collections.deque(maxlen=max_history)` 实现 **append-only + 自动窗口截断**。`max_history=None` 表示不截断（主要给测试用）；`max_history ≤ 0` 在构造时直接报 `ValueError`，符合 plan.md §技术栈"Pydantic 强类型 + 严格校验"的整体风格。

### 3.2 与检索的协作方式

Memory 本身不感知策略。协作发生在 `IntentAgentRetriever.retrieve()` 的开头：

```python
history = []
if self.memory and query.session_id:
    history = self.memory.get_history(query.session_id)
```

历史拿到后，会被放进 `plan.params["history"]`，等 `ContextAwareStrategy`（follow_up 意图专用）取用。**策略对 Memory 没有硬依赖**——它只读 plan 里的参数袋，这符合 plan.md §实施策略"依赖方向强约束"的总则。

一次检索结束后，当前轮再 append 回去：

```python
if self.memory and query.session_id:
    self.memory.append(query.session_id, query.text)
```

---

## 4. 阶段 2 — Reasoning（意图 + 计划）

### 4.1 6 类意图（plan.md 第 3 条枚举）

`core/types.py::IntentLabel` 用 `Literal` 类型锁定下列 6 项，任何其他字符串都无法构造合法 `Intent`：

| label | 典型表述 | 对应策略 |
|---|---|---|
| `factual` | who / what / when / where | `direct` |
| `analytical` | why / how come / explain why | `iterative` |
| `comparative` | compare / vs / difference between | `parallel` |
| `multi_hop` | impact / influence / relation between A and B | `multi_hop` |
| `exploratory` | tell me about / explain / overview | `expansion` |
| `follow_up` | and / then / what about her? / 单独代词 | `context_aware` |

这一映射由 `reasoning.py::_INTENT_TO_STRATEGY` 的查表完成，并为每个意图附带可供 UI 展示的 `steps` 列表（例如 "Decompose the question into sub-questions" / "BFS multi-hop expansion over the graph"）。

### 4.2 LLM + 规则 双路分类器

`intent_classifier.py::LLMIntentClassifier` 的执行顺序是**先 LLM，后规则兜底**：

1. 用 `CLASSIFY_PROMPT`（英文 + JSON 约束）向 LLM 发一次 completion；
2. `_safe_parse_label()`：
   - 先剥离 markdown 代码围栏；
   - 尝试 `json.loads`；
   - 失败则贪心提取最外层 `{...}` 再 parse；
   - 校验 `label ∈ CANONICAL_LABELS` 且 `confidence ∈ [0,1]`；
3. 若以上任何一步失败，或 `confidence < threshold`（默认 0.6），或 LLM 调用抛异常 → 走 `rule_based_classify(query)`。

### 4.3 规则兜底的 7 条正则

顺序有意义（从最特殊到最一般）：

```
comparative  : \b(compare|comparison|vs\.?|difference between)\b
multi_hop    : \b(impact|effect|influence|relation|between \w+ and \w+)\b
analytical   : \b(why|how come|explain why)\b
follow_up    : ^(and|then|what about|how about)\b  |  \b(her|him|them|it|this|that|these|those)\b\s*\??$
exploratory  : \b(tell me about|explain|describe|overview)\b
factual      : \b(who|what|when|where|which)\b
default      : factual (confidence=0.4)
```

这套规则的意义不是"想要做规则分类"，而是**保证在 Mock Provider / 零 API key 场景下，`tests/test_intent_agent_classifier.py` 能稳定覆盖 6 类意图**——这正是 plan.md §实施注意事项"Mock 优先"的兑现方式。

### 4.4 IntentReasoner：纯函数映射

`reasoning.py::IntentReasoner.plan(query, intent) -> ReasoningPlan` 是一个无副作用的查表函数。它把 `Intent` 转成 `ReasoningPlan(intent, strategy_name, steps, params)`，并在 `query.session_id` 存在时把它塞到 `params["session_id"]`。没有 LLM 调用、没有 I/O，利于单测稳定。

---

## 5. 阶段 3 — Decision（策略选择器）

`agent.py::StrategySelector` 只做一件事：

```python
name = f"intent_agent.{plan.strategy_name}"
cls = registry.get("strategy", name)   # 查表
return cls(retriever=self.base_retriever)
```

两个设计细节：

1. **缓存**：同一 session 多次触发同一策略不会反复构造（`self._cache`）；
2. **兜底**：`registry.get` 抛 `PluginNotFoundError` 时，logger.warning 后回退到 `intent_agent.direct`。这保证"LLM 幻觉出不存在的策略名"也不会让系统崩掉，而是优雅降级。

这种"**查表 + fallback 到 direct**"的模式让**第三方新增策略时无需改 IntentAgent 本体**——只要用 `@register("strategy", "intent_agent.my_new")` 就会被 Decision 阶段正确发现。

---

## 6. 阶段 4 — Execution（6 种检索策略）

全部策略继承 `BaseRetrievalStrategy`（`interfaces/strategy.py`，async 接口），签名统一为：

```python
async def execute(self, query: Query, plan: ReasoningPlan) -> RetrievalResult: ...
```

| 策略 | 注册名 | 对应意图 | 核心逻辑 |
|---|---|---|---|
| **DirectStrategy** | `intent_agent.direct` | factual | 直接委托 base retriever 一次 top-k 检索 |
| **MultiHopStrategy** | `intent_agent.multi_hop` | multi_hop | 临时把 `retriever.neighbor_hops` 放大到 `max(original, 2)`，检索后 try/finally 还原 |
| **ParallelStrategy** | `intent_agent.parallel` | comparative | 用正则 `\s+(and|vs|versus|compared to)\s+` 切 sub-query，`asyncio.gather` 并发，按 chunk_id 去重合并（取最高 score） |
| **IterativeStrategy** | `intent_agent.iterative` | analytical | MVP：单轮直检（plan.md §核心功能 第 3 条"反馈循环 TODO"留在后续） |
| **ExpansionStrategy** | `intent_agent.expansion` | exploratory | 在 `neighbor_hops` 基础上再加 `extra_hops=2`，取更广邻域 |
| **ContextAwareStrategy** | `intent_agent.context_aware` | follow_up | 从 `plan.params["history"]` 取最近 N 轮拼到 query 前，`session_id` 保留透传 |

### 6.1 MultiHop / Expansion 的可逆副作用

两个策略都需要临时修改 `base_retriever.neighbor_hops`，采用 `try/finally` 还原：

```python
original = getattr(self.retriever, "neighbor_hops", None)
try:
    if original is not None:
        self.retriever.neighbor_hops = ...
    res = self.retriever.retrieve(query, top_k=self.top_k)
finally:
    if original is not None:
        self.retriever.neighbor_hops = original
```

这是并发策略共用同一 base retriever 时**不会相互污染**的前提。

### 6.2 Parallel 的合并逻辑

```
for each sub-query (A, B, ...):
    run retrieve(sub) concurrently via asyncio.gather
merge chunks: dict[chunk_id -> (chunk, max_score)]  （同一 chunk 只记一次，取最高分）
sort by score descending
merge triples: 按 (s,p,o) 去重保序
```

这个设计保证 `top_k` 预算**不会被相同 chunk 浪费**——对比较类查询是否命中同一个支撑段落很常见。

### 6.3 ContextAware 的身份保留

```python
effective_text = " ".join([*history[-N:], query.text])
res = self.retriever.retrieve(
    Query(text=effective_text, session_id=query.session_id),  # session_id 必须透传
    top_k=self.top_k,
)
```

`session_id` 的透传保证本轮结束后 `IntentAgentRetriever.retrieve()` 末尾还能正确 append 回 Memory，形成闭环。

---

## 7. 三个可插拔 Hook

plan.md §核心功能 第 3 条列出的三个 Hook 全部已落地（对应 plan.md §目录结构 第 257-259 行）：

### 7.1 HeuristicReflectionHook（reflector 插槽）

`hooks/reflection_hook.py` 实装一个**零 LLM** 的启发式答案质量评估：

- **refusal 检测**：答案若包含 "I don't know / not enough information / unknown" → quality=0.2；
- **length score**：$\min(1, \text{len}/\text{min\_length}/4)$；
- **query token overlap**：$\dfrac{|q \cap a|}{|q|}$；
- **evidence 存在**：有 `evidence_chunk_ids` 得 1.0，否则 0.5；
- 综合：$\text{quality} = 0.5 \cdot \text{relevance} + 0.5 \cdot \text{completeness}$。

plan.md §目录结构 第 257 行原本规划的是 `LLMReflectionHook`；当前版本选择了启发式以满足"Mock 优先 / 零 API key"的 CI 需求。升级到 LLM 版本只是换个注册名的工作量。

当前 `IntentAgentRetriever.retrieve()` 对 reflector 只做**观测性写回**（`result.metadata["reflection_quality"]`），并**不**触发重检索——这正是 plan.md §核心功能 第 3 条标注 "反馈循环 TODO" 的部分，后续会在一个 `AdaptiveExecutor` 中加入 while-loop。

### 7.2 TokenOverlapRerankerHook（reranker 插槽）

`hooks/reranking_hook.py` 用 token Jaccard 对检索结果重排：

$$
\text{score}(c) = \dfrac{|\text{tokens}(q) \cap \text{tokens}(c)|}{|\text{tokens}(q) \cup \text{tokens}(c)|}
$$

得分降序，可选 `top_n` 截断。选择 Jaccard 而非 cross-encoder 同样出于零依赖考量（plan.md §目录结构 第 258 行原计划 `CrossEncoderRerankingHook`）——`sentence-transformers` 不是 MVP 必需项。

### 7.3 ParallelRetrievalPool（工具类）

`hooks/parallel_pool_hook.py` 是一个 `asyncio.Semaphore` 限流的小工具：

```python
pool = ParallelRetrievalPool(max_workers=4)
results = await pool.run([coro1, coro2, ...])  # 顺序与输入一致
```

它不占用任何 slot，作为一个独立的公共设施存在，便于未来批量评测 / 并发子检索复用同一限流器。`ParallelStrategy` 当前使用的是直接 `asyncio.gather`，切换到 pool 只是一行代码。

---

## 8. IntentAgentRetriever：四阶段装配

`agent.py::IntentAgentRetriever.retrieve()` 是整个插件唯一对外暴露的 `BaseRetriever` 入口，签名**保持同步**（以兼容 `QueryPipeline.run()` 的同步上下文），内部用 `_run_async` 承接 async 策略：

```python
def retrieve(self, query, top_k=5):
    # Stage 1. Memory
    history = self.memory.get_history(query.session_id) if self.memory and query.session_id else []

    # Stage 2. Reasoning
    intent = self.classifier.classify(query)
    plan = self.reasoner.plan(query, intent)
    if history:
        plan.params.setdefault("history", history)

    # Stage 3. Decision
    strategy = self._selector.pick(plan)

    # Stage 4. Execution + optional hooks
    result = _run_async(strategy.execute(query, plan))
    if self.reranker is not None:
        result = _run_async(self.reranker.rerank(query, result))
    if self.reflector is not None:
        dummy_answer = Answer(text=" ".join(c.text for c in result.chunks))
        feedback = _run_async(self.reflector.evaluate(query, dummy_answer))
        result.metadata["reflection_quality"] = feedback.quality

    # 闭环：把本轮写回 Memory
    if self.memory and query.session_id:
        self.memory.append(query.session_id, query.text)

    # 贯穿 trace：UI / 评测 / 论文分析共享这些字段
    result.metadata["intent"] = intent.label
    result.metadata["intent_confidence"] = intent.confidence
    result.metadata.setdefault("strategy", plan.strategy_name)

    # 遵守 top_k 合约
    if len(result.chunks) > top_k:
        result.chunks = result.chunks[:top_k]
        result.scores = result.scores[:top_k]
    return result
```

几个要点：

- **同步外观 + 异步内核**：`retrieve` 是同步，策略是 async。`_run_async` 是一个小包装器（复用自 `defaults/llm_extractor.py`），让两种签名共存；
- **trace 透传**：`intent / intent_confidence / strategy / hops / parallel_subqueries / history_turns_used / reranked_by / reflection_quality` 都落在 `result.metadata` 里，通过 `QueryPipeline.run()` 继续落到 `Answer.trace.retrieval_metadata`，最终被 `/api/query` 返回、被 `/trace` 页面展示；
- **top_k 合约**：不管策略内部取了多少 chunk，对外一定截断到 top_k——这是 `BaseRetriever` 的接口契约。

---

## 9. 在 QueryPipeline 中的接入

源码：`src/chimera_rag/core/pipeline.py::QueryPipeline.run()`

```
query
  → self.intent_classifier.classify(query)   ← intent_agent.llm
  → self.retriever.retrieve(query, top_k)    ← intent_agent.strategy_selector
                                               （内部完成 M-R-D-E + hooks）
  → self.generator.generate(query, retrieval)
  → Answer.intent_label = intent.label
  → Answer.strategy_name = retrieval.metadata["strategy"]
  → Answer.trace.retrieval_metadata = retrieval.metadata
  return Answer
```

值得注意：`QueryPipeline` 自己也会跑一次 `intent_classifier.classify()`（这是 plan.md §核心功能 第 1 条基线就有的行为）。当 IntentAgent 启用时，这次分类与 IntentAgentRetriever 内部的分类**都走 `LLMIntentClassifier`**——结果一致。这条冗余在 MVP 阶段是有意保留的，以便 vanilla 配置也能单独跑通 `QueryPipeline`。

---

## 10. 对应测试

plan.md §目录结构 第 349–351 行规划的三个测试文件，实际仓库里扩展为 7 个：

| 测试文件 | 主要覆盖点 |
|---|---|
| `tests/test_intent_agent_memory.py` | append / 窗口截断 / `max_history` 合法性 |
| `tests/test_intent_agent_classifier.py` | 6 类 LLM mock 分类；规则兜底每一分支；confidence 阈值控制 |
| `tests/test_intent_agent_reasoning.py` | 6 类意图 → 正确策略名；steps 非空；session_id 透传 |
| `tests/test_intent_agent_strategies.py` | 每个策略的独立行为；`neighbor_hops` 副作用可逆 |
| `tests/test_intent_agent_hooks.py` | 三个 hook 的返回结构；相似度 / 启发式评分边界 |
| `tests/test_intent_agent_agent.py` | 四阶段串联；top_k 合约；metadata 透传 |
| `tests/test_pipeline_intent_agent.py` | 端到端：`intent_agent_only` 配置的 `ingest → query` 用 MockProvider 跑通 |

另有 `tests/test_pipeline_full.py` 覆盖 **AdaGraph × IntentAgent** 同时启用的完整链路，对应 plan.md §核心功能 第 1 条"两个创新点完全关闭 / 完全启用 / 任一单开"四种部署都应该跑通的要求。

---

## 11. 现状与 honest roadmap

严格按照 plan.md §核心功能 第 3 条"反馈循环 TODO"以及 §目录结构 第 246、257-258 行的标注，下列功能**接口已就位，实装延后到下一轮 TDD**：

| 方向 | 当前状态 | 下一步 |
|---|---|---|
| `AdaptiveExecutor` 的反馈循环 | IntentAgentRetriever 只做观测写回 | 低质量答案触发策略升级 / 重检索的 while-loop |
| `IterativeStrategy` 的真正分解 | 单轮直检 | LLM 拆 sub-question + 多轮聚合 |
| `LLMReflectionHook` | 由启发式 hook 占位 | LLM 打分 relevance / completeness |
| `CrossEncoderRerankingHook` | 由 Jaccard hook 占位 | sentence-transformers cross-encoder |

代码内均有 `# TODO` / `# Reserved` / `MVP` 字样标识，新 TDD 循环落地时无需改动已稳定测试。

---

## 12. 与 AdaGraph 的正交性

IntentAgent **只读** `PipelineState.graph_store / vector_store / chunk_lookup`——也就是说，它**不关心**这些存储是怎么被构建出来的。因此：

- 开启 `adagraph_only`：IntentAgent 关闭，只检查"图好了之后裸检索效果"；
- 开启 `intent_agent_only`：AdaGraph 关闭，只检查"固定图上意图感知的效果"；
- 开启 `full`：两创新点叠加，观察超加性增益。

这种结构让双创新点的贡献可以被**独立量化**，也是 plan.md §实施策略所说"关闭创新点就是改一行 yaml"的工程兑现。AdaGraph 的实装细节见 `docs/innovation_1_adagraph.md`。
