# 创新点 2b · LangGraph Agentic 检索 — 让 Decision 阶段"自主编排"

> 🧬 本文是创新点 2（IntentAgent）的**进阶演化**：把现在写死的"意图 → 单策略"静态查表（`StrategySelector`），升级为由 **LangGraph 状态图**驱动、**LLM Agent 自主选择并组合多个检索工具**的动态编排。它作为**第三种检索器形态**与 `defaults.direct`（vanilla）、`intent_agent.strategy_selector`（静态路由）并列，通过一行 yaml 切换，可被独立消融对比。

---

## 📖 目录

- [0. 一句话定位](#0-一句话定位)
- [1. 为什么要做这件事（动机）](#1-为什么要做这件事动机)
- [2. 设计目标与硬约束](#2-设计目标与硬约束)
- [3. 整体架构：第三种检索器](#3-整体架构第三种检索器)
- [4. 工具层：把现有原子能力封装成 LangChain Tools](#4-工具层把现有原子能力封装成-langchain-tools)
- [5. 编排层：LangGraph 状态图](#5-编排层langgraph-状态图)
- [6. 反思回路：把 README 里的 TODO 真正落地](#6-反思回路把-readme-里的-todo-真正落地)
- [7. LLM 适配：复用现有 providers，不引第二套 client](#7-llm-适配复用现有-providers不引第二套-client)
- [8. 配置开关](#8-配置开关)
- [9. 依赖与降级策略](#9-依赖与降级策略)
- [10. 与现有架构的兼容性（三条 CI 红线）](#10-与现有架构的兼容性三条-ci-红线)
- [11. 消融实验设计](#11-消融实验设计)
- [12. 测试计划（TDD）](#12-测试计划tdd)
- [13. 落地步骤（roadmap）](#13-落地步骤roadmap)
- [14. 风险与权衡](#14-风险与权衡)

---

## 0. 一句话定位

> **从「意图 → 单策略」的静态查表，演化为「LLM Agent 在 LangGraph 状态图里自主选择 / 组合 / 迭代多个检索工具」的 Agentic 检索。**

它不重写任何检索逻辑——所有底层能力（向量召回、图谱跳转、多跳、并行、质量自评）都复用现有代码，只是被封装成 **LangChain Tools**，再由 LangGraph 编排。

---

## 1. 为什么要做这件事（动机）

现有 IntentAgent 的 Decision 阶段（`reasoning.py::_INTENT_TO_STRATEGY` + `agent.py::StrategySelector`）是一张**写死的查表**：意图分类完成后，锁定唯一一个策略跑到底。这套设计在 MVP 阶段足够清晰，但存在三个本质局限：

| 局限 | 现状 | 后果 |
|---|---|---|
| **意图是离散单标签** | 一个 query 只能命中 6 类之一 | "比较 A 和 B 为什么不同"这类**跨意图**问题被迫二选一 |
| **策略一锤子买卖** | 选定策略后单轮执行 | 检索质量差时**没有补救机会** |
| **反思只做观测** | `reflection_quality` 仅写进 metadata | README §11 标注的"反馈循环"始终是 TODO，**未真正驱动二次检索** |

LangGraph 方案恰好对症下药：

- **多工具组合** 替代 单策略 → 解决跨意图
- **状态图 + conditional edge** 替代 一次性执行 → 让低质量结果触发补检
- **显式反思回路** 替代 观测写回 → 把 TODO 真正落地

这构成一个**算法层面**的创新点（而非纯工程封装）：检索决策从"规则路由"跃迁到"自主规划"。

---

## 2. 设计目标与硬约束

### 2.1 目标

1. **复用优先**：不重写任何检索算法，全部封装现有 `strategies/*` 与 `stores/*`。
2. **可消融**：作为独立检索器与静态版并列，能定量对比"动态编排 vs 静态路由"的增益。
3. **可降级**：未安装 `langgraph` / `langchain-core` 时，系统其余 315 个测试与配置不受任何影响。

### 2.2 硬约束（继承自项目三条红线）

1. `core/` ❌ 不 import `plugins/`
2. `plugins/adagraph/` 与 `plugins/intent_agent/` ❌ 互不依赖
3. 关闭本检索器（回退到 `defaults.direct` 或 `intent_agent.strategy_selector`）后，系统仍能端到端跑通

> 本方案选择"**新增而非替换**"：保留 `StrategySelector` 作为消融基线，新代码全部以增量形式落在 `plugins/intent_agent/langgraph/` 子目录下，`git diff` 可清晰呈现纯增量改动。

---

## 3. 整体架构：第三种检索器

三种检索器形态对外都实现同一个 `BaseRetriever.retrieve(query, top_k) -> RetrievalResult` 契约，对 `QueryPipeline` 完全透明：

| 检索器 | 注册名 | Decision 机制 | 角色 |
|---|---|---|---|
| `DirectRetriever` | `defaults.direct` | 无（直接向量召回） | 📉 vanilla 基线 |
| `IntentAgentRetriever` | `intent_agent.strategy_selector` | **静态查表** | 🧠 创新点 2（已有） |
| `LangGraphAgentRetriever` | `intent_agent.langgraph` | **LLM 动态编排** | 🚀 创新点 2b（本文，新增） |

新增检索器的内部装配（M-R-D-E 四阶段保留，仅 D 阶段被替换）：

```
            ┌────────────────────────────────────────────────┐
            │  LangGraphAgentRetriever (BaseRetriever)        │
            │  同步外观 retrieve()  ←  _run_async(graph.ainvoke)│
            └──────────────────────┬─────────────────────────┘
                                   │
   ┌───────────────────────────────▼───────────────────────────────┐
   │                    LangGraph StateGraph                         │
   │                                                                │
   │  START → [Memory] → [Reasoning] → [Decision(LLM Agent)]        │
   │                                        │  ▲                     │
   │                                        ▼  │ (补检)              │
   │                                  [Tool Node]                    │
   │                                        │                        │
   │                                        ▼                        │
   │                                   [Reflect] ──质量够?──► END     │
   │                                        └──不够&未超限──► Decision │
   └────────────────────────────────────────────────────────────────┘
                                   │
                         复用底层能力（不重写）
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        │ vector_store │ graph_store  │ strategies/* │ hooks/*      │
        └──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 4. 工具层：把现有原子能力封装成 LangChain Tools

所有 Tool 都是对**已有能力的薄封装**（thin wrapper），不含新检索逻辑。建议落在 `plugins/intent_agent/langgraph/tools.py`。

| Tool 名 | 入参 | 封装的现有能力 | LLM 何时调用 |
|---|---|---|---|
| `vector_search` | `query: str, top_k: int` | `base_retriever.retrieve` / `vector_store` | 需要语义召回 chunk |
| `graph_neighbors` | `entity: str, hops: int` | `graph_store.get_neighbors(entity, hops)` | 顺着实体在图上跳转 |
| `multi_hop_retrieve` | `query: str` | 现有 `MultiHopStrategy.execute` | 桥接型多跳问题 |
| `parallel_retrieve` | `query: str` | 现有 `ParallelStrategy.execute` | 可拆分的比较类问题 |
| `assess_evidence` | `query: str` | 现有 `HeuristicReflectionHook.evaluate` | 自评当前证据是否充分 |

### 4.1 Tool 的统一契约

- 每个 Tool 用 `langchain_core.tools.tool` 装饰（或 `StructuredTool.from_function`），入参用 **Pydantic schema** 严格约束，与项目"显式校验"风格一致。
- Tool 返回的不是裸文本，而是把召回的 `chunk_id` 列表写进**共享状态**（见 §5.2），避免大段文本反复进 LLM 上下文、控制 token 成本。
- Tool 内部统一通过 `_run_async`（复用自 `defaults/llm_extractor.py`）承接现有 async 策略，保持同步外观。

### 4.2 工具实例化注入

Tool 需要持有 `base_retriever / graph_store / vector_store` 等运行时依赖。采用**闭包工厂**模式：`build_tools(deps) -> list[BaseTool]`，在 `LangGraphAgentRetriever.__init__` 时一次性构造，避免全局单例污染并发。

---

## 5. 编排层：LangGraph 状态图

落在 `plugins/intent_agent/langgraph/graph.py`。

### 5.1 节点（Node）

| 节点 | 复用的现有组件 | 职责 |
|---|---|---|
| `memory_node` | `BaseMemory`（`SessionMemory`） | 读 session 历史 → 写入 state |
| `reasoning_node` | `LLMIntentClassifier` | 意图分类 → 写入 state（仅作为 Agent 的**提示**，不再硬路由） |
| `decision_node` | LLM（经 §7 适配器）+ 绑定的 tools | 让 LLM 看 state，决定调用哪个 / 哪些 tool |
| `tool_node` | LangGraph `ToolNode` | 执行 LLM 选中的 tool，observation 回灌 state |
| `reflect_node` | `HeuristicReflectionHook` | 评估累积证据质量，决定是否继续 |

### 5.2 共享状态（AgentState）

用 `TypedDict`（LangGraph 推荐）定义，字段对应四阶段产物：

```
class AgentState(TypedDict):
    query: str
    session_id: str | None
    intent: str                       # reasoning_node 写
    messages: list                    # LLM ↔ tool 的对话轨迹（LangGraph add_messages reducer）
    accumulated_chunk_ids: list[str]  # tool_node 累积，去重
    accumulated_triples: list         # 图谱跳转累积
    iteration_count: int              # 反思回路计数
    quality: float                    # reflect_node 写
```

`accumulated_*` 字段用 LangGraph 的 **reducer**（如 `operator.add` + 去重）做增量合并，保证多轮 tool 调用的结果不互相覆盖。

### 5.3 边（Edge）

```
START          → memory_node
memory_node    → reasoning_node
reasoning_node → decision_node
decision_node  → (conditional) tools_condition:
                    若 LLM 发起 tool_call → tool_node
                    若 LLM 直接给结论     → reflect_node
tool_node      → reflect_node
reflect_node   → (conditional) should_continue:
                    quality >= τ  或  iteration_count >= MAX_ITERS → END
                    否则                                            → decision_node
```

`MAX_ITERS`（默认 2）与质量阈值 `τ`（默认 0.6，复用 `HeuristicReflectionHook` 口径）都来自配置，防止死循环。

### 5.4 出口：状态 → RetrievalResult

图跑完后，`LangGraphAgentRetriever` 把 `state.accumulated_chunk_ids` 经 `chunk_lookup` 还原成 `Chunk`，组装成 `RetrievalResult`，并把决策轨迹写进 `metadata`：

```
metadata = {
    "retriever": "langgraph",
    "intent": state["intent"],
    "tool_calls": [...],            # 调用了哪些 tool、顺序、参数
    "iterations": state["iteration_count"],
    "final_quality": state["quality"],
}
```

这套 `metadata` 沿用现有透传链路（`QueryPipeline.run` → `Answer.trace.retrieval_metadata` → `/api/query` → 前端 trace 页），论文里的"决策轨迹图"可直接由此渲染。

---

## 6. 反思回路：把 README 里的 TODO 真正落地

现有 `IntentAgentRetriever` 对 reflector 只做**观测写回**（README §11 列为 TODO）。本方案在 `reflect_node` + `should_continue` conditional edge 上**真正实现** Reflexion 式回路：

1. 每轮 tool 执行后，`reflect_node` 用 `HeuristicReflectionHook` 对"当前累积证据拼成的 draft answer"打分；
2. 质量 `< τ` 且 `iteration_count < MAX_ITERS` → 回到 `decision_node`，LLM 在已有 observation 基础上**换工具 / 加跳数 / 拆子问题**重试；
3. 质量达标或触顶 → 结束。

其中阈值 `τ` 与上限 `MAX_ITERS` **直接复用** `default.yaml` 里既有的 `plugins.intent_agent.hooks.reflection.quality_threshold` / `max_iterations`（见 §8），不新增配置字段。

这正是 `docs/innovation_2_intent_agent.md` §11 中标注的 `AdaptiveExecutor` while-loop，在 LangGraph 里以声明式状态图形式自然落地，无需手写循环控制。

---

## 7. LLM 适配：复用现有 providers，不引第二套 client

项目已有完整的 `providers/`（openai / deepseek / ollama / st / mock），统一实现 `BaseLLMProvider.complete(prompt) -> str`。**绝不**再引入 `langchain-openai` 等重复 client。

适配方案（落在 `langgraph/llm_adapter.py`）：

- 写一个轻量 `ChimeraChatModel`，继承 `langchain_core.language_models.BaseChatModel`，内部委托给现有 `BaseLLMProvider`。
- **关键问题**：LangGraph 的 Agent 依赖模型的 **tool-calling** 能力，而现有 `BaseLLMProvider` 只有纯文本 `complete`。两种落地路径：
  - **路径 A（推荐，零 provider 改动）**：用 **ReAct 文本协议**——在 prompt 里描述 tools，让 LLM 输出 `Action: tool_name / Action Input: {...}` 文本，适配器解析后映射成 LangGraph tool_call。优点：MockProvider 也能跑通，满足"零 API key 测试"。
  - **路径 B**：给 `BaseLLMProvider` 增加可选的 `complete_with_tools()`，对接原生 function-calling。增益是更稳，代价是要动 provider 接口（触及核心约定，留作后续）。

> MVP 采用**路径 A**，与项目"Mock 优先、零 API key CI"的总则一致。

---

## 8. 配置开关

`configs/` 新增一份预设 `deepseek_intent_langgraph.yaml`。它从 `deepseek_intent_only.yaml` 派生，**复用现有配置结构与字段**，关键差异只有两处：

1. `query.retriever.active` 切到新检索器；
2. **复用** `plugins.intent_agent.hooks.reflection`（`default.yaml` 里已存在的 `quality_threshold` / `max_iterations` 字段）来驱动反思回路——不另造新字段。

```yaml
# 1) 检索器切换点（与现有 query.retriever 结构完全一致）
query:
  retriever:
    active: intent_agent.langgraph          # ← 唯一切换点（原值 intent_agent.strategy_selector）
    params:
      top_k: 5
      neighbor_hops: 1
      llm_protocol: react                   # react | function_calling（见 §7）
      tools: [vector_search, graph_neighbors, multi_hop_retrieve, parallel_retrieve, assess_evidence]

# 2) 反思回路复用现有 hooks.reflection 字段（无需新增配置项）
plugins:
  intent_agent:
    enabled: true
    hooks:
      reflection:
        enabled: true
        quality_threshold: 0.6              # 复用现有字段：reflect_node 的退出阈值 τ
        max_iterations: 2                   # 复用现有字段：反思回路上限 MAX_ITERS
```

> **字段归属说明**（对照 `default.yaml` 真实结构）：检索器自身参数（`top_k` / `neighbor_hops` / `llm_protocol` / `tools`）落在 `query.retriever.params`；反思回路阈值复用 `plugins.intent_agent.hooks.reflection.{quality_threshold, max_iterations}`。本方案**不**引入 `query.langgraph` 这类新顶层节点。

切回静态版只需把 `active` 改回 `intent_agent.strategy_selector`；切回 vanilla 改回 `defaults.direct`。

---

## 9. 依赖与降级策略

新增**可选 extra**，不入主依赖。采用 **LangGraph / LangChain 1.x GA（LTS）稳定版**：

```toml
# pyproject.toml
[project.optional-dependencies]
agent = [
    "langgraph>=1.0,<2",
    "langchain-core>=1.0,<2",
]
```

安装：`uv sync --extra agent`（或 `uv add --optional agent "langgraph>=1.0,<2" "langchain-core>=1.0,<2"`）。

> **为何用 1.x**：LangGraph 1.0 是稳定性导向的 LTS 发布，**核心 `StateGraph` / `ToolNode` / conditional-edge API 保持不变**，承诺到 2.0 前无破坏性变更。本方案用底层 `StateGraph` 自行编排（见 §5），**不**依赖 1.x 中已废弃的 `langgraph.prebuilt.create_react_agent`（其迁移目标 `langchain.agents.create_agent` 我们也不需要），因此天然规避了 1.x 唯一的破坏性变更。

**降级**：`langgraph/__init__.py` 用 `try/except ImportError` 守护，import 失败时注册逻辑跳过 `intent_agent.langgraph` 的注册（该 slot/name 不进 registry）。

误配场景的处理需要正视一个**现状事实**：`src/chimera_rag/__init__.py::ChimeraRAG._build()` 当前用的是 `registry.get("retriever", cfg.query.retriever.active)`（见该文件第 87 行），缺失时直接抛 `PluginNotFoundError`，**并未**走 `get_or_default`。因此有两条可选路径：

- **路径甲（零核心改动，推荐 MVP）**：不改 `_build()`。未装 extra 又误配 `active: intent_agent.langgraph` 时，`PluginNotFoundError` 会带着清晰的 `slot/name` 信息抛出。代价是错误信息不含"装 extra"指引——可在 `langgraph/__init__.py` 的 import 守护处先 `logger.warning` 一条安装提示作为补偿。
- **路径乙（一行核心改动，更友好）**：把 `_build()` 第 87 行的 `registry.get` 换成 `core/registry.py` 已有的 `registry.get_or_default("retriever", cfg.query.retriever.active, default="defaults.direct")`，缺失时自动回退到 vanilla 检索器并打 warning。`get_or_default` 是现成原语，但此改动**触及 core**，需在 PR 中显式标注（你已授权"可适度调整核心约定"）。

无论哪条路径，未装 extra 的环境，现有 315 个测试与 4 份 yaml 配置**零影响**（新检索器仅在被显式配置时才参与）。

---

## 10. 与现有架构的兼容性（三条 CI 红线）

| 红线 | 本方案是否满足 | 说明 |
|---|---|---|
| `core/` 不依赖 `plugins/` | ✅ | 全部新代码在 `plugins/intent_agent/langgraph/`。走**路径甲**时 `core` 完全不动；走**路径乙**时仅在 `__init__.py` 第 87 行改 `get` → `get_or_default`，仍不 import 任何 plugin |
| 两 plugin 互不依赖 | ✅ | 不 import `plugins/adagraph/` 任何内容；只读 `PipelineState` 里的 store |
| 关插件可降级 | ✅ | 切回 `defaults.direct` / 静态版即降级；extra 未装时按 §9 路径甲/乙处理 |

`tests/test_plugin_toggle_regression.py` 守护的依赖方向，本方案完全遵守——所有跨边界调用仍只走 `interfaces/*` ABC + `registry`。

### 10.1 一处必须正视的现状限制：Web 热切换

`src/chimera_rag/__init__.py::toggle_plugin("intent_agent", True)`（该文件第 165 行起）目前**硬编码**把 `query.retriever.active` 切到 `intent_agent.strategy_selector`（静态版），无法切到 `intent_agent.langgraph`。因此：

- **通过 yaml 配置启动**（`active: intent_agent.langgraph`）→ ✅ 正常生效；
- **通过 Web `/api/plugins/toggle` 热切换** → ❌ 只会切到静态版，**切不到 LangGraph 版**。

落地时需要二选一（这是一处必须做的取舍，不能默认"自动支持"）：

- **不动 `toggle_plugin`**：LangGraph 版仅支持 yaml 启动，Web 热切换维持现有"开/关静态 IntentAgent"语义（MVP 推荐，零核心改动）；
- **扩展 `toggle_plugin`**：增加一个"检索器变体"参数或在前端暴露第三档，使 Web 也能切到 langgraph。此改动触及 `core/` 的 `__init__.py` 与 `web/` 路由，工作量更大，留作后续增强。

---

## 11. 消融实验设计

新增检索器后，消融矩阵从 4 组扩展到 **5 组**，可定量回答"动态编排是否优于静态路由"：

| 配置 | AdaGraph | 检索器 | 论文里回答的问题 |
|---|:---:|---|---|
| `deepseek_vanilla` | ❌ | `defaults.direct` | 基线 |
| `deepseek_intent_only` | ❌ | `intent_agent.strategy_selector` | 静态意图路由的增益 |
| **`deepseek_intent_langgraph`** | ❌ | **`intent_agent.langgraph`** | **动态编排 vs 静态路由的增量** |
| `deepseek_adagraph_only` | ✅ | `defaults.direct` | 图谱构建的增益 |
| `deepseek_full` | ✅ | `intent_agent.langgraph` | 双创新点叠加 |

核心对照：**`intent_only` vs `intent_langgraph`** —— 同数据集、同图谱、同评测逻辑，差异只在 Decision 机制，直接隔离出"Agentic 编排"的贡献。

观测指标（复用现有 `eval.py` 报告）：

- EM / F1 / ROUGE-L（尤其看 `multi_hop` / `comparative` 这两类**跨意图**问题的分组指标）
- 平均 tool 调用次数、平均反思迭代次数（新增 metadata 字段）
- 延迟与 Token 成本（动态编排预期更贵，需量化 trade-off）

实验设计文档建议补到 `docs/experiments/` 目录下，与现有创新点 2 实验文档并列。

---

## 12. 测试计划（TDD）

沿用项目 RED → GREEN → REFACTOR → COMMIT 节奏，全部用 MockProvider，零 API key：

| 测试文件 | 覆盖点 |
|---|---|
| `tests/test_langgraph_tools.py` | 5 个 tool 的入参校验、返回结构、对底层能力的正确委托 |
| `tests/test_langgraph_llm_adapter.py` | ReAct 文本协议解析；MockProvider 输出 → tool_call 映射 |
| `tests/test_langgraph_graph.py` | 状态图节点/边；反思回路在低质量时触发补检、达标/触顶时正确退出 |
| `tests/test_langgraph_retriever.py` | `LangGraphAgentRetriever` 的 `BaseRetriever` 契约：top_k、metadata 透传 |
| `tests/test_pipeline_langgraph.py` | 端到端 `ingest → query`（`intent_langgraph` 配置）用 Mock 跑通 |
| `tests/test_langgraph_degradation.py` | 未装 extra 时优雅报错；装了 extra 时正常注册 |

并补充进 `test_plugin_toggle_regression.py` 的断言，确保三条红线不被打破。

---

## 13. 落地步骤（roadmap）

1. **依赖**：`pyproject.toml` 加 `agent` extra → `uv sync --extra agent`。
2. **骨架**：建 `plugins/intent_agent/langgraph/` 子包（`__init__.py` 带 ImportError 守护）。
3. **TDD-1 工具层**：先写 `test_langgraph_tools.py`（RED）→ 实现 `tools.py`（GREEN）。
4. **TDD-2 LLM 适配**：`test_langgraph_llm_adapter.py` → `llm_adapter.py`（ReAct 协议）。
5. **TDD-3 状态图**：`test_langgraph_graph.py` → `graph.py`（含反思回路）。
6. **TDD-4 检索器**：`test_langgraph_retriever.py` → `retriever.py`（注册 `intent_agent.langgraph`）。
7. **TDD-5 端到端**：`test_pipeline_langgraph.py` + 新 yaml `deepseek_intent_langgraph.yaml`。
8. **回归**：跑全量 `uv run pytest -q`，确认原有 315 测试不挂；`ruff check`。
9. **文档**：补 `docs/experiments/` 实验设计 + 更新 README 的消融矩阵表与文档索引。

---

## 14. 风险与权衡

| 风险 | 影响 | 缓解 |
|---|---|---|
| LangGraph/LangChain API 演进快 | 版本升级可能 break | 锁定 `langgraph>=0.2, langchain-core>=0.3` 并写进 `uv.lock`；适配层薄、迁移面小 |
| ReAct 文本协议解析脆弱 | LLM 输出格式飘 → tool_call 失败 | 解析失败时 fallback 到 `vector_search` 单工具（等价静态 direct），保证不崩 |
| 动态编排延迟/Token 更高 | 成本上升 | `MAX_ITERS` 限流 + 在报告里显式量化 trade-off，让"贵但准"可被论证 |
| MockProvider 无真实推理能力 | 测试只能验证流程不验证质量 | 流程正确性用 Mock 覆盖；检索质量增益在 §11 真实 LLM 消融实验里验证 |
| 与静态版能力重叠 | 审稿质疑"是否多余" | 用 §11 的 `intent_only` vs `intent_langgraph` 对照，定量证明跨意图问题上的增益 |

---

## 📎 关联文档

| 文档 | 关系 |
|---|---|
| [`innovation_2_intent_agent.md`](innovation_2_intent_agent.md) | 本文的前身：静态路由版 IntentAgent（M-R-D-E 四阶段、6 策略、3 hook） |
| [`architecture_overview.md`](architecture_overview.md) | 注册表 / 插槽 / 依赖注入骨架，本文新检索器即按此骨架接入 |
| [`innovation_1_adagraph.md`](innovation_1_adagraph.md) | 正交的构建侧创新点，与本文检索侧创新点可叠加 |
