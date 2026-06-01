# Chimera-RAG 文档目录

本目录存放项目的设计与实现文档，是仓库版本化的"事实来源"（single source of truth）。所有文档基于 `docs/plan.md` 与 `src/chimera_rag/` 实际实装整理，全部使用简体中文。

## 文档清单

| 文件 | 用途 |
|------|------|
| `plan.md` | 项目最初的实施计划（从 Plan 模式同步；**本目录下其他文档的上游事实来源**） |
| `architecture_overview.md` | **架构总览**：分层、14 个 ABC 插槽、注册表机制、`ChimeraRAG` 门面装配流程、四种部署组合、Web 系统单端口部署 |
| `innovation_1_adagraph.md` | **创新点一：AdaGraph（原 DEKER）**。作用于知识构建阶段。DynamicChunker（自适应公式已实装）/ LayeredTripleExtractor（L1 实装，L2-L4 TODO）/ DualRedundancyPruner（相似性实装，传递性 TODO） |
| `innovation_2_intent_agent.md` | **创新点二：IntentAgent（原 MRDE）**。作用于查询阶段。Memory / Reasoning / Decision / Execution 四阶段 + 6 种检索策略 + 3 个可插拔 Hook |
| `innovation_2b_langgraph_agent.md` | **创新点 2b：LangGraph Agentic 检索**。创新点二的进阶演化——把静态的"意图→单策略"查表换成 LLM 在 LangGraph 状态图里**自主选择 / 组合检索工具 + 反思回路**。作为第三种检索器 `intent_agent.langgraph` 与 vanilla / 静态版并列消融 |
| `experiments/` | **实验文档目录**：存放各创新点「需要做的实验」相关设计文档（可独立消融、同管线、同评测）。当前含 `experiments_intent_agent.md`（创新点 2 实验设计） |
| `README.md`（本文件） | 文档索引 |

## 阅读顺序

1. **`architecture_overview.md`** — 先建立"注册表 + 插槽 + yaml 开关"这套骨架的心智模型；
2. **`innovation_1_adagraph.md`** — 深入第一个创新点的三个子模块；
3. **`innovation_2_intent_agent.md`** — 深入第二个创新点的四阶段 + 六策略 + 三 Hook；
4. **`innovation_2b_langgraph_agent.md`** — 在创新点二基础上理解 LangGraph 动态工具编排版（创新点 2b）；
5. **`experiments/`** — 各创新点的实验设计文档（消融对比、按意图分组归因、SOTA 横向对比等），动手做实验前读；
6. **`plan.md`** — 如需追溯最初的设计决策再读。

## 名称映射（出自 plan.md §产品概述）

| 当前代码命名 | plan.md 中标注的原设计命名 |
|---|---|
| **Chimera-RAG** | KG-RAG / Agent-Graph-RAG |
| **AdaGraph** | DEKER（知识构建双重增强） |
| **IntentAgent** | MRDE（Memory-Reasoning-Decision-Execution） |

## 注册表插槽速查

AdaGraph 占 3 个构建插槽，IntentAgent 占 6 个查询插槽，两者天然不重叠：

| 插槽（slot） | 默认实现 | AdaGraph | IntentAgent |
|---|---|---|---|
| `chunker` | `defaults.fixed` | `adagraph.dynamic` | — |
| `extractor` | `defaults.simple_llm` | `adagraph.layered` | — |
| `pruner` | `defaults.noop` | `adagraph.dual_redundancy` | — |
| `intent_classifier` | `defaults.rule` | — | `intent_agent.llm` |
| `retriever` | `defaults.direct` | — | `intent_agent.strategy_selector`（静态路由）/ `intent_agent.langgraph`（创新点 2b 动态编排，需 `--extra agent`） |
| `strategy` | — | — | `intent_agent.{direct / multi_hop / parallel / iterative / expansion / context_aware}` |
| `memory` | `defaults.none` | — | `intent_agent.session` |
| `reflector` | `defaults.noop` | — | `intent_agent.heuristic` |
| `reranker` | `defaults.noop` | — | `intent_agent.token_overlap` |
| `generator` | `defaults.prompt` | — | — |

## 开发方式

项目采用 **TDD（测试驱动开发）**：**RED → GREEN → REFACTOR → COMMIT**。所有测试默认使用 `MockLLMProvider` + `MockEmbeddingProvider`，`uv run pytest -q` 无需任何真实 API key 即可跑完。详见 `plan.md` §实施策略。
