# Chimera-RAG 文档目录

本目录存放项目的设计与实现文档。

## 文档清单

| 文件 | 用途 |
|------|------|
| `architecture_overview.md` | **架构总览**：分层、接口插槽、注册表机制、`ChimeraRAG` 门面装配流程、多Agent内部架构、四种部署组合 |
| `innovation_1_adagraph.md` | **创新点一：AdaGraph**。作用于知识构建阶段。DynamicChunker / LayeredTripleExtractor / DualRedundancyPruner |
| `innovation_intent_agent.md` | **创新点二旧版文档（历史参考）**。原 M-R-D-E 四阶段 + 6 策略 + LangGraph 动态编排 + 经验记忆增强 |
| `experiment_guide.md` | **实验指南**：三阶段工作流（ingest → infer → eval）、数据集格式、操作手册 |
| `plan.md` | 项目最初的实施计划 |
| `experiments/` | 实验设计文档目录 |
| `old/` | 归档目录：旧版创新点二的原始文档 |
| `superpowers/specs/` | **多Agent架构设计文档**：2026-07-07 版本的完整设计（5 Agent + 19 工具 + 双层记忆） |
| `superpowers/plans/` | 多Agent架构实现计划 |

## 阅读顺序

1. **`architecture_overview.md`** — 先建立"注册表 + 插槽 + yaml 开关"这套骨架的心智模型，以及多Agent内部架构；
2. **`innovation_1_adagraph.md`** — 深入创新点一的三个子模块；
3. **`superpowers/specs/2026-07-07-multi-agent-architecture-design.md`** — 创新点二的多Agent架构完整设计；
4. **`experiment_guide.md`** — 动手做实验前读。

## 注册表插槽速查

AdaGraph 占 3 个构建插槽，IntentAgent 占 2 个查询插槽：

| 插槽（slot） | 默认实现 | AdaGraph | IntentAgent |
|---|---|---|---|
| `chunker` | `defaults.fixed` | `adagraph.dynamic` | — |
| `extractor` | `defaults.simple_llm` | `adagraph.layered` | — |
| `pruner` | `defaults.noop` | `adagraph.dual_redundancy` | — |
| `intent_classifier` | `defaults.rule` | — | `intent_agent.tree`（树形分类：greeting / single_hop / multi_hop / summarization / other） |
| `retriever` | `defaults.direct` | — | `intent_agent.multi_agent`（MultiAgentRetriever：5 Agent + 19 工具 + 双层记忆） |
| `generator` | `defaults.prompt` | — | — |
