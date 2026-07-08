# Chimera-RAG 文档目录

本目录存放项目的设计与实现文档。

## 文档清单

| 文件 | 用途 |
|------|------|
| `architecture_overview.md` | **架构总览**：分层、接口插槽、注册表机制、`ChimeraRAG` 门面装配流程、多Agent内部架构、四种部署组合 |
| `plan.md` | 项目最初的实施计划 |
| `innovation-1/` | **创新点一：AdaGraph** 文档 |
| `innovation-1/adagraph.md` | AdaGraph 实装细节：DynamicChunker / LayeredTripleExtractor / DualRedundancyPruner |
| `innovation-2/` | **创新点二：CollaRAG** 文档 |
| `innovation-2/colla-rag.md` | CollaRAG 多Agent架构实装细节 |
| `innovation-2/design.md` | 多Agent架构设计文档（5 Agent + 19 工具 + 双层记忆） |
| `innovation-2/plan.md` | 多Agent架构实现计划 |
| `experiments/` | **实验相关** 文档 |
| `experiments/` | 实验结果与实验设计文档 |
| `experiments/experiments_colla_rag.md` | CollaRAG 实验设计文档 |
| `experiments/main-results.md` | 主实验结果 |
| `experiments/ablation.md` | 消融实验与进一步分析 |
| `experiments/performance.md` | 性能优势分析 |
| `experiments/case-study.md` | 案例研究 |
| `experiments/summary.md` | 实验整理汇总 |

## 阅读顺序

1. **`architecture_overview.md`** — 先建立"注册表 + 插槽 + yaml 开关"这套骨架的心智模型，以及多Agent内部架构；
2. **`innovation-1/adagraph.md`** — 深入创新点一的三个子模块；
3. **`innovation-2/design.md`** — 创新点二的多Agent架构完整设计；
4. **`experiments/`** — 动手做实验前读。

## 注册表插槽速查

AdaGraph 占 3 个构建插槽，CollaRAG 占 2 个查询插槽：

| 插槽（slot） | 默认实现 | AdaGraph | CollaRAG |
|---|---|---|---|
| `chunker` | `defaults.fixed` | `adagraph.dynamic` | — |
| `extractor` | `defaults.simple_llm` | `adagraph.layered` | — |
| `pruner` | `defaults.noop` | `adagraph.dual_redundancy` | — |
| `intent_classifier` | `defaults.rule` | — | `colla_rag.tree`（树形分类：greeting / single_hop / multi_hop / summarization / other） |
| `retriever` | `defaults.direct` | — | `colla_rag.multi_agent`（MultiAgentRetriever：5 Agent + 19 工具 + 双层记忆） |
| `generator` | `defaults.prompt` | — | — |