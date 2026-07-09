# Chimera-RAG 文档目录

本目录存放项目的设计与实现文档。

## 文档清单

| 文件 | 用途 |
|------|------|
| `architecture_overview.md` | **架构总览**：分层、接口插槽、注册表机制、`ChimeraRAG` 门面装配流程、多Agent内部架构、四种部署组合 |
| `innovation-1/` | **创新点一：AdaGraph** 文档 |
| `innovation-1/adagraph.md` | AdaGraph 实装细节：DynamicChunker / LayeredTripleExtractor / DualRedundancyPruner |
| `innovation-2/` | **创新点二：CollaRAG** 文档 |
| `innovation-2/colla-rag.md` | CollaRAG 多Agent架构实装细节 |
| `innovation-2/design.md` | 多Agent架构设计文档（5 Agent + 31 工具 + 双层记忆） |
| `innovation-2/plan.md` | 多Agent架构实现计划 |
| `experiments/` | **实验结果**文档（10 份实验数据表） |

## 实验文档

`experiments/` 目录下存放 10 份实验结果文档。每份文档对应 `experiments/` 目录（项目根目录）下的一个可复现脚本：

| 实验结果文档 | 对应实验脚本 | 说明 |
|---|---|---|
| `main-results-full.md` | `experiments/exp1_main_results.py` | 主实验：7 数据集 × 10 方法 × 2 骨干模型 |
| `ablation-full.md` | `experiments/exp2_ablation.py` | 消融：w/o Query Triage / PER / Memory |
| `hyperparameter-sensitivity.md` | `experiments/exp3_hyperparameter_sensitivity.py` | 超参数敏感性：5 参数扫描 |
| `intent-classification-accuracy.md` | `experiments/exp4_intent_classification.py` | 意图分类准确率：430 条标注集 |
| `retrieval-quality.md` | `experiments/exp5_retrieval_quality.py` | 检索质量与忠实性：Recall@k / 幻觉率 |
| `routing-counterfactual.md` | `experiments/exp6_routing_counterfactual.py` | 路由反事实：4 种路由策略 |
| `routing_distribution.md` | `experiments/exp7_routing_distribution.py` | 路由分布统计：各数据集意图标签分布 |
| `hop-depth-analysis.md` | `experiments/exp8_hop_depth.py` | 多跳深度：MuSiQue 2/3/4 跳衰减 |
| `memory-accumulation.md` | `experiments/exp9_memory_accumulation.py` | 记忆累积：冷启动 → 充分预热曲线 |
| `case-study.md` | — | 案例研究（定性分析，无需实验脚本） |

运行实验：

```bash
# 一键运行全部（500 题/数据集，对应论文结果）
bash experiments/run_all.sh --limit 500

# 烟雾测试
bash experiments/run_all.sh --limit 50

# 单独运行
uv run python experiments/exp2_ablation.py --datasets hotpotqa --limit 100
```

## 阅读顺序

1. **`architecture_overview.md`** — 先建立"注册表 + 插槽 + yaml 开关"这套骨架的心智模型，以及多Agent内部架构；
2. **`innovation-1/adagraph.md`** — 深入创新点一的三个子模块；
3. **`innovation-2/design.md`** — 创新点二的多Agent架构完整设计；
4. **`experiments/`** — 实验结果数据表；配合 `experiments/` 脚本目录复现。

## 注册表插槽速查

AdaGraph 占 3 个构建插槽，CollaRAG 占 2 个查询插槽：

| 插槽（slot） | 默认实现 | AdaGraph | CollaRAG |
|---|---|---|---|
| `chunker` | `defaults.fixed` | `adagraph.dynamic` | — |
| `extractor` | `defaults.simple_llm` | `adagraph.layered` | — |
| `pruner` | `defaults.noop` | `adagraph.dual_redundancy` | — |
| `intent_classifier` | `defaults.rule` | — | `colla_rag.tree`（树形分类：greeting / single_hop / multi_hop / summarization / other） |
| `retriever` | `defaults.direct` | — | `colla_rag.multi_agent`（MultiAgentRetriever：5 Agent + 31 工具 + 双层记忆） |
| `generator` | `defaults.prompt` | — | — |
