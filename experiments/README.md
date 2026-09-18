# Chimera-RAG 实验代码

> 本目录包含论文中所有实验的可复现脚本。每个实验对应 `docs/experiments/` 中的一份结果文档。

## 实验列表

| # | 实验名称 | 脚本 | 对应文档 | 说明 |
|---|---|---|---|---|
| 1 | 主实验 (Table 1 & 2) | `exp1_main_results.py` | `main-results-full.md` | 全数据集 × 全方法对比 |
| 2 | 消融实验 | `exp2_ablation.py` | `ablation-full.md` | w/o Query Triage / PER / Memory |
| 3 | 超参数敏感性 | `exp3_hyperparameter_sensitivity.py` | `hyperparameter-sensitivity.md` | 5 个关键超参数扫描 |
| 4 | 意图分类准确率 | `exp4_intent_classification.py` | `intent-classification-accuracy.md` | TreeIntentClassifier 430 条评测 |
| 5 | 检索质量与忠实性 | `exp5_retrieval_quality.py` | `retrieval-quality.md` | Recall@k / 忠实性 / 幻觉率 / 证据支撑率 |
| 6 | 路由反事实 | `exp6_routing_counterfactual.py` | `routing-counterfactual.md` | 4 种路由策略对比 |
| 7 | 路由分布统计 | `exp7_routing_distribution.py` | `routing_distribution.md` | 各数据集意图标签分布 |
| 8 | 多跳深度分析 | `exp8_hop_depth.py` | `hop-depth-analysis.md` | MuSiQue 按 2/3/4 跳拆分 |
| 9 | 记忆累积效应 | `exp9_memory_accumulation.py` | `memory-accumulation.md` | 冷启动 → 充分预热曲线 |

## 运行方式

### 前置条件

```bash
# 1. 安装依赖（agent 与 ragas 互斥，不要用 --all-extras）
uv sync --extra web --extra agent --extra llm --extra vec --dev

# 2. 配置环境变量（.env 文件）
#    LLM_API_KEY=<your-api-key>

# 3. 下载数据集
uv run python scripts/download_datasets.py
```

**可选数据集名**（`--datasets` 的 choices，见 `utils.py`）：
`nq` · `popqa` · `hotpotqa` · `two_wiki` · `musique` · `asqa`

> ⚠️ 注意是 `two_wiki` 而不是 `2wiki`。
> ⚠️ `asqa` 出现在 `DATASETS_ALL` / `exp1` 的 choices 里，但 `data/` 下**没有**对应目录，也没在 `scripts/download_datasets.py` 的下载列表内——用它之前需先补数据。

### 运行单个实验

```bash
# 例：运行主实验（NQ 数据集，CollaRAG 方法）
uv run python experiments/exp1_main_results.py \
    --datasets nq \
    --methods collarag \
    --backbone qwen3-8b \
    --limit 500

# 例：运行消融实验
uv run python experiments/exp2_ablation.py \
    --datasets hotpotqa two_wiki \
    --limit 500
```

### 运行全部实验

```bash
bash experiments/run_all.sh
```

## 输出结构

所有实验结果输出到 `output/experiments/` 目录：

```
output/experiments/
├── exp1_main_results/
│   ├── qwen3-8b/
│   │   ├── nq/
│   │   │   ├── collarag/predictions.jsonl
│   │   │   ├── vanilla_rag/predictions.jsonl
│   │   │   └── ...
│   │   └── compare/
│   │       ├── report.md
│   │       └── report.json
│   └── qwen3-32b/
├── exp2_ablation/
├── exp3_hyperparameter/
├── ...
└── summary.json          # 汇总所有实验指标
```

## 共享工具

- `utils.py`：通用工具函数（`make_config()` 动态生成临时配置、结果收集、Markdown 表格生成）
- `eval.py` / `eval_ragas.py`：离线打分（EM/F1/ROUGE-L 与 RAGAS 语义指标）

> 注：实验用的配置**不**以独立目录形式存在，而是由 `utils.py::make_config()` 在运行时
> 基于 `configs/*.yaml` 派生，落到 `output/experiments/<exp_name>/configs/`。
