# Relink 论文实验结果梳理

> 论文：**Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG**
> 作者：Manzong Huang, Chenyang Bu, Yi He, Xingrui Zhuo, Xindong Wu（合肥工业大学 + W&M）
> 出处：arXiv:2601.07192v1（2026-01-12 提交，AAAI 2026 收录）
> 代码：https://github.com/DMiC-Lab-HFUT/Relink

---

## 1. 实验设置速览

| 项目 | 内容 |
|---|---|
| **LLM（所有 RAG 方法 backbone）** | `deepseek-v3-0324` |
| **LLM-only baseline 模型** | `deepseek-v3-0324`、`gpt-4o-2024-07-06` |
| **Embedding（消融对照用）** | OpenAI `text-embedding-3-small`（仅在 Ranker 替换实验中出现，主表为 Relink 自训 Ranker） |
| **数据集** | 2WikiMultiHopQA / HotpotQA / ConcurrentQA / MuSiQue-Ans / MuSiQue-Full |
| **评测指标** | EM（Exact Match）、F1 |
| **采样规模** | 每个数据集 **500 题**（test split 随机采样） |
| **答案后处理** | 所有模型输出统一经 LLM 抽取标准化答案串后再打分 |

> 关键澄清：Relink **没有跨多个 LLM 做主实验**——所有 RAG 方法（Relink 与 ToG / G-Retriever / LightRAG / GraphRAG / HippoRAG）都用同一个 `deepseek-v3-0324`，确保横向公平。GPT-4o 仅作为 LLM-only baseline 出现一行。

---

## 2. 主结果表（Table 1，完整复刻）

> Backbone LLM：`deepseek-v3-0324`（除了 LLM-only 行另写 LLM 名）
> 评测：500 题采样，EM / F1
> 最佳加粗，第二好下划线（按论文原标注）

| 类型 | 方法 | 2Wiki EM | 2Wiki F1 | HotpotQA EM | HotpotQA F1 | ConcurrentQA EM | ConcurrentQA F1 | MuSiQue-Ans EM | MuSiQue-Ans F1 | MuSiQue-Full EM | MuSiQue-Full F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LLM only** | Deepseek-v3 | 0.312 | 0.365 | 0.294 | 0.402 | 0.114 | 0.170 | 0.066 | 0.167 | 0.078 | 0.164 |
| **LLM only** | GPT-4o | 0.292 | 0.358 | 0.330 | 0.424 | 0.086 | 0.158 | 0.106 | 0.213 | 0.106 | 0.214 |
| **Text-based** | Vanilla RAG | 0.264 | 0.295 | 0.376 | 0.494 | 0.310 | 0.389 | 0.118 | 0.178 | 0.080 | 0.145 |
| **Text-based** | RAPTOR | 0.467 | 0.548 | 0.472 | 0.631 | 0.400 | 0.476 | 0.258 | 0.371 | 0.188 | 0.286 |
| **Graph-based** | ToG | 0.328 | 0.373 | 0.241 | 0.292 | 0.166 | 0.185 | 0.095 | 0.127 | 0.053 | 0.070 |
| **Graph-based** | G-Retriever | 0.362 | 0.455 | 0.369 | 0.493 | 0.162 | 0.209 | 0.166 | 0.271 | 0.092 | 0.169 |
| **Hybrid** | LightRAG | 0.300 | 0.364 | 0.384 | 0.498 | 0.308 | 0.389 | 0.124 | 0.183 | 0.104 | 0.171 |
| **Hybrid** | GraphRAG | 0.318 | 0.379 | 0.450 | 0.569 | 0.398 | 0.475 | 0.203 | 0.358 | 0.138 | 0.206 |
| **Hybrid** | HippoRAG | <u>0.578</u> | <u>0.684</u> | <u>0.498</u> | <u>0.647</u> | <u>0.458</u> | <u>0.536</u> | <u>0.254</u> | <u>0.381</u> | <u>0.190</u> | <u>0.298</u> |
| **Proposed** | **Relink** | **0.628** | **0.722** | **0.558** | **0.704** | **0.505** | **0.596** | **0.304** | **0.413** | **0.252** | **0.370** |

---

## 3. 与 Chimera-RAG 实验对齐的子表（HotpotQA / 2Wiki / MuSiQue）

> 把 MuSiQue-Ans / MuSiQue-Full 都列出（论文未给"合并 MuSiQue"数字），实际复现时取 `MuSiQue-Ans` 与本仓库 `data/musique/` 对齐。

### 3.1 LLM-only baseline（无检索）

| LLM | HotpotQA EM/F1 | 2Wiki EM/F1 | MuSiQue-Ans EM/F1 | MuSiQue-Full EM/F1 |
|---|---|---|---|---|
| **Deepseek-v3-0324** | 0.294 / 0.402 | 0.312 / 0.365 | 0.066 / 0.167 | 0.078 / 0.164 |
| **GPT-4o-2024-07-06** | 0.330 / 0.424 | 0.292 / 0.358 | 0.106 / 0.213 | 0.106 / 0.214 |

### 3.2 各 RAG 方法（统一用 Deepseek-v3-0324 作 backbone）

| 方法 | HotpotQA EM/F1 | 2Wiki EM/F1 | MuSiQue-Ans EM/F1 | MuSiQue-Full EM/F1 |
|---|---|---|---|---|
| Vanilla RAG | 0.376 / 0.494 | 0.264 / 0.295 | 0.118 / 0.178 | 0.080 / 0.145 |
| RAPTOR | 0.472 / 0.631 | 0.467 / 0.548 | 0.258 / 0.371 | 0.188 / 0.286 |
| ToG | 0.241 / 0.292 | 0.328 / 0.373 | 0.095 / 0.127 | 0.053 / 0.070 |
| G-Retriever | 0.369 / 0.493 | 0.362 / 0.455 | 0.166 / 0.271 | 0.092 / 0.169 |
| LightRAG | 0.384 / 0.498 | 0.300 / 0.364 | 0.124 / 0.183 | 0.104 / 0.171 |
| GraphRAG | 0.450 / 0.569 | 0.318 / 0.379 | 0.203 / 0.358 | 0.138 / 0.206 |
| HippoRAG | 0.498 / 0.647 | 0.578 / 0.684 | 0.254 / 0.381 | 0.190 / 0.298 |
| **Relink** | **0.558 / 0.704** | **0.628 / 0.722** | **0.304 / 0.413** | **0.252 / 0.370** |

### 3.3 Relink 相对最强 baseline（HippoRAG）的提升

| 数据集 | EM 提升（绝对 / 相对） | F1 提升（绝对 / 相对） |
|---|---|---|
| HotpotQA | +0.060 / **+12.0%** | +0.057 / +8.8% |
| 2Wiki | +0.050 / +8.7% | +0.038 / +5.6% |
| MuSiQue-Ans | +0.050 / +19.7% | +0.032 / +8.4% |
| MuSiQue-Full | +0.062 / **+32.6%** | +0.072 / +24.2% |

---

## 4. 消融实验（Table 2，仅在 2Wiki + HotpotQA）

> Backbone LLM 同样为 deepseek-v3-0324

| 配置 | 2Wiki EM | 2Wiki F1 | HotpotQA EM | HotpotQA F1 |
|---|---|---|---|---|
| **Relink (Full)** | **0.628** | **0.722** | **0.558** | **0.704** |
| w/o Explicit Graph (Gb) | 0.582 | 0.672 | 0.486 | 0.636 |
| w/o Dynamic Repair (Rc) | 0.616 | 0.714 | 0.526 | 0.680 |
| w/o Query-Driven Ranker | 0.552 | 0.649 | 0.450 | 0.600 |
| w/o L_contra | 0.603 | 0.695 | 0.518 | 0.675 |

> 关键发现：
> - **去 Query-Driven Ranker**（换成 OpenAI `text-embedding-3-small` 通用语义相似度）跌幅最大（HotpotQA EM 相对 −19.4%），说明针对 query 的"有用 vs 相关"判别比通用语义相似度更关键。
> - **去 Explicit Graph (Gb)** 在 HotpotQA EM 相对 −12.9%，骨架图缺失同样损失明显。
> - **去 Dynamic Repair (Rc)** 在 HotpotQA EM 相对 −5.7%，潜在关系池主要负责补全断裂链。

---

## 5. 对 Chimera-RAG（实验 6 横向对比）的可用性结论

1. **可直接横向引用的数字**：Relink 主表是目前 4 篇 baseline 论文里**唯一**在 HotpotQA / 2Wiki / MuSiQue 上同时给齐 EM/F1、且所有 baseline 都用同一 LLM（Deepseek-v3）的工作 → 对本项目最有参考价值。
2. **LLM 对齐**：本项目 `configs/deepseek_*.yaml` 用的也是 deepseek 系列，理论上能与 Relink 主表做"同 LLM 公平对比"；但 Relink 用 `deepseek-v3-0324` 具体版本，复现时需在 `.env` 里固定该版本号。
3. **采样规模警告**：Relink 每集只用 500 题随机采样，本项目 `eval.py` 跑全量时数字会与之有偏差，比较时需说明口径。
4. **指标口径一致**：Relink 用 EM + F1，与本仓库 `eval.py` 完全对齐，无需重打分（但仍建议本地跑一遍 Relink 开源代码以排除答案后处理差异）。
5. **当前 SOTA 排序**（Deepseek-v3 backbone，5 集 EM/F1 平均）：**Relink > HippoRAG > RAPTOR > GraphRAG > LightRAG / Vanilla RAG > G-Retriever > ToG > LLM-only**。HippoRAG 仍是非 Relink 阵营里最强的 baseline。
6. **关于 GPT-4o**：仅作为无检索 LLM 上限对照，**未与其他 RAG 方法在同 LLM 下对比** → 不能用 GPT-4o 行的数字横向比 RAG 方法。
