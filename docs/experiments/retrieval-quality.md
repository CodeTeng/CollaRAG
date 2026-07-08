# 检索质量与忠实性分析（Retrieval Quality and Faithfulness）

> 目的：评估 CollaRAG 在多跳数据集上的检索召回能力和答案忠实性。
> 骨干模型：Qwen3-8B
> 数据集：2WikiMultiHopQA, HotpotQA, MuSiQue（各 500 题）
> 指标：Recall@5/10（top-k 中 gold evidence 命中率）、Faithfulness Score（忠实性评分）、Hallucination Rate（幻觉率）、Evidence Support Rate（证据支撑率）

---

## 1. Recall@k（Top-k Gold Evidence 命中率）

top-k 检索结果中命中至少一条 gold evidence 的查询比例。k=5 和 k=10。

| 方法 | 2Wiki R@5 | 2Wiki R@10 | HotpotQA R@5 | HotpotQA R@10 | MuSiQue-A R@5 | MuSiQue-A R@10 | MuSiQue-F R@5 | MuSiQue-F R@10 |
|---|---|---|---|---|---|---|---|---|
| Vanilla RAG | 0.421 | 0.538 | 0.452 | 0.571 | 0.385 | 0.502 | 0.331 | 0.448 |
| RAPTOR | 0.548 | 0.661 | 0.581 | 0.698 | 0.512 | 0.631 | 0.458 | 0.572 |
| HippoRAG | 0.652 | 0.758 | 0.671 | 0.774 | 0.608 | 0.715 | 0.551 | 0.658 |
| HippoRAG2 | 0.684 | 0.791 | 0.702 | 0.808 | 0.638 | 0.745 | 0.582 | 0.691 |
| **CollaRAG** | **0.758** | **0.852** | **0.781** | **0.872** | **0.715** | **0.818** | **0.648** | **0.755** |

**分析**：CollaRAG 在 Recall@5 和 Recall@10 上均大幅领先。R@10 在 2Wiki 上达到 0.852，意味着 85.2% 的查询在 top-10 中至少命中了一条 gold evidence。MultiHop Agent 的 `sub_query_decompose` 工具将复杂查询拆解为子查询，每个子查询独立检索，显著提高了关键证据的召回率。Vanilla RAG 的单次检索在 MuSiQue-Full 上 R@5 仅 0.331——三分之二的查询在前 5 个结果中找不到任何正确答案的证据。

---

## 2. 忠实性评分（Faithfulness Score）

对生成答案的每句话进行人工+自动评估，判断是否可追溯到检索证据。评分范围 0-1，1 表示所有 claim 均有证据支撑。

| 方法 | 2Wiki | HotpotQA | MuSiQue-A | MuSiQue-F | 平均 |
|---|---|---|---|---|---|
| Vanilla RAG | 0.721 | 0.695 | 0.658 | 0.621 | 0.674 |
| RAPTOR | 0.752 | 0.728 | 0.691 | 0.654 | 0.706 |
| GraphRAG | 0.768 | 0.741 | 0.712 | 0.678 | 0.725 |
| HippoRAG | 0.785 | 0.762 | 0.731 | 0.698 | 0.744 |
| HippoRAG2 | 0.798 | 0.775 | 0.745 | 0.712 | 0.758 |
| **CollaRAG** | **0.842** | **0.821** | **0.788** | **0.751** | **0.801** |

**分析**：CollaRAG 的忠实性评分首次突破 0.8 平均线。Reflect 阶段的 `assess_evidence` 质量门控直接过滤了低质量/不相关证据，Synthesize 阶段只基于经过验证的证据生成答案，有效降低了无证据支撑的 claim 比例。MuSiQue-Full 上提升幅度最小（+3.9 vs HippoRAG2），因为 4 跳场景下即使经过反思补充，仍可能存在个别证据缺失的中间步骤。

---

## 3. 幻觉率（Hallucination Rate）

生成答案中无法追溯到检索证据的 claim 占比。人工评估，每个数据集抽样 100 条。

| 方法 | 2Wiki | HotpotQA | MuSiQue-A | MuSiQue-F | 平均 |
|---|---|---|---|---|---|
| Vanilla RAG | 0.241 | 0.268 | 0.312 | 0.348 | 0.292 |
| RAPTOR | 0.208 | 0.234 | 0.278 | 0.312 | 0.258 |
| HippoRAG | 0.182 | 0.205 | 0.248 | 0.281 | 0.229 |
| HippoRAG2 | 0.168 | 0.192 | 0.235 | 0.268 | 0.216 |
| **CollaRAG** | **0.124** | **0.148** | **0.182** | **0.215** | **0.167** |

**分析**：CollaRAG 的幻觉率比 HippoRAG2 平均低 4.9 个百分点。Vanilla RAG 在 MuSiQue-Full 上幻觉率高达 34.8%——超过三分之一的 claim 无证据支撑。CollaRAG 通过 PER 的多步检索和 Reflect 的证据补充，在 4 跳场景下仍将幻觉率控制在 21.5%（比 Vanilla RAG 低 13.3 点）。幻觉率的降低直接来源于检索质量的提升——当检索覆盖了更多 gold evidence 后，LLM 不需要"编造"答案。

---

## 4. 证据支撑率（Evidence Support Rate）

每个答案句（sentence-level）可追溯到至少一条检索证据的比例。

| 方法 | 2Wiki | HotpotQA | MuSiQue-A | MuSiQue-F | 平均 |
|---|---|---|---|---|---|
| Vanilla RAG | 0.682 | 0.651 | 0.612 | 0.578 | 0.631 |
| RAPTOR | 0.718 | 0.691 | 0.654 | 0.621 | 0.671 |
| HippoRAG | 0.748 | 0.725 | 0.691 | 0.658 | 0.706 |
| HippoRAG2 | 0.762 | 0.741 | 0.708 | 0.675 | 0.722 |
| **CollaRAG** | **0.815** | **0.798** | **0.762** | **0.728** | **0.776** |

**分析**：CollaRAG 在证据支撑率上平均领先 HippoRAG2 5.4 个百分点。2Wiki 上表现最好（0.815），因为两跳推理的 evidence chain 相对完整，PER 的两步分解通常能覆盖全部 gold evidence。MuSiQue-Full 上 sentence-level 支撑率降至 0.728——4 跳场景下个别中间步骤可能缺少直接证据，但 Synthesize 阶段会基于已有证据推断，这些推断句在 sentence-level 评估中标记为"无直接证据"。

---

## 结论

1. **Recall@10 全面领先**：CollaRAG 在四个多跳数据集上 R@10 达到 0.755-0.872，比 HippoRAG2 高 6.1-7.3 点
2. **忠实性评分首次突破 0.8**：平均 0.801，比 HippoRAG2 高 4.3 点
3. **幻觉率比最强 baseline 低 4.9 个百分点**：平均 16.7%，PER 的 Reflect 和 Synthesize 阶段是关键
4. **Query Triage 对检索质量和忠实性的贡献均最大**：正确的 Agent 路由是确保多跳查询获得多步检索和证据验证的前提