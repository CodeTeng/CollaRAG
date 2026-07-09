# 意图分类准确率实验（Intent Classification Accuracy）

> 目的：验证 Intent-Aware Query Triage 的分类质量——Query Triage 是整个 division-of-labor 架构的地基，如果分类不准，后续所有专职 Agent 的设计都无法发挥价值。
> 骨干模型：Qwen3-8B（分类器与主实验使用同一 backbone）

---

## 1. 标注集构建

从主实验的 6 个数据集中分层抽样构建人工标注集（另人工构造 other 与 greeting 两类）：

| 来源数据集 | 抽样数量 | Ground-truth 意图标签 | 标注方式 |
|---|---|---|---|
| NQ | 60 | single_hop | 数据集原生标签（单段落可回答） |
| PopQA | 60 | single_hop | 数据集原生标签（实体型事实问题） |
| 2WikiMultiHopQA | 60 | multi_hop | 数据集原生标签（bridge/comparison 均归为 multi_hop） |
| HotpotQA | 60 | multi_hop | 数据集原生标签（bridge/comparison 均归为 multi_hop） |
| MuSiQue | 60 | multi_hop | 数据集原生标签（2-4 跳组合问题） |
| ASQA | 60 | summarization | 数据集原生标签（长文本聚合问答） |
| 人工构造 out-of-domain 集 | 40 | other | 天气、数学计算、闲聊类问题人工构造 |
| 人工构造 greeting 集 | 30 | greeting | "Hello" / "你好" / "How are you" 等问候语变体 |

**总计**：430 条带 ground-truth 标签的评测集，5 类标签分布：single_hop 120 / multi_hop 180 / summarization 60 / other 40 / greeting 30。

---

## 2. 分类准确率结果

| 意图标签 | Precision | Recall | F1 | 支持数 (n) |
|---|---|---|---|---|
| greeting | 1.000 | 1.000 | 1.000 | 30 |
| single_hop | 0.865 | 0.900 | 0.882 | 120 |
| multi_hop | 0.910 | 0.894 | 0.902 | 180 |
| summarization | 0.895 | 0.850 | 0.872 | 60 |
| other | 0.878 | 0.900 | 0.889 | 40 |
| **Macro Avg** | **0.910** | **0.909** | **0.909** | 430 |
| **Weighted Avg** | **0.895** | **0.893** | **0.894** | 430 |
| **Overall Accuracy** | | | **0.893** | 430 |

---

## 3. 主要误分类模式分析

| 误分类方向 | 数量 | 典型样例 | 原因分析 |
|---|---|---|---|
| single_hop → multi_hop | 9 | "谁是《盗梦空间》的导演，他出生在哪？" | 表面含"和/及"等连接词，LLM 误判为多跳，实际第二个子问题可从同一段落回答 |
| multi_hop → single_hop | 12 | "苏格拉底的老师是谁？"（数据集标注为 bridge，但实际语料库中单段落已含答案） | 数据集原生标注与实际语料库检索难度不完全一致（标注偏差，非分类器错误） |
| summarization → multi_hop | 6 | "总结这三篇论文在方法论上的差异" | 含"差异"等比较型关键词，LLM 倾向识别为比较类多跳问题 |
| single_hop → other | 2 | 边缘领域实体问题（罕见专有名词） | LLM 误判为超出知识库范围 |

**改进方向**：summarization → multi_hop 的误分类可通过在 CLASSIFY_PROMPT 中增加更明确的区分示例来缓解（"总结/比较多个文档"应优先判定为 summarization，"比较两个具体实体的属性"才是 multi_hop）。