# 消融实验

> 骨干模型：Qwen3-8B
> 评测设置：500 题/数据集，seed=42，temperature=0
> 指标：EM (Exact Match)
> 三个变体分别对应论文的三个创新点，移除方式详见各组件说明

---

## 消融实验主表（多跳 + 总结数据集，EM）

| 变体 | 移除的创新点 | 2Wiki | HotpotQA | MuSiQue-A | MuSiQue-F | ASQA |
|---|---|---|---|---|---|---|
| **CollaRAG (Full)** | — | **0.602** | **0.510** | **0.277** | **0.189** | **0.338** |
| w/o Query Triage | 创新点 1：Intent-Aware Query Triage | 0.485 | 0.409 | 0.205 | 0.130 | 0.216 |
| w/o PER | 创新点 2：Plan-Execute-Reflect | 0.559 | 0.467 | 0.247 | 0.170 | 0.301 |
| w/o Memory | 创新点 3：Dual-Layer Experiential Memory | 0.572 | 0.479 | 0.259 | 0.179 | 0.316 |

**变体说明：**

- **w/o Query Triage**：移除 TreeIntentClassifier，所有查询统一路由到 SingleHop Agent（单轮 hybrid search → generate）。用于量化"意图感知分诊"的贡献。
- **w/o PER**：MultiHop Agent 的 Plan-Execute-Reflect 退化为标准 ReAct 循环（单轮 Thought→Action→Observation，移除 Plan 和 Reflect 阶段）。SingleHop/Summarization/Other Agent 不受影响。用于量化"三阶段工作流"的贡献。
- **w/o Memory**：同时禁用 SharedMemory（QA cache、routing stats、entity aliases）和 AgentPrivateMemory。每次查询从零开始，无跨查询经验积累。用于量化"双层记忆"的贡献。
