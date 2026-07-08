# 路由反事实实验（Routing Counterfactual）

> 目的：验证性能提升确实来自"路由决策正确"，而非仅仅因为某个 Agent（如 MultiHop）本身更强。
> 骨干模型：Qwen3-8B

---

## 实验设置

四种路由策略对比：

| 策略 | 说明 |
|---|---|
| **Normal routing** | TreeIntentClassifier 正常分类 → 分派到对应 Agent（默认配置） |
| **All → SingleHop** | 所有查询强制路由到 SingleHop Agent |
| **All → MultiHop** | 所有查询强制路由到 MultiHop Agent（Plan-Execute-Reflect） |
| **Random routing** | 每个查询随机分配到 {SingleHop, MultiHop, Summarization, Other} 之一 |

评测集：从 7 个数据集各抽样 100 题，共 700 题混合查询集。

---

## 主表：四种路由策略对比（N=700, Qwen3-8B）

| 路由策略 | EM | F1 |
|---|---|---|
| Normal routing | **0.427** | **0.558** |
| All → SingleHop | 0.349 | 0.447 |
| All → MultiHop | 0.367 | 0.503 |
| Random routing | 0.315 | 0.420 |

---

## 分数据集类型细分（EM）

| 路由策略 | 单跳 (NQ+PopQA, n=200) | 多跳 (2Wiki+Hotpot+MuSiQue, n=400) | 总结 (ASQA, n=100) |
|---|---|---|---|
| Normal routing | **0.400** | **0.463** | **0.338** |
| All → SingleHop | 0.377 | 0.380 | 0.168 |
| All → MultiHop | 0.264 | 0.457 | 0.215 |
| Random routing | 0.314 | 0.344 | 0.198 |

---