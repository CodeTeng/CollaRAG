# 评测报告 · `ollama_intent_only`

## 一、总览

### 推理阶段

| 指标 | 数值 |
|---|---|
| 样本数 | 10 |
| 总耗时 | 15.6 秒 |
| 吞吐 | 0.64 题/秒 |
| LLM 调用次数 | 30 |
| 输入 Token | 8,075 |
| 输出 Token | 426 |
| 合计 Token | 8,501 |
| 平均每题 Token | 850.1 |

### 构图阶段（一次性）

| 指标 | 数值 |
|---|---|
| 文档数 | 37 |
| Chunk 数 | 37 |
| 三元组 | 127 抽取 → 127 保留 |
| 构图耗时 | 32.4 秒 |
| LLM 调用次数 | 37 |
| 合计 Token | 7,126 |

### 总计（构图 + 推理）

| 指标 | 构图 | 推理 | 合计 |
|---|---|---|---|
| 耗时 | 32.4 秒 | 15.6 秒 | 48.0 秒 |
| LLM 调用 | 37 | 30 | 67 |
| Token | 7,126 | 8,501 | 15,627 |

### 综合指标

| 指标 | 数值 |
|---|---|
| 精确匹配 EM | 0.8000 |
| F1 | 0.9015 |
| ROUGE-L | 0.9015 |
| 平均延迟 (秒) | 1.56 |

## 二、耗时分布（秒）

| 统计量 | 数值 |
|---|---|
| 平均 | 1.56 |
| 最小 | 1.34 |
| 最大 | 2.04 |
| P50  | 1.50 |
| P95  | 1.92 |
| 合计 | 15.6 |

## 三、Token 分布

| 字段 | 合计 | 平均 | 最小 | 最大 | P50 | P95 |
|---|---|---|---|---|---|---|
| 输入 Token | 8,075 | 807.5 | 656 | 1351 | 735 | 1147 |
| 输出 Token | 426 | 42.6 | 37 | 49 | 42 | 48 |
| 合计 Token | 8,501 | 850.1 | 693 | 1397 | 778 | 1194 |

## 四、按意图分组

| 意图 | 样本数 | 精确匹配 EM | F1 | ROUGE-L | 平均延迟(秒) | 平均 Token |
|---|---|---|---|---|---|---|
| 比较型 | 2 | 1.000 | 1.000 | 1.000 | 1.72 | 928 |
| 事实型 | 7 | 0.714 | 0.859 | 0.859 | 1.44 | 750 |
| 多跳型 | 1 | 1.000 | 1.000 | 1.000 | 2.04 | 1397 |

### 策略分布

| 策略 | 命中次数 |
|---|---|
| `direct` | 7 |
| `parallel` | 2 |
| `multi_hop` | 1 |

## 五、最差样本 Top-10（按 F1）

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| mock-009 | 事实型 | direct | Why is Farid Azar associated with natural language processin… | because Farid Azar leads research on natural language proces… | natural language processing | 0.000 | 0.400 | 1.520 | 790 |
| mock-002 | 事实型 | direct | Tell me about Iris Orlova's role at Tundra Dynamics. | Iris Orlova leads research on optimization at Tundra Dynamic… | leads research on optimization | 0.000 | 0.615 | 1.518 | 747 |
| mock-000 | 事实型 | direct | In what year was Yggdrasil AI founded? | 2019 | 2019 | 1.000 | 1.000 | 1.468 | 713 |
| mock-001 | 事实型 | direct | In what year was Xanadu Media founded? | 2017 | 2017 | 1.000 | 1.000 | 1.357 | 765 |
| mock-003 | 事实型 | direct | Where is Dmitri Varga based? | Prague | Prague | 1.000 | 1.000 | 1.484 | 793 |
| mock-004 | 比较型 | parallel | Compare the research fields of Cara Solis and Jonah Reed. Wh… | cryptography | cryptography | 1.000 | 1.000 | 1.769 | 945 |
| mock-005 | 多跳型 | multi_hop | Which company acquired the employer of Jonah Reed? | Winterfell Labs | Winterfell Labs | 1.000 | 1.000 | 2.045 | 1397 |
| mock-006 | 事实型 | direct | Which company does Ada Verdant work at? | Lumentech Corp | Lumentech Corp | 1.000 | 1.000 | 1.383 | 746 |
| mock-007 | 比较型 | parallel | Compare the research fields of Hiro Tanaka and Omar Diallo. … | reinforcement learning | reinforcement learning | 1.000 | 1.000 | 1.675 | 912 |
| mock-008 | 事实型 | direct | Where is Farid Azar based? | Cairo | Cairo | 1.000 | 1.000 | 1.340 | 693 |

## 六、最好样本 Top-10（按 F1）

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| mock-008 | 事实型 | direct | Where is Farid Azar based? | Cairo | Cairo | 1.000 | 1.000 | 1.340 | 693 |
| mock-007 | 比较型 | parallel | Compare the research fields of Hiro Tanaka and Omar Diallo. … | reinforcement learning | reinforcement learning | 1.000 | 1.000 | 1.675 | 912 |
| mock-006 | 事实型 | direct | Which company does Ada Verdant work at? | Lumentech Corp | Lumentech Corp | 1.000 | 1.000 | 1.383 | 746 |
| mock-005 | 多跳型 | multi_hop | Which company acquired the employer of Jonah Reed? | Winterfell Labs | Winterfell Labs | 1.000 | 1.000 | 2.045 | 1397 |
| mock-004 | 比较型 | parallel | Compare the research fields of Cara Solis and Jonah Reed. Wh… | cryptography | cryptography | 1.000 | 1.000 | 1.769 | 945 |
| mock-003 | 事实型 | direct | Where is Dmitri Varga based? | Prague | Prague | 1.000 | 1.000 | 1.484 | 793 |
| mock-001 | 事实型 | direct | In what year was Xanadu Media founded? | 2017 | 2017 | 1.000 | 1.000 | 1.357 | 765 |
| mock-000 | 事实型 | direct | In what year was Yggdrasil AI founded? | 2019 | 2019 | 1.000 | 1.000 | 1.468 | 713 |
| mock-002 | 事实型 | direct | Tell me about Iris Orlova's role at Tundra Dynamics. | Iris Orlova leads research on optimization at Tundra Dynamic… | leads research on optimization | 0.000 | 0.615 | 1.518 | 747 |
| mock-009 | 事实型 | direct | Why is Farid Azar associated with natural language processin… | because Farid Azar leads research on natural language proces… | natural language processing | 0.000 | 0.400 | 1.520 | 790 |

## 七、全部 10 条样本

<details><summary>点击展开完整列表</summary>

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| mock-000 | 事实型 | direct | In what year was Yggdrasil AI founded? | 2019 | 2019 | 1.000 | 1.000 | 1.468 | 713 |
| mock-001 | 事实型 | direct | In what year was Xanadu Media founded? | 2017 | 2017 | 1.000 | 1.000 | 1.357 | 765 |
| mock-002 | 事实型 | direct | Tell me about Iris Orlova's role at Tundra Dynamics. | Iris Orlova leads research on optimization at Tundra Dynamics | leads research on optimization | 0.000 | 0.615 | 1.518 | 747 |
| mock-003 | 事实型 | direct | Where is Dmitri Varga based? | Prague | Prague | 1.000 | 1.000 | 1.484 | 793 |
| mock-004 | 比较型 | parallel | Compare the research fields of Cara Solis and Jonah Reed. What does Cara Solis research? | cryptography | cryptography | 1.000 | 1.000 | 1.769 | 945 |
| mock-005 | 多跳型 | multi_hop | Which company acquired the employer of Jonah Reed? | Winterfell Labs | Winterfell Labs | 1.000 | 1.000 | 2.045 | 1397 |
| mock-006 | 事实型 | direct | Which company does Ada Verdant work at? | Lumentech Corp | Lumentech Corp | 1.000 | 1.000 | 1.383 | 746 |
| mock-007 | 比较型 | parallel | Compare the research fields of Hiro Tanaka and Omar Diallo. What does Hiro Tanaka research? | reinforcement learning | reinforcement learning | 1.000 | 1.000 | 1.675 | 912 |
| mock-008 | 事实型 | direct | Where is Farid Azar based? | Cairo | Cairo | 1.000 | 1.000 | 1.340 | 693 |
| mock-009 | 事实型 | direct | Why is Farid Azar associated with natural language processing? | because Farid Azar leads research on natural language processing at Quillfire Corp | natural language processing | 0.000 | 0.400 | 1.520 | 790 |

</details>
