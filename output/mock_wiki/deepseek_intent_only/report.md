# 评测报告 · `deepseek_intent_only`

## 一、总览

| 指标           | 数值         |
| -------------- | ------------ |
| 样本数         | 10           |
| 总耗时         | 1 分 16.6 秒 |
| 吞吐           | 0.13 题/秒   |
| LLM 调用次数   | 30           |
| 输入 Token     | 4,717        |
| 输出 Token     | 2,767        |
| 合计 Token     | 7,484        |
| 平均每题 Token | 748.4        |

### 综合指标

| 指标          | 数值   |
| ------------- | ------ |
| 精确匹配 EM   | 0.6000 |
| F1            | 0.7664 |
| ROUGE-L       | 0.7664 |
| 平均延迟 (秒) | 7.65   |

## 二、耗时分布（秒）

| 统计量 | 数值  |
| ------ | ----- |
| 平均   | 7.65  |
| 最小   | 5.86  |
| 最大   | 10.41 |
| P50    | 7.15  |
| P95    | 10.28 |
| 合计   | 76.5  |

## 三、Token 分布

| 字段       | 合计  | 平均  | 最小 | 最大 | P50 | P95 |
| ---------- | ----- | ----- | ---- | ---- | --- | --- |
| 输入 Token | 4,717 | 471.7 | 425  | 521  | 466 | 518 |
| 输出 Token | 2,767 | 276.7 | 187  | 481  | 241 | 449 |
| 合计 Token | 7,484 | 748.4 | 634  | 1002 | 714 | 946 |

## 四、按意图分组

| 意图   | 样本数 | 精确匹配 EM | F1    | ROUGE-L | 平均延迟(秒) | 平均 Token |
| ------ | ------ | ----------- | ----- | ------- | ------------ | ---------- |
| 分析型 | 1      | 0.000       | 0.957 | 0.957   | 10.41        | 877        |
| 比较型 | 2      | 0.000       | 0.354 | 0.354   | 8.93         | 907        |
| 事实型 | 6      | 1.000       | 1.000 | 1.000   | 6.88         | 676        |
| 多跳型 | 1      | 0.000       | 0.000 | 0.000   | 7.00         | 735        |

### 策略分布

| 策略        | 命中次数 |
| ----------- | -------- |
| `direct`    | 6        |
| `parallel`  | 2        |
| `multi_hop` | 1        |
| `iterative` | 1        |

## 五、最差样本 Top-10（按 F1）

| 编号     | 意图   | 策略      | 问题                                                          | 参考答案                                                      | 模型回答                                                      | EM    | F1    | 耗时(秒) | Token |
| -------- | ------ | --------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ----- | ----- | -------- | ----- |
| mock-005 | 多跳型 | multi_hop | Which company acquired the employer of Jonah Reed?            | Winterfell Labs                                               | I do not know.                                                | 0.000 | 0.000 | 6.996    | 735   |
| mock-007 | 比较型 | parallel  | Compare the research fields of Hiro Tanaka and Omar Diallo. … | reinforcement learning                                        | Hiro Tanaka researches reinforcement learning, while Omar Di… | 0.000 | 0.308 | 10.124   | 1002  |
| mock-004 | 比较型 | parallel  | Compare the research fields of Cara Solis and Jonah Reed. Wh… | cryptography                                                  | Cara Solis researches cryptography.                           | 0.000 | 0.400 | 7.740    | 812   |
| mock-009 | 分析型 | iterative | Why is Farid Azar associated with natural language processin… | because Farid Azar leads research on natural language proces… | Farid Azar leads research on natural language processing at … | 0.000 | 0.957 | 10.412   | 877   |
| mock-000 | 事实型 | direct    | In what year was Yggdrasil AI founded?                        | 2019                                                          | 2019                                                          | 1.000 | 1.000 | 6.556    | 669   |
| mock-001 | 事实型 | direct    | In what year was Xanadu Media founded?                        | 2017                                                          | 2017                                                          | 1.000 | 1.000 | 5.864    | 634   |
| mock-002 | 事实型 | direct    | Tell me about Iris Orlova's role at Tundra Dynamics.          | Iris Orlova leads research on optimization at Tundra Dynamic… | Iris Orlova leads research on optimization at Tundra Dynamic… | 1.000 | 1.000 | 8.911    | 738   |
| mock-003 | 事实型 | direct    | Where is Dmitri Varga based?                                  | Prague                                                        | Prague                                                        | 1.000 | 1.000 | 6.447    | 692   |
| mock-006 | 事实型 | direct    | Which company does Ada Verdant work at?                       | Lumentech Corp                                                | Lumentech Corp                                                | 1.000 | 1.000 | 6.169    | 652   |
| mock-008 | 事实型 | direct    | Where is Farid Azar based?                                    | Cairo                                                         | Cairo                                                         | 1.000 | 1.000 | 7.304    | 673   |

## 六、最好样本 Top-10（按 F1）

| 编号     | 意图   | 策略      | 问题                                                          | 参考答案                                                      | 模型回答                                                      | EM    | F1    | 耗时(秒) | Token |
| -------- | ------ | --------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ----- | ----- | -------- | ----- |
| mock-008 | 事实型 | direct    | Where is Farid Azar based?                                    | Cairo                                                         | Cairo                                                         | 1.000 | 1.000 | 7.304    | 673   |
| mock-006 | 事实型 | direct    | Which company does Ada Verdant work at?                       | Lumentech Corp                                                | Lumentech Corp                                                | 1.000 | 1.000 | 6.169    | 652   |
| mock-003 | 事实型 | direct    | Where is Dmitri Varga based?                                  | Prague                                                        | Prague                                                        | 1.000 | 1.000 | 6.447    | 692   |
| mock-002 | 事实型 | direct    | Tell me about Iris Orlova's role at Tundra Dynamics.          | Iris Orlova leads research on optimization at Tundra Dynamic… | Iris Orlova leads research on optimization at Tundra Dynamic… | 1.000 | 1.000 | 8.911    | 738   |
| mock-001 | 事实型 | direct    | In what year was Xanadu Media founded?                        | 2017                                                          | 2017                                                          | 1.000 | 1.000 | 5.864    | 634   |
| mock-000 | 事实型 | direct    | In what year was Yggdrasil AI founded?                        | 2019                                                          | 2019                                                          | 1.000 | 1.000 | 6.556    | 669   |
| mock-009 | 分析型 | iterative | Why is Farid Azar associated with natural language processin… | because Farid Azar leads research on natural language proces… | Farid Azar leads research on natural language processing at … | 0.000 | 0.957 | 10.412   | 877   |
| mock-004 | 比较型 | parallel  | Compare the research fields of Cara Solis and Jonah Reed. Wh… | cryptography                                                  | Cara Solis researches cryptography.                           | 0.000 | 0.400 | 7.740    | 812   |
| mock-007 | 比较型 | parallel  | Compare the research fields of Hiro Tanaka and Omar Diallo. … | reinforcement learning                                        | Hiro Tanaka researches reinforcement learning, while Omar Di… | 0.000 | 0.308 | 10.124   | 1002  |
| mock-005 | 多跳型 | multi_hop | Which company acquired the employer of Jonah Reed?            | Winterfell Labs                                               | I do not know.                                                | 0.000 | 0.000 | 6.996    | 735   |

## 七、全部 10 条样本

<details><summary>点击展开完整列表</summary>

| 编号     | 意图   | 策略      | 问题                                                                                        | 参考答案                                                                           | 模型回答                                                                                      | EM    | F1    | 耗时(秒) | Token |
| -------- | ------ | --------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----- | ----- | -------- | ----- |
| mock-000 | 事实型 | direct    | In what year was Yggdrasil AI founded?                                                      | 2019                                                                               | 2019                                                                                          | 1.000 | 1.000 | 6.556    | 669   |
| mock-001 | 事实型 | direct    | In what year was Xanadu Media founded?                                                      | 2017                                                                               | 2017                                                                                          | 1.000 | 1.000 | 5.864    | 634   |
| mock-002 | 事实型 | direct    | Tell me about Iris Orlova's role at Tundra Dynamics.                                        | Iris Orlova leads research on optimization at Tundra Dynamics                      | Iris Orlova leads research on optimization at Tundra Dynamics.                                | 1.000 | 1.000 | 8.911    | 738   |
| mock-003 | 事实型 | direct    | Where is Dmitri Varga based?                                                                | Prague                                                                             | Prague                                                                                        | 1.000 | 1.000 | 6.447    | 692   |
| mock-004 | 比较型 | parallel  | Compare the research fields of Cara Solis and Jonah Reed. What does Cara Solis research?    | cryptography                                                                       | Cara Solis researches cryptography.                                                           | 0.000 | 0.400 | 7.740    | 812   |
| mock-005 | 多跳型 | multi_hop | Which company acquired the employer of Jonah Reed?                                          | Winterfell Labs                                                                    | I do not know.                                                                                | 0.000 | 0.000 | 6.996    | 735   |
| mock-006 | 事实型 | direct    | Which company does Ada Verdant work at?                                                     | Lumentech Corp                                                                     | Lumentech Corp                                                                                | 1.000 | 1.000 | 6.169    | 652   |
| mock-007 | 比较型 | parallel  | Compare the research fields of Hiro Tanaka and Omar Diallo. What does Hiro Tanaka research? | reinforcement learning                                                             | Hiro Tanaka researches reinforcement learning, while Omar Diallo researches knowledge graphs. | 0.000 | 0.308 | 10.124   | 1002  |
| mock-008 | 事实型 | direct    | Where is Farid Azar based?                                                                  | Cairo                                                                              | Cairo                                                                                         | 1.000 | 1.000 | 7.304    | 673   |
| mock-009 | 分析型 | iterative | Why is Farid Azar associated with natural language processing?                              | because Farid Azar leads research on natural language processing at Quillfire Corp | Farid Azar leads research on natural language processing at Quillfire Corp.                   | 0.000 | 0.957 | 10.412   | 877   |

</details>
