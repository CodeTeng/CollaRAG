# 评测报告 · `ollama_intent_only`

## 一、总览

### 推理阶段

| 指标 | 数值 |
|---|---|
| 样本数 | 3 |
| 总耗时 | 7.4 秒 |
| 吞吐 | 0.40 题/秒 |
| LLM 调用次数 | 9 |
| 输入 Token | 4,084 |
| 输出 Token | 138 |
| 合计 Token | 4,222 |
| 平均每题 Token | 1407.3 |

### 综合指标

| 指标 | 数值 |
|---|---|
| 精确匹配 EM | 0.6667 |
| F1 | 0.6667 |
| ROUGE-L | 0.6667 |
| 平均延迟 (秒) | 2.48 |

## 二、耗时分布（秒）

| 统计量 | 数值 |
|---|---|
| 平均 | 2.48 |
| 最小 | 1.93 |
| 最大 | 3.13 |
| P50  | 2.37 |
| P95  | 3.05 |
| 合计 | 7.4 |

## 三、Token 分布

| 字段 | 合计 | 平均 | 最小 | 最大 | P50 | P95 |
|---|---|---|---|---|---|---|
| 输入 Token | 4,084 | 1361.3 | 1089 | 1610 | 1385 | 1588 |
| 输出 Token | 138 | 46.0 | 46 | 46 | 46 | 46 |
| 合计 Token | 4,222 | 1407.3 | 1135 | 1656 | 1431 | 1634 |

## 四、按意图分组

| 意图 | 样本数 | 精确匹配 EM | F1 | ROUGE-L | 平均延迟(秒) | 平均 Token |
|---|---|---|---|---|---|---|
| 比较型 | 1 | 0.000 | 0.000 | 0.000 | 3.13 | 1431 |
| 事实型 | 2 | 1.000 | 1.000 | 1.000 | 2.15 | 1396 |

### 策略分布

| 策略 | 命中次数 |
|---|---|
| `direct` | 2 |
| `parallel` | 1 |

## 五、最差样本 Top-10（按 F1）

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| 5a8b57f25542995d1e6f1371 | 比较型 | parallel | Were Scott Derrickson and Ed Wood of the same nationality? | yes | American | 0.000 | 0.000 | 3.130 | 1431 |
| 5a8c7595554299585d9e36b6 | 事实型 | direct | What government position was held by the woman who portrayed… | Chief of Protocol | Chief of Protocol | 1.000 | 1.000 | 1.932 | 1135 |
| 5a85ea095542994775f606a8 | 事实型 | direct | What science fantasy young adult series, told in first perso… | Animorphs | Animorphs | 1.000 | 1.000 | 2.367 | 1656 |

## 六、最好样本 Top-10（按 F1）

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| 5a85ea095542994775f606a8 | 事实型 | direct | What science fantasy young adult series, told in first perso… | Animorphs | Animorphs | 1.000 | 1.000 | 2.367 | 1656 |
| 5a8c7595554299585d9e36b6 | 事实型 | direct | What government position was held by the woman who portrayed… | Chief of Protocol | Chief of Protocol | 1.000 | 1.000 | 1.932 | 1135 |
| 5a8b57f25542995d1e6f1371 | 比较型 | parallel | Were Scott Derrickson and Ed Wood of the same nationality? | yes | American | 0.000 | 0.000 | 3.130 | 1431 |

## 七、全部 3 条样本

<details><summary>点击展开完整列表</summary>

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| 5a8b57f25542995d1e6f1371 | 比较型 | parallel | Were Scott Derrickson and Ed Wood of the same nationality? | yes | American | 0.000 | 0.000 | 3.130 | 1431 |
| 5a8c7595554299585d9e36b6 | 事实型 | direct | What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell? | Chief of Protocol | Chief of Protocol | 1.000 | 1.000 | 1.932 | 1135 |
| 5a85ea095542994775f606a8 | 事实型 | direct | What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species? | Animorphs | Animorphs | 1.000 | 1.000 | 2.367 | 1656 |

</details>
