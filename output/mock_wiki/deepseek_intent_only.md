# 评测报告 · `deepseek_intent_only`

## 一、总览

| 指标 | 数值 |
|---|---|
| 样本数 | 50 |
| 总耗时 | 6 分 26.3 秒 |
| 吞吐 | 0.13 题/秒 |
| LLM 调用次数 | 150 |
| 输入 Token | 26,753 |
| 输出 Token | 14,270 |
| 合计 Token | 41,023 |
| 平均每题 Token | 820.5 |

### 综合指标

| 指标 | 数值 |
|---|---|
| 精确匹配 EM | 0.6600 |
| F1 | 0.7924 |
| ROUGE-L | 0.7924 |
| 平均延迟 (秒) | 7.72 |

## 二、按意图分组

| 意图 | 样本数 | 精确匹配 EM | F1 | ROUGE-L | 平均延迟(秒) | 平均 Token |
|---|---|---|---|---|---|---|
| 分析型 | 4 | 0.000 | 0.952 | 0.952 | 11.77 | 1046 |
| 比较型 | 5 | 0.000 | 0.372 | 0.372 | 9.81 | 979 |
| 事实型 | 30 | 0.967 | 0.998 | 0.998 | 6.84 | 710 |
| 多跳型 | 11 | 0.364 | 0.364 | 0.364 | 7.70 | 968 |

### 策略分布

| 策略 | 命中次数 |
|---|---|
| `direct` | 32 |
| `multi_hop` | 11 |
| `parallel` | 5 |
| `iterative` | 2 |

## 三、最差样本 Top-10（按 F1）

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| mock-005 | 多跳型 | multi_hop | Which company acquired the employer of Jonah Reed? | Winterfell Labs | I do not know. | 0.000 | 0.000 | 7.409 | 920 |
| mock-012 | 多跳型 | multi_hop | In which city is the company that Cara Solis works at headqu… | Barcelona | I do not know. | 0.000 | 0.000 | 7.421 | 943 |
| mock-013 | 多跳型 | multi_hop | In which city is the company that Dmitri Varga works at head… | Prague | I do not know. | 0.000 | 0.000 | 7.980 | 1047 |
| mock-015 | 多跳型 | multi_hop | Which company acquired the employer of Ada Verdant? | Helios Systems | I do not know. | 0.000 | 0.000 | 6.488 | 882 |
| mock-016 | 多跳型 | multi_hop | Which company acquired the employer of Farid Azar? | Riverbend Studios | I do not know. | 0.000 | 0.000 | 7.036 | 941 |
| mock-031 | 多跳型 | multi_hop | Which company acquired the employer of Elena Moss? | Obsidian AI | I do not know. | 0.000 | 0.000 | 8.109 | 1020 |
| mock-037 | 多跳型 | multi_hop | Which company acquired the employer of Noa Kessler? | Aether Graph | I do not know. | 0.000 | 0.000 | 6.634 | 879 |
| mock-045 | 比较型 | parallel | Compare the research fields of Dmitri Varga and Kira Lang. W… | robotics | Dmitri Varga researches robotics. Kira Lang researches graph… | 0.000 | 0.182 | 9.483 | 1023 |
| mock-004 | 比较型 | parallel | Compare the research fields of Cara Solis and Jonah Reed. Wh… | cryptography | Cara Solis researches cryptography. Jonah Reed researches ti… | 0.000 | 0.200 | 9.214 | 952 |
| mock-014 | 比较型 | parallel | Compare the research fields of Jonah Reed and Bruno Falk. Wh… | time-series forecasting | Jonah Reed researches time-series forecasting. Bruno Falk re… | 0.000 | 0.333 | 7.750 | 801 |

## 四、最好样本 Top-10（按 F1）

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| mock-049 | 事实型 | direct | In what year was Polaris Networks founded? | 2003 | 2003 | 1.000 | 1.000 | 6.620 | 689 |
| mock-048 | 事实型 | direct | Which company does Cara Solis work at? | Nimbus Labs | Nimbus Labs | 1.000 | 1.000 | 6.068 | 684 |
| mock-047 | 多跳型 | multi_hop | In which city is the company that Hiro Tanaka works at headq… | Osaka | Osaka | 1.000 | 1.000 | 7.330 | 926 |
| mock-046 | 多跳型 | multi_hop | In which city is the company that Iris Orlova works at headq… | Moscow | Moscow | 1.000 | 1.000 | 7.316 | 966 |
| mock-044 | 事实型 | direct | Where is Greta Halden based? | Oslo | Oslo | 1.000 | 1.000 | 7.615 | 738 |
| mock-043 | 事实型 | direct | In what year was Winterfell Labs founded? | 2015 | 2015 | 1.000 | 1.000 | 6.441 | 657 |
| mock-042 | 事实型 | direct | In what year was Vanta Analytics founded? | 2013 | 2013 | 1.000 | 1.000 | 5.691 | 660 |
| mock-041 | 事实型 | direct | What field does Hiro Tanaka research? | reinforcement learning | Reinforcement learning. | 1.000 | 1.000 | 6.861 | 714 |
| mock-040 | 事实型 | direct | In what year was Riverbend Studios founded? | 2007 | 2007 | 1.000 | 1.000 | 5.963 | 674 |
| mock-039 | 事实型 | direct | Where is Elena Moss based? | Stockholm | Stockholm | 1.000 | 1.000 | 6.601 | 786 |

## 五、全部 50 条样本

<details><summary>点击展开完整列表</summary>

| 编号 | 意图 | 策略 | 问题 | 参考答案 | 模型回答 | EM | F1 | 耗时(秒) | Token |
|---|---|---|---|---|---|---|---|---|---|
| mock-000 | 事实型 | direct | In what year was Yggdrasil AI founded? | 2019 | 2019 | 1.000 | 1.000 | 7.059 | 749 |
| mock-001 | 事实型 | direct | In what year was Xanadu Media founded? | 2017 | 2017 | 1.000 | 1.000 | 7.226 | 660 |
| mock-002 | 事实型 | direct | Tell me about Iris Orlova's role at Tundra Dynamics. | Iris Orlova leads research on optimization at Tundra Dynamics | Iris Orlova leads research on optimization at Tundra Dynamics. | 1.000 | 1.000 | 7.346 | 761 |
| mock-003 | 事实型 | direct | Where is Dmitri Varga based? | Prague | Prague | 1.000 | 1.000 | 6.221 | 684 |
| mock-004 | 比较型 | parallel | Compare the research fields of Cara Solis and Jonah Reed. What does Cara Solis research? | cryptography | Cara Solis researches cryptography. Jonah Reed researches time-series forecasting. | 0.000 | 0.200 | 9.214 | 952 |
| mock-005 | 多跳型 | multi_hop | Which company acquired the employer of Jonah Reed? | Winterfell Labs | I do not know. | 0.000 | 0.000 | 7.409 | 920 |
| mock-006 | 事实型 | direct | Which company does Ada Verdant work at? | Lumentech Corp | Lumentech Corp | 1.000 | 1.000 | 6.894 | 711 |
| mock-007 | 比较型 | parallel | Compare the research fields of Hiro Tanaka and Omar Diallo. What does Hiro Tanaka research? | reinforcement learning | Hiro Tanaka researches reinforcement learning. | 0.000 | 0.571 | 12.020 | 1081 |
| mock-008 | 事实型 | direct | Where is Farid Azar based? | Cairo | Cairo | 1.000 | 1.000 | 6.724 | 678 |
| mock-009 | 分析型 | iterative | Why is Farid Azar associated with natural language processing? | because Farid Azar leads research on natural language processing at Quillfire Corp | Farid Azar leads research on natural language processing at Quillfire Corp. | 0.000 | 0.957 | 9.130 | 969 |
| mock-010 | 分析型 | direct | Why is Ada Verdant associated with machine learning? | because Ada Verdant leads research on machine learning at Lumentech Corp | Ada Verdant leads research on machine learning at Lumentech Corp. | 0.000 | 0.952 | 14.209 | 1187 |
| mock-011 | 事实型 | direct | Which company does Dmitri Varga work at? | Obsidian AI | Obsidian AI | 1.000 | 1.000 | 5.997 | 715 |
| mock-012 | 多跳型 | multi_hop | In which city is the company that Cara Solis works at headquartered? | Barcelona | I do not know. | 0.000 | 0.000 | 7.421 | 943 |
| mock-013 | 多跳型 | multi_hop | In which city is the company that Dmitri Varga works at headquartered? | Prague | I do not know. | 0.000 | 0.000 | 7.980 | 1047 |
| mock-014 | 比较型 | parallel | Compare the research fields of Jonah Reed and Bruno Falk. What does Jonah Reed research? | time-series forecasting | Jonah Reed researches time-series forecasting. Bruno Falk researches quantum computing. | 0.000 | 0.333 | 7.750 | 801 |
| mock-015 | 多跳型 | multi_hop | Which company acquired the employer of Ada Verdant? | Helios Systems | I do not know. | 0.000 | 0.000 | 6.488 | 882 |
| mock-016 | 多跳型 | multi_hop | Which company acquired the employer of Farid Azar? | Riverbend Studios | I do not know. | 0.000 | 0.000 | 7.036 | 941 |
| mock-017 | 事实型 | direct | Tell me about Jonah Reed's role at Vanta Analytics. | Jonah Reed leads research on time-series forecasting at Vanta Analytics | Jonah Reed leads research on time-series forecasting at Vanta Analytics. | 1.000 | 1.000 | 7.958 | 779 |
| mock-018 | 分析型 | direct | Why is Dmitri Varga associated with robotics? | because Dmitri Varga leads research on robotics at Obsidian AI | Dmitri Varga leads research on robotics at Obsidian AI. | 0.000 | 0.947 | 10.787 | 996 |
| mock-019 | 事实型 | direct | In what year was Sendai Robotics founded? | 2009 | 2009 | 1.000 | 1.000 | 6.651 | 668 |
| mock-020 | 事实型 | direct | In what year was Lumentech Corp founded? | 1995 | 1995 | 1.000 | 1.000 | 7.287 | 624 |
| mock-021 | 事实型 | direct | Where is Hiro Tanaka based? | Osaka | Osaka | 1.000 | 1.000 | 6.700 | 692 |
| mock-022 | 事实型 | direct | In what year was Nimbus Labs founded? | 1999 | 1999 | 1.000 | 1.000 | 7.253 | 645 |
| mock-023 | 事实型 | direct | What is the headquarters city of Obsidian AI? | Prague | Prague | 1.000 | 1.000 | 5.965 | 694 |
| mock-024 | 事实型 | direct | Why is Hiro Tanaka associated with reinforcement learning? | because Hiro Tanaka leads research on reinforcement learning at Sendai Robotics | Hiro Tanaka leads research on reinforcement learning at Sendai Robotics. | 0.000 | 0.952 | 7.230 | 740 |
| mock-025 | 多跳型 | multi_hop | Which company acquired the employer of Hiro Tanaka? | Tundra Dynamics | Tundra Dynamics | 1.000 | 1.000 | 9.005 | 997 |
| mock-026 | 事实型 | direct | Tell me about Ada Verdant's role at Lumentech Corp. | Ada Verdant leads research on machine learning at Lumentech Corp | Ada Verdant leads research on machine learning at Lumentech Corp. | 1.000 | 1.000 | 7.524 | 814 |
| mock-027 | 事实型 | direct | What field does Iris Orlova research? | optimization | Optimization. | 1.000 | 1.000 | 5.780 | 757 |
| mock-028 | 事实型 | direct | Tell me about Elena Moss's role at Polaris Networks. | Elena Moss leads research on distributed systems at Polaris Networks | Elena Moss leads research on distributed systems at Polaris Networks. | 1.000 | 1.000 | 6.693 | 766 |
| mock-029 | 事实型 | direct | In what year was Obsidian AI founded? | 2001 | 2001 | 1.000 | 1.000 | 6.217 | 634 |
| mock-030 | 事实型 | direct | What field does Cara Solis research? | cryptography | Cryptography. | 1.000 | 1.000 | 6.746 | 746 |
| mock-031 | 多跳型 | multi_hop | Which company acquired the employer of Elena Moss? | Obsidian AI | I do not know. | 0.000 | 0.000 | 8.109 | 1020 |
| mock-032 | 多跳型 | multi_hop | In which city is the company that Greta Halden works at headquartered? | Oslo | Oslo | 1.000 | 1.000 | 10.024 | 1127 |
| mock-033 | 比较型 | parallel | Compare the research fields of Ada Verdant and Hiro Tanaka. What does Ada Verdant research? | machine learning | Ada Verdant researches machine learning. | 0.000 | 0.571 | 10.584 | 1036 |
| mock-034 | 分析型 | iterative | Why is Bruno Falk associated with quantum computing? | because Bruno Falk leads research on quantum computing at Helios Systems | Bruno Falk leads research on quantum computing at Helios Systems. | 0.000 | 0.952 | 12.941 | 1031 |
| mock-035 | 事实型 | direct | In what year was Aether Graph founded? | 2023 | 2023 | 1.000 | 1.000 | 9.886 | 670 |
| mock-036 | 事实型 | direct | What is the headquarters city of Quillfire Corp? | Cairo | Cairo | 1.000 | 1.000 | 6.151 | 693 |
| mock-037 | 多跳型 | multi_hop | Which company acquired the employer of Noa Kessler? | Aether Graph | I do not know. | 0.000 | 0.000 | 6.634 | 879 |
| mock-038 | 事实型 | direct | Tell me about Cara Solis's role at Nimbus Labs. | Cara Solis leads research on cryptography at Nimbus Labs | Cara Solis leads research on cryptography at Nimbus Labs. | 1.000 | 1.000 | 7.893 | 817 |
| mock-039 | 事实型 | direct | Where is Elena Moss based? | Stockholm | Stockholm | 1.000 | 1.000 | 6.601 | 786 |
| mock-040 | 事实型 | direct | In what year was Riverbend Studios founded? | 2007 | 2007 | 1.000 | 1.000 | 5.963 | 674 |
| mock-041 | 事实型 | direct | What field does Hiro Tanaka research? | reinforcement learning | Reinforcement learning. | 1.000 | 1.000 | 6.861 | 714 |
| mock-042 | 事实型 | direct | In what year was Vanta Analytics founded? | 2013 | 2013 | 1.000 | 1.000 | 5.691 | 660 |
| mock-043 | 事实型 | direct | In what year was Winterfell Labs founded? | 2015 | 2015 | 1.000 | 1.000 | 6.441 | 657 |
| mock-044 | 事实型 | direct | Where is Greta Halden based? | Oslo | Oslo | 1.000 | 1.000 | 7.615 | 738 |
| mock-045 | 比较型 | parallel | Compare the research fields of Dmitri Varga and Kira Lang. What does Dmitri Varga research? | robotics | Dmitri Varga researches robotics. Kira Lang researches graph neural networks. | 0.000 | 0.182 | 9.483 | 1023 |
| mock-046 | 多跳型 | multi_hop | In which city is the company that Iris Orlova works at headquartered? | Moscow | Moscow | 1.000 | 1.000 | 7.316 | 966 |
| mock-047 | 多跳型 | multi_hop | In which city is the company that Hiro Tanaka works at headquartered? | Osaka | Osaka | 1.000 | 1.000 | 7.330 | 926 |
| mock-048 | 事实型 | direct | Which company does Cara Solis work at? | Nimbus Labs | Nimbus Labs | 1.000 | 1.000 | 6.068 | 684 |
| mock-049 | 事实型 | direct | In what year was Polaris Networks founded? | 2003 | 2003 | 1.000 | 1.000 | 6.620 | 689 |

</details>
