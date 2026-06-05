# Qwen3:8b 实验结果

## Qwen3-8B Backbone，全数据集汇总

| Method Type | Method       | 2Wiki EM | 2Wiki F1 | HotpotQA EM | HotpotQA F1 | MuSiQue-Ans EM | MuSiQue-Ans F1 | MuSiQue-Full EM | MuSiQue-Full F1 |
|---|--------------|---|---|---|---|---|---|---|---|
| **LLM only** | Deepseek-v3  | 0.312 | 0.365 | 0.294 | 0.402 | 0.066 | 0.167 | 0.078 | 0.164 |
| **LLM only** | **Qwen3-8B** | **0.270** | **0.300** | **0.222** | **0.306** | **0.050** | **0.134** | 0.064 | 0.128 |
| **Text-based** | Vanilla RAG  | 0.231 | 0.283 | 0.341 | 0.428 | 0.113 | 0.156 | 0.058 | 0.105 |
| **Text-based** | RAPTOR       | 0.428 | 0.464 | 0.382 | 0.547 | 0.205 | 0.318 | 0.171 | 0.215 |
| **Graph-based** | TOG          | 0.265 | 0.324 | 0.189 | 0.211 | 0.062 | 0.087 | 0.055 | 0.059 |
| **Graph-based** | G-Retriever  | 0.357 | 0.395 | 0.308 | 0.439 | 0.156 | 0.214 | 0.082 | 0.118 |
| **Hybrid** | LightRAG     | 0.301 | 0.319 | 0.342 | 0.412 | 0.121 | 0.142 | 0.078 | 0.146 |
| **Hybrid** | GraphRAG     | 0.314 | 0.327 | 0.371 | 0.498 | 0.169 | 0.295 | 0.137 | 0.158 |
| **Hybrid** | HippoRAG     | 0.535 | 0.638 | 0.451 | 0.554 | 0.241 | 0.305 | 0.162 | 0.258 |
| **Proposed** | **My**       | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** |

**Table 1' — Estimated performance under Qwen3-8B backbone (calibrated by measured LLM-only).**

- 本地实测（500 题，seed=42，temperature=0，2026-06-04 跑）

---

## 附：MuSiQue-Ans 与 MuSiQue-Full 的区别

MuSiQue（**Mu**lti-hop **Si**ngle-hop **Que**stions）由 Trivedi 等人在 *TACL 2022* 提出，专门解决 HotpotQA / 2Wiki 上"假多跳捷径"问题。它有两个评测子集：

| 维度 | **MuSiQue-Ans** | **MuSiQue-Full** |
|---|---|---|
| 全称 | MuSiQue-Answerable | MuSiQue-Full |
| 题量（dev） | ~2,417 | ~2,417 + 不可答题（共 ~4,834） |
| 题目类型 | **全部可答** | 可答题 + **故意构造的不可答题**（unanswerable） |
| 不可答题怎么造 | — | 把可答题里"关键 supporting paragraph"替换掉，使推理链断裂 |
| 评测指标 | 直接算 EM / F1 | 模型必须**同时**回答可答题、并对不可答题输出"insufficient"。EM/F1 只在可答题上算；不可答题用"是否正确拒答"打分（Support F1 / Answer F1 联合） |
| 难度 | 高（强多跳） | **极高**（强多跳 + 抗"猜答"） |
| 论文里数值典型差 | EM 通常 0.25~0.35 | EM 通常 0.15~0.20（同方法低 30%~40%） |

**直白理解**：
- **MuSiQue-Ans** = "纯多跳考试"——题目都能答，比的是推理能不能走通。
- **MuSiQue-Full** = "多跳 + 拒答能力测试"——里面混了"故意让你答不出来"的陷阱题，模型如果硬猜就会被扣分。Full 是 Ans 的严格超集。

**为什么 Relink 论文里两列分开报**：
- Ans 列衡量"推理链构造能力"；
- Full 列衡量"推理链构造 + 知道自己不知道"。
- RAG 系统在 Ans 上往往不错，但 Full 上掉得很惨——因为大部分系统**倾向于硬给一个答案**，不会主动说"证据不足"。这也是 Relink 论文里所有方法 MuSiQue-Full EM 都低于 MuSiQue-Ans 的原因（HippoRAG 0.254→0.190，Relink 0.304→0.252）。

**对 Chimera-RAG 实验的实操影响**：
- 如果只关注"推理能力"——只跑 MuSiQue-Ans。
- 如果想测 IntentAgent 的反思 hook 能不能"识别证据不足"——必须跑 MuSiQue-Full，这是它的舞台。
- 本仓库 `data/musique/qa.jsonl` 目前来自 `musique_ans_v1.0_dev.jsonl`，是 **Ans 子集**；若要测 Full，需额外下载 `musique_full_v1.0_dev.jsonl` 并独立配 dataset 入口。
