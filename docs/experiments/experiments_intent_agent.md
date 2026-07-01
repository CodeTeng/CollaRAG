# 创新点 2（IntentAgent）实验设计

> 目标：用一套**可独立消融、同管线、同评测**的实验，量化证明 **IntentAgent（意图感知检索）** 的有效性。
>
> 核心论点：**「按意图分道检索」优于「千篇一律直检」**。

---

## 0. 实验前置说明

### 0.1 现有指标（仓库已实装）

`src/chimera_rag/evaluation/metrics.py` 当前实装 3 个**文本匹配类**指标：

| 指标 | 含义 | 备注 |
|---|---|---|
| **EM** (Exact Match) | 归一化后预测与参考完全相等才记 1.0 | QA 任务里 ≈ **准确率 Accuracy** |
| **F1** (token-level) | token 重叠的精确率/召回率调和平均 | 内部已算 precision、recall，仅返回 F1 |
| **ROUGE-L** | 最长公共子序列的 F1 | 衡量长答案的覆盖度 |

> ⚠️ config 中预留了 `bertscore`，但 `metrics.py` 未实装。

### 0.2 建议补充的指标（当前缺失，更对口创新点 2）

| 指标 | 衡量对象 | 是否已实装 |
|---|---|:---:|
| **意图分类准确率 (Intent Accuracy)** | `LLMIntentClassifier` 6 类分类是否分对 | ❌ 需补 |
| **检索 Precision@k / Recall@k** | top-k chunk 中真证据的比例 | ❌ 需补（需数据集带证据标注） |
| **token Precision / Recall** | 从 `token_f1` 内部暴露出来 | ❌ 需补（改动极小） |

### 0.3 数据集与配置（仓库现状）

- **数据集 loader**（`datasets/`）：`mock_wiki`（冒烟用，仅 10 QA）、`hotpotqa`、`two_wiki`、`musique`、`plain_text`、`generic_jsonl`
- **消融配置**（`configs/`）：`deepseek_vanilla` / `deepseek_intent_only` / `deepseek_adagraph_only` / `deepseek_full`
- ⚠️ 论文级结论必须接**真实多跳数据集**（hotpotqa / 2wiki / musique），`mock_wiki` 仅够冒烟。

---

## 1. 实验总表（速览）

| # | 实验名称 | 对比方法 | 核心指标 | 优先级 |
|---|---|---|---|:---:|
| 1 | 主消融对比 | vanilla / intent_only / adagraph_only / full | EM, F1, ROUGE-L | 🔴 P0 |
| 2 | 按意图分组归因 | intent_only vs vanilla（按 6 类意图拆分） | 各意图的 EM/F1/ROUGE-L + 延迟 + Token | 🔴 P0 |
| 3 | 策略路由有效性 | 正常路由 vs 强制全 direct（反事实） | EM, F1, 策略命中分布, 意图分类准确率 | 🟢 P2 |
| 4 | Hook 组件消融 | base / +rerank / +reflect / all | F1, ROUGE-L, reflection_quality 相关性 | 🟡 P1 |
| 5 | 效率 / 成本权衡 | intent_only vs vanilla | 延迟 P50/P95, Token/问, ΔF1÷Δ成本 | 🟡 P1 |
| 6 | **与 SOTA 框架横向对比** | Zero-shot LLM / NaiveRAG / GraphRAG / LightRAG / HippoRAG / YoutuGraphRAG | EM, F1, ROUGE-L, Token/问, 延迟 | 🔴 P0 |

---

## 2. 实验 1 · 主消融对比 🔴 P0

**目的**：回答「IntentAgent 到底有没有用」，是论文主表。

**对比方法（4 组，同数据集、同 limit、同管线代码，差异只由 yaml 驱动）**：

| 配置 | AdaGraph | IntentAgent | 作用 |
|---|:---:|:---:|---|
| `deepseek_vanilla` | ❌ | ❌ | 基线下界 |
| `deepseek_intent_only` | ❌ | ✅ | **创新点 2 净增益（主对比）** |
| `deepseek_adagraph_only` | ✅ | ❌ | 对照，隔离创新点 1 |
| `deepseek_full` | ✅ | ✅ | 验证正交性 / 超加性 |

**指标**：EM、F1、ROUGE-L（三者全用）。

**关键看点**：
- `intent_only − vanilla` = 创新点 2 的净贡献。
- 正交性验证：`full − adagraph_only` 应 ≈ `intent_only − vanilla`。

**执行命令**：
```bash
for cfg in vanilla intent_only adagraph_only full; do
  uv run python main.py infer --config configs/deepseek_${cfg}.yaml --dataset hotpotqa --limit 200
done
uv run python eval.py \
  --predictions output/hotpotqa/deepseek_{vanilla,intent_only,adagraph_only,full}/ \
  --out output/hotpotqa/compare/ --metrics em,f1,rouge_l
# → compare.md / compare.json / compare.csv + compare_bar.png + compare_radar.png
```

---

## 3. 实验 2 · 按意图分组归因 🔴 P0

**目的**：创新点 2 **最有说服力**的实验——证明「分道」的合理性。报告第四章「按意图分组」原生支持。

**对比方法**：`intent_only` vs `vanilla`，把每个指标**按 6 类意图拆开**横向对比。

**指标**：每类意图下的 EM / F1 / ROUGE-L + 平均延迟 + 平均 Token。

**预期结论**：
- `multi_hop / comparative / analytical / follow_up` 等**复杂意图提升明显**（路由到 multi_hop / parallel / iterative / context_aware）。
- `factual` 提升有限（它本来就走 direct ≈ vanilla）。
- 「提升分布不均」恰好反向证明路由的价值。

**执行**：复用实验 1 的预测产物，直接看单目录报告的「按意图分组」与「策略分布」两章即可。

---

## 4. 实验 3 · 策略路由有效性（机制验证）🟢 P2

**目的**：证明增益确实来自「意图→策略」路由，而非其他因素。

**对比方法**：
1. **路由命中分布**：报告第五章「策略分布」直接给出各策略命中次数，核对是否与数据集真实意图分布吻合。
2. **反事实实验**：临时改 `reasoning.py::_INTENT_TO_STRATEGY`，把所有意图强制路由到 `direct`，与正常路由对比。若性能掉回接近 vanilla，则归因成立。
3. **意图分类准确率**（需数据集带意图标注）：统计 `LLMIntentClassifier` 6 类分类准确率；并扫描 `confidence_threshold`（默认 0.55）观察 LLM↔规则兜底的触发比例。

**指标**：EM、F1、**策略命中分布**、**意图分类准确率**。

---

## 5. 实验 4 · Hook 组件消融 🟡 P1

**目的**：拆解 3 个 Hook（`hooks.*.enabled` 可独立开关）各自的贡献。

**对比方法**：

| 实验组 | reflection | reranking | parallel_pool | 验证目标 |
|---|:---:|:---:|:---:|---|
| base | ❌ | ❌ | ✅ | 仅四阶段 M-R-D-E |
| +rerank（token_overlap） | ❌ | ✅ token_overlap | ✅ | TokenOverlapReranker 的增益 |
| +rerank（cross_encoder） | ❌ | ✅ cross_encoder | ✅ | **CrossEncoder 的语义重排增益（最可能提升 F1）** |
| +reflect（heuristic） | ✅ heuristic | ❌ | ✅ | heuristic reflection_quality 与真实 F1 的相关性 |
| +reflect（llm_reflection） | ✅ llm_reflection | ❌ | ✅ | **Self-Reflection 多轮循环的净增益（策略回退驱动重检索）** |
| all | ✅ llm_reflection | ✅ cross_encoder | ✅ | 全开（最强配置） |

**指标**：F1、ROUGE-L；外加 **reflection_quality 与真实 F1 的皮尔逊相关系数**。

**注意**：
- **Reranking** 是直接改检索顺序的 hook，CrossEncoder 比 TokenOverlap 更可能提升 F1（语义匹配 vs 词面匹配）。
- **Self-Reflection 现已驱动重检索**（`_execute_with_reflection` 多轮循环 + 策略回退），不再是"仅观测写回"；可以设计「反思后性能提升」的实验。`+reflect(llm_reflection)` 组预期在复杂意图上有净增益。

---

## 6. 实验 5 · 效率 / 成本权衡 🟡 P1

**目的**：审稿人必问——增益是否划算。报告自带耗时分布与 Token 分布。

**对比方法**：`intent_only` vs `vanilla`。

**指标**：
- **延迟**：P50 / P95（重点看 parallel 并发、multi_hop/expansion 放大 hops 的开销）。
- **Token**：IntentAgent 多一次意图分类 LLM 调用，统计每问平均增量 Token。
- **性价比**：`ΔF1 ÷ Δ延迟` 或 `ΔF1 ÷ ΔToken`。

---

## 7. 实验 6 · 与 SOTA 框架横向对比 🔴 P0

**目的**：**CCF-A 论文的硬门槛**。仅有内部消融（实验 1~5）不足以证明竞争力，必须与近 3 年顶会/顶刊上有代表性的 RAG / GraphRAG 框架在**同数据集、同 LLM、同评测**下横向对比，回答「Chimera-RAG（intent_only / full）相比当前 SOTA 强在哪」。

### 7.1 对比的 Baseline 框架（近 3 年代表性工作）

按「是否用图谱」分三组，覆盖无检索（纯 LLM 参数知识）、朴素检索、图谱检索三条技术路线：

| 类别 | 框架 | 出处（会议/年份） | 核心机制 | 选它的理由 |
|---|---|---|---|---|
| 无检索 | **Zero-shot LLM** | 通用基线（直接 prompt LLM） | 不检索，仅靠 LLM 参数化知识直接作答 | 绝对下界，量化「检索本身带来多少增益」 |
| 朴素 | **NaiveRAG / Vanilla RAG** | 通用基线（Lewis et al., NeurIPS 2020） | chunk + 稠密检索 + 生成 | 公认检索下界，对应本仓库 `deepseek_vanilla` |
| 图谱 | **GraphRAG** (Microsoft) | 2024（Edge et al.） | 社区检测 + 全局/局部摘要 | 图谱 RAG 事实标准，必比 |
| 图谱 | **LightRAG** | 2024（Guo et al.） | 双层（low/high-level）图检索 | GraphRAG 的高效替代，对照组高频出现 |
| 图谱+记忆 | **HippoRAG / HippoRAG 2** | NeurIPS 2024 / ICML 2025 | 个性化 PageRank + 海马索引 | 多跳 KG-RAG 当前最强基线之一 |
| 图谱+智能体 | **YoutuGraphRAG** (腾讯优图) | ICLR 2026（Dong et al., 2025） | 图模式（Graph Schema）垂直统一构建/索引/检索 + 分层知识树 + 双感知社区发现 + 智能体检索 | 2025 最新开源 SOTA，多跳推理强、构建成本省 90%+，最贴近本项目「智能体化图检索」路线 |

> 📌 必比的全集（P0）：**Zero-shot LLM + NaiveRAG + GraphRAG + LightRAG + HippoRAG + YoutuGraphRAG**——既包含无检索/朴素检索下界，又覆盖图谱 RAG 方向审稿人默认期待的全部主流对照（含 2025 最新 YoutuGraphRAG）。
>
> 💡 设置两端下界（Zero-shot LLM、NaiveRAG）的意义：`NaiveRAG − Zero-shot LLM` 量化「检索」的价值，`本方法 − NaiveRAG` 量化「图谱 + 意图路由」相对朴素检索的净增益，证据链更完整。

### 7.2 对比方法（公平性是关键）

为保证对比公平、避免审稿人质疑，必须**控制三个变量一致**：

1. **同数据集 + 同切分**：HotpotQA / 2WikiMultiHopQA / MuSiQue 的同一 test/dev 子集与同一 `limit`。
2. **同 LLM + 同 embedding**：所有 baseline 与本方法都用同一 DeepSeek 模型 + 同一 `all-MiniLM-L6-v2`，杜绝"赢在更强 LLM"。
3. **同评测脚本**：所有框架的预测统一转成本仓库 `predictions.jsonl` 格式，再用 `eval.py` 打分，确保 EM/F1/ROUGE-L 口径完全一致。

参赛选手（本方法侧）：`deepseek_intent_only`（主打创新点 2）与 `deepseek_full`（完整系统）两栏都列出。

### 7.3 指标

EM、F1、ROUGE-L（与内部实验同口径）+ **每问 Token** + **平均延迟**（体现「同等效果下更省/更快」或「同等成本下更准」）。

### 7.4 落地方式

baseline 框架**不在本仓库内实现**，建议二选一：

- **A（推荐）**：用各框架官方开源实现在相同数据集上跑出预测 → 转成 `predictions.jsonl` → 用本仓库 `eval.py` 统一打分。各框架开源现状：
  - **Zero-shot LLM**：无需 repo，直接用同一 DeepSeek 模型零样本 prompt 作答即可。
  - **NaiveRAG**：直接复用本仓库 `deepseek_vanilla`（chunk + 稠密检索 + 生成）。
  - **GraphRAG**：微软官方 repo `microsoft/graphrag`。
  - **LightRAG**：官方 repo `HKUDS/LightRAG`。
  - **HippoRAG**：官方 repo `OSU-NLP-Group/HippoRAG`（含 HippoRAG 2）。
  - **YoutuGraphRAG**：腾讯优图官方 repo `TencentCloudADP/youtu-graphrag`（2025 开源，ICLR 2026）。
- **B**：直接引用各原论文在 HotpotQA/2Wiki/MuSiQue 上的公开数值（需脚注标注「数据集子集/LLM 可能不同，非严格可比」），作为弱对照补充。

> ⚠️ A 与 B 不要混用在同一张主表里；若用 B，必须在表注明确标注来源，否则会被审稿人判为不可比。

### 7.5 预期结论与卖点

- 相比 **Zero-shot LLM**：大幅领先，证明检索的必要性（坐实「问题非靠参数化知识可解」，多跳数据集尤其明显）。
- 相比 **NaiveRAG**：全面领先（证明图谱 + 意图路由的价值，且 `本方法 − NaiveRAG` 是相对朴素检索的净增益）。
- 相比 **GraphRAG/LightRAG**：在**多跳/比较类**问题上 F1 更高，且**意图路由让简单问题不浪费多跳开销**（Token/延迟更优）→ 这是 IntentAgent 的差异化卖点。
- 相比 **HippoRAG**：强调本方法的**可插拔消融能力 + 意图分道**这一正交创新维度（即使指标接近，方法论新颖性是加分项）。
- 相比 **YoutuGraphRAG**（2025 最新 SOTA、同为智能体化图检索路线）：即便指标接近，也要突出本方法的**意图感知分道路由 + 可独立消融的 Hook 架构**这一正交差异——YoutuGraphRAG 走「图模式垂直统一」，本方法走「按意图分道 + 即插即用组件」，两者创新维度不同，可作为同期工作并列对比凸显差异化；若在简单意图上 Token/延迟更省，则进一步坐实「不千篇一律重检索」的效率优势。

---

## 8. 优先级与执行建议

| 优先级 | 实验 | 理由 |
|---|---|---|
| 🔴 P0 | 实验 1 主消融 + 实验 2 按意图分组 + **实验 6 SOTA 横向对比** | 论文主结论 + 竞争力证明，缺一不可 |
| 🟡 P1 | 实验 5 效率成本 + 实验 4 Reranking 消融 | 审稿人必问 + 组件贡献 |
| 🟢 P2 | 实验 3 路由反事实 + 意图分类准确率 | 机制深挖，加分项 |

---

## 9. 两个诚实提醒（基于代码现状）

来自 `docs/innovation_intent_agent.md` 第一部分（静态路由版），以下为**占位实现**，做实验时勿过度宣称：

1. **`IterativeStrategy` 目前是单轮直检**（MVP），并非真正多轮分解 → `analytical` 意图提升有限属正常，非 bug。
2. ~~**Reflection 不触发重检索**~~ → **已实装** Self-Reflection 多轮循环（创新点 2d），策略回退驱动重检索，可以设计「反思后性能提升」的实验。
