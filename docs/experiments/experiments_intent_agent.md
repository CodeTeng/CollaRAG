# 创新点 2（IntentAgent 多Agent架构）实验设计

> 目标：用一套**可独立消融、同管线、同评测**的实验，量化证明 **IntentAgent（多Agent意图感知检索）** 的有效性。
>
> 核心论点：**「按意图分派到专职 Agent」优于「千篇一律直检」**。

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
| **意图分类准确率 (Intent Accuracy)** | `TreeIntentClassifier` 5 类分类（greeting/single_hop/multi_hop/summarization/other）是否分对 | ❌ 需补 |
| **检索 Precision@k / Recall@k** | top-k chunk 中真证据的比例 | ❌ 需补（需数据集带证据标注） |
| **token Precision / Recall** | 从 `token_f1` 内部暴露出来 | ❌ 需补（改动极小） |

### 0.3 数据集与配置（仓库现状）

- **数据集 loader**（`datasets/`）：`mock_wiki`（冒烟用，仅 10 QA）、`hotpotqa`、`two_wiki`、`musique`、`plain_text`、`generic_jsonl`
- **消融配置**（`configs/`）：`deepseek_vanilla` / `deepseek_intent_only` / `deepseek_adagraph_only` / `deepseek_full` / `ollama_intent_only`
- ⚠️ 论文级结论必须接**真实多跳数据集**（hotpotqa / 2wiki / musique），`mock_wiki` 仅够冒烟。

---

## 1. 实验总表（速览）

| # | 实验名称 | 对比方法 | 核心指标 | 优先级 |
|---|---|---|---|:---:|
| 1 | 主消融对比 | vanilla / intent_only / adagraph_only / full | EM, F1, ROUGE-L | 🔴 P0 |
| 2 | 按意图分组归因 | intent_only vs vanilla（按 5 类意图拆分） | 各意图的 EM/F1/ROUGE-L + 延迟 + Token | 🔴 P0 |
| 3 | Agent 路由有效性 | 正常路由 vs 强制全走 Native RAG（反事实） | EM, F1, Agent 命中分布, 意图分类准确率 | 🟢 P2 |
| 4 | 多Agent组件消融 | Full / w/o Preprocess / w/o TreeIntent / w/o Reflect / w/o Memory / w/o BM25 / w/o Graph | EM, F1, ROUGE-L | 🟡 P1 |
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

**目的**：创新点 2 **最有说服力**的实验——证明「分派到专职 Agent」的合理性。

**对比方法**：`intent_only` vs `vanilla`，把每个指标**按 5 类意图拆开**横向对比。

**5 类意图 → 路由到的 Agent**：

| 意图 | 路由 Agent | 典型问法 |
|---|---|---|
| `greeting` | LLM 直答（短路，不进入任何 Agent） | "你好" |
| `single_hop` | Native RAG Agent | "爱因斯坦的出生地？" |
| `multi_hop` | ReAct Agent（含比较型、探索型、追问型，靠 ReAct 灵活覆盖） | "比较 A 和 B 的贡献差异" |
| `summarization` | Map-Reduce Agent | "总结这篇文档的主要观点" |
| `other` | WebSearch + LLM Agent | "今天东京天气怎样？" |

**指标**：每类意图下的 EM / F1 / ROUGE-L + 平均延迟 + 平均 Token。

**预期结论**：
- `multi_hop`（涵盖原来的比较/探索/追问语义）**提升明显**，因为路由到 ReAct Agent 的工具选择循环 + 反思。
- `single_hop` 提升有限（它本来就走轻量 Native RAG ≈ vanilla 直检）。
- `summarization` 单独看增益最大——这是 vanilla 完全不具备的能力（vanilla 只会退化为直接检索+生成，没有 Map-Reduce 式全局聚合）。
- 「提升分布不均」恰好反向证明按意图分派的价值。

**执行**：复用实验 1 的预测产物，`predictions.jsonl` 每行的 `intent_label` 字段（新架构下取值为 `greeting/single_hop/multi_hop/summarization/other`）用于分组统计。

---

## 4. 实验 3 · Agent 路由有效性（机制验证）🟢 P2

**目的**：证明增益确实来自「意图→Agent」路由，而非其他因素。

**对比方法**：
1. **路由命中分布**：统计 `predictions.jsonl` 中各 `intent_label` 出现频次，核对是否与数据集真实意图分布吻合。
2. **反事实实验**：临时修改 `MultiAgentRetriever.retrieve()`，把所有意图强制路由到 `single_hop`（即所有查询都走 Native RAG Agent），与正常路由对比。若性能掉回接近 vanilla，则归因成立。
3. **意图分类准确率**（需数据集带意图标注）：统计 `TreeIntentClassifier` 5 类分类准确率；并扫描 `classifier.confidence_threshold`（默认 0.7）观察 LLM↔规则兜底的触发比例。

**指标**：EM、F1、**Agent 命中分布**、**意图分类准确率**。

---

## 5. 实验 4 · 多Agent组件消融 🟡 P1

**目的**：拆解多Agent架构各组件的独立贡献。

**对比方法（对应设计文档 `docs/superpowers/specs/2026-07-07-multi-agent-architecture-design.md` §11 的消融表）**：

| 实验组 | 做法 | 验证目标 |
|---|---|---|
| Full | 完整多Agent架构 | 基线（上界） |
| w/o Preprocess | 跳过 Preprocessor Agent，原始 query 直接分类 | 预处理（查询改写/指代消解/追问拼接）的价值 |
| w/o TreeIntent | 所有 query 统一走 ReAct Agent | 意图路由的必要性 |
| w/o Reflect | ReAct Agent 只跑单轮，不做反思循环 | 反思机制的贡献 |
| w/o Memory | 禁用双层记忆（SharedMemory + AgentPrivateMemory 都关） | 经验记忆的累积价值 |
| w/o SharedMem | 只禁用共享记忆，保留 Agent 私有记忆 | 共享 vs 私有记忆的贡献分离 |
| w/o BM25 | 去掉 BM25 和混合检索，只用向量检索 | BM25/混合检索的增益 |
| w/o Graph | 去掉图相关工具（graph_neighbors/graph_path_search/graph_community_search） | 知识图谱工具的贡献 |
| w/o MapReduce | 归纳总结类查询退化为 Native RAG Agent | Map-Reduce 的增量价值 |

**指标**：EM、F1、ROUGE-L。

> ⚠️ 上述消融变体目前**均未实装为可直接开关的 yaml 配置**，需要临时代码改动（如注释掉 Preprocessor 调用、固定路由标签等）才能跑。若要把它们落地为正式的 yaml 消融开关（类似旧版 `hooks.*.enabled`），需要额外的工程量——这属于新增实验基建，见文末「需要确认的后续实验」。

---

## 6. 实验 5 · 效率 / 成本权衡 🟡 P1

**目的**：审稿人必问——增益是否划算。

**对比方法**：`intent_only` vs `vanilla`。

**指标**：
- **延迟**：P50 / P95（重点看 ReAct Agent 的多轮工具调用、Map-Reduce Agent 的逐 chunk map 阶段的开销）。
- **Token**：IntentAgent 多一次意图分类 LLM 调用 + Preprocessor 的改写调用，统计每问平均增量 Token。
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
> 💡 设置两端下界（Zero-shot LLM、NaiveRAG）的意义：`NaiveRAG − Zero-shot LLM` 量化「检索」的价值，`本方法 − NaiveRAG` 量化「图谱 + 多Agent意图路由」相对朴素检索的净增益，证据链更完整。

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
- 相比 **NaiveRAG**：全面领先（证明图谱 + 多Agent意图路由的价值，且 `本方法 − NaiveRAG` 是相对朴素检索的净增益）。
- 相比 **GraphRAG/LightRAG**：在**多跳/归纳总结类**问题上 F1 更高，且**意图路由让简单问题不浪费多轮工具调用开销**（Token/延迟更优）→ 这是 IntentAgent 的差异化卖点。
- 相比 **HippoRAG**：强调本方法的**可插拔消融能力 + 意图分道**这一正交创新维度（即使指标接近，方法论新颖性是加分项）。
- 相比 **YoutuGraphRAG**（2025 最新 SOTA、同为智能体化图检索路线）：即便指标接近，也要突出本方法的**意图感知分道路由 + 专职 Agent + 可独立消融的工具架构**这一正交差异——YoutuGraphRAG 走「图模式垂直统一」，本方法走「按意图分派专职 Agent + 即插即用工具」，两者创新维度不同，可作为同期工作并列对比凸显差异化；若在简单意图上 Token/延迟更省，则进一步坐实「不千篇一律重检索」的效率优势。

---

## 8. 优先级与执行建议

| 优先级 | 实验 | 理由 |
|---|---|---|
| 🔴 P0 | 实验 1 主消融 + 实验 2 按意图分组 + **实验 6 SOTA 横向对比** | 论文主结论 + 竞争力证明，缺一不可 |
| 🟡 P1 | 实验 5 效率成本 + 实验 4 多Agent组件消融 | 审稿人必问 + 组件贡献 |
| 🟢 P2 | 实验 3 路由反事实 + 意图分类准确率 | 机制深挖，加分项 |

---

## 9. 两个诚实提醒（基于代码现状）

1. **实验 4 的消融变体目前不是 yaml 一键开关**。旧版架构有 `hooks.reflection.enabled` / `hooks.reranking.enabled` 这类现成开关，新的多Agent架构下 `w/o Preprocess`、`w/o TreeIntent`、`w/o BM25` 等变体需要临时代码改动才能跑（例如让 `MultiAgentRetriever` 跳过预处理步骤、固定路由到某个 Agent、从 `ToolSetBuilder` 的 `TOOL_MATRIX` 里剔除 `bm25_search`）。若要把它们变成正式消融开关，需要额外工程量。
2. **ReAct Agent 的 `max_iterations` 默认值为 3**（新架构，比旧版 LangGraph 的 `max_iterations=2` 多一轮，因为工具集更丰富）。跑效率对比实验时注意这个默认值差异。

---

## 需要确认的后续实验（未直接添加，等待确认）

以下是新多Agent架构下**可能值得补充**的实验方向，但目前尚未在本文档中正式定义为消融变体（也未落地为 yaml 开关）。是否要将它们纳入实验计划，请明确告知：

1. **消融实验的 yaml 化**：把 §5 表格中的 8 个消融变体（w/o Preprocess/TreeIntent/Reflect/Memory/SharedMem/BM25/Graph/MapReduce）落地为 `configs/` 下可直接跑的 yaml + 对应的代码开关（类似旧版 `hooks.*.enabled` 的做法），而不是要求临时改代码。
2. **Map-Reduce Agent 专项实验**：单独评测归纳总结类查询（需要专门构造或筛选"总结/综述"类 QA 数据，现有 HotpotQA/2Wiki/MuSiQue 均为事实问答，不含摘要类问题）。
3. **WebSearch Agent 专项实验**：单独评测超出知识库范围的查询（需要构造知识库之外的问题集，现有多跳数据集不适用）。
4. **双层记忆的飞轮效应曲线**（对标旧版 `docs/result/消融实验.md` 中的经验记忆增长曲线实验），验证新架构 `AgentPrivateMemory` 是否有同等的跨查询学习效应。
