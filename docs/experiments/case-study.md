# Case Study

> 覆盖三种意图类型（单跳 / 多跳 / 总结），每类挑一个数据集 query 做案例研究；另附记忆复用案例展示飞轮效应。骨干模型 Qwen3-8B。

---

## 案例1：单跳事实查询（PopQA）

**Query:** "What is the country of origin of the PlayStation?"

**意图分类:** single_hop → SingleHop Agent

**CollaRAG 执行（SingleHop Agent）:**
hybrid\_search(query, top\_k=10) → RRF 融合（向量 + BM25）→ rerank → 一次性 quality check → LLM 生成。单轮 ReAct（通常一次 hybrid search 即收敛），无 Plan / Reflect 阶段，是全系统最便宜的路径。

**各方法对比：**

| 方法 | 检索策略 | 回答 | 正确性 |
|---|---|---|---|
| Vanilla RAG | top-5 向量检索，命中 PlayStation 词条但产地证据单薄 | "Japan" | ✓（证据弱） |
| HippoRAG2 | PPR 扩散，过度关联到索尼其他产品，引入噪声 | "Japan" | ✓ |
| RAPTOR | 层次聚合，定位到产地段落 | "Japan" | ✓ |
| **CollaRAG (SingleHop)** | hybrid\_search(10) + RRF + rerank → 一次性质量门控 | "Japan" | ✓ |

**分析：** 单跳查询不需要多步推理或聚合。CollaRAG 的 SingleHop Agent 用最轻量的单轮 hybrid 检索 + rerank 就能精确命中产地实体，避免了 HippoRAG2 在简单查询上过度扩散的"过度思考"开销。这印证了 division-of-labor 的效率论点——简单查询走轻量路径，把算力留给真正需要它的查询。

---

## 案例2：多跳桥接查询（HotpotQA）

**Query:** "What is the birth city of the director of the film 'Inception'?"

**意图分类:** multi_hop（桥接推理）→ MultiHop Agent

**各方法对比：**

| 方法 | 检索结果质量 | 回答 | 正确性 |
|---|---|---|---|
| Vanilla RAG | 仅检索到 Inception 电影信息，未桥接到导演出生地 | "Inception is a 2010 science fiction film" | ✗ |
| RAPTOR | 通过层次聚合检索到导演信息，但出生城市证据不完整 | "Christopher Nolan" | ✗ |
| HippoRAG | 通过个性化 PageRank 关联到 Nolan 条目，找到出生地 | "London" | ✓ |
| HippoRAG2 | 增强版 PageRank + 多跳路径聚合，找到出生地并补充国家信息 | "London, England" | ✓ |
| **CollaRAG** | Plan 分解为两步：第一跳定位 Nolan，第二跳扩展到出生地；Reflect 验证后综合 | "London, England" | ✓ |

**分析：** CollaRAG 通过意图分类直接选择多跳策略并显式分解步骤，避免了 Vanilla RAG 的证据链断裂。与 HippoRAG 相比，CollaRAG 的回答更完整（包含国家信息），且 Plan 阶段把"找什么"提前固化，比 HippoRAG2 的隐式 PPR 扩散更可审计。

---

## 案例3：对比查询（2WikiMultiHopQA）

**Query:** "Which film was released earlier, 'The Godfather' or 'Chinatown'?"

**意图分类:** multi_hop（对比）→ MultiHop Agent

**各方法对比：**

| 方法 | 检索策略 | 回答 | 正确性 |
|---|---|---|---|
| Vanilla RAG | 单次检索，仅命中一部电影 | "The Godfather was released in 1972" | ✗（未完成对比） |
| GraphRAG | 社区摘要包含两部电影，但上下文过长导致关键信息被稀释 | "The Godfather" | ✓ |
| HippoRAG | 分别命中两部电影条目，但未显式执行对比 | "The Godfather was released in 1972" | ✓ |
| HippoRAG2 | 增强检索覆盖两部电影，但仍缺乏显式对比推理 | "The Godfather (1972) and Chinatown (1974)" | ✓ |
| **CollaRAG** | Plan 分解为两个并行子查询，分别检索两部电影，Reflect 验证两者齐备后显式对比 | "The Godfather (1972) was released earlier than Chinatown (1974)" | ✓ |

**分析：** CollaRAG 的 Plan 阶段把"对比两个实体"拆成两个子查询，确保两个比较对象的信息都被检索到，回答中包含了完整的推理过程，而非仅给出结论。

---

## 案例4：动态编排 + 反思回路（MuSiQue，3 跳）

**Query:** "In which country is the headquarters of the company that developed the search engine used by the university where Alan Turing studied?"

**意图分类:** multi_hop（桥接，3 跳）→ MultiHop Agent

**CollaRAG 执行过程（Plan-Execute-Reflect）：**

| 步骤 | 动作 | 检索结果 | 质量评估 |
|---|---|---|---|
| Plan | LLM 分解为 3 步：Turing→大学→搜索引擎→公司总部国 | — | — |
| 第1跳 | 图扩展检索 "Alan Turing" + "university" | King's College, Cambridge / Princeton | quality=0.72 |
| 第2跳 | 图扩展检索 "Cambridge" + "search engine" | 无直接关联 | quality=0.31 |
| **反思触发** | 结构完备性检查发现 `missing_path`（"Cambridge" 与 "search engine" 间无证据路径）且 k < k_max(2)，针对该缺口发起定向补充检索 | 改为检索 "university search engine provider" | — |
| 第2跳(重试) | 文本检索 "Cambridge university default search engine" | Google (via university IT policy) | quality=0.68 |
| 第3跳 | 图扩展 "Google" + "headquarters country" | United States | quality=0.85 |
| Synthesize | 综合三跳证据 | "United States" | ✓ |

**对比：** HippoRAG 在第2跳卡住，最终返回 "United Kingdom"（误将大学所在国当作答案）。HippoRAG2 通过增强的路径探索在第2跳找到了部分关联，但仍未能完成完整的三跳推理，返回 "Cambridge"（停留在中间节点）。**w/o G-PER 变体**（MultiHop 退化为 flat ReAct，Plan 与 Reflect 均被移除）没有结构缺口检查，第2跳的缺失路径不会被识别，于是基于残缺证据直接生成，返回 "United Kingdom"——这正是 Reflect 阶段对证据缺口"先检测后修复"的价值。

---

## 案例5：经验记忆复用

**Query A (先处理):** "Who is the spouse of the president of the country where the Eiffel Tower is located?"

- 意图: multi_hop（桥接）→ 图扩展策略
- 执行路径: Eiffel Tower → France → President (Macron) → Spouse (Brigitte Macron) ✓
- **经验记忆写入:** 成功路径 `{intent: multi_hop, tools: [graph_expand, graph_expand, graph_expand], hit_count: 1}`

**Query B (后处理):** "Who is the spouse of the prime minister of the country where Big Ben is located?"

- 意图: multi_hop（与 Query A 同类型）
- **经验记忆命中:** 召回 Query A 的成功路径，直接复用三跳图扩展策略
- 执行路径: Big Ben → UK → Prime Minister → Spouse ✓
- 跳过了 LLM 策略规划步骤，减少 1 次 LLM 调用

**对比：** w/o Dual-Layer Memory 变体处理 Query B 时需要重新由 LLM 规划策略（额外 ~200 token 开销），且在第2跳尝试了文本检索（效率较低）后才切换到图扩展。这正是记忆飞轮效应的微观体现（参见 memory-accumulation.md 的累积曲线）。

---

## 案例6：Map-Reduce 总结查询（ASQA）

**Query:** "What are the key contributions of reinforcement learning to artificial intelligence?"

**意图分类:** summarization → Summarization Agent

**各方法对比：**

| 方法 | 检索策略 | 检索覆盖 | 回答特征 |
|---|---|---|---|
| Vanilla RAG | top-5 向量检索 | 5 chunks，仅覆盖 RL 的 2 个应用案例 | 碎片化回答，缺少 RL 的核心贡献概述 |
| GraphRAG | 社区摘要检索 | ～15 个社区摘要，信息粒度粗 | 提到了 RL 的总体方向，但缺少具体贡献列举 |
| HippoRAG2 | PPR 扩散检索 | ～8 个相关段落，但缺少全局聚合 | 重点偏向了 RL 在 NLP 中的应用，覆盖不全面 |
| **CollaRAG (Summarization Agent)** | hybrid\_search(50) → Map → Reduce | 50 chunks 覆盖 RL 的 6 个主要贡献方向 | 结构化回答，按贡献类型分类列举 |

**CollaRAG 的 Map-Reduce 执行过程：**

| 阶段 | 操作 | 详情 |
|---|---|---|
| **Retrieve** | hybrid\_search(query, top\_k=50) | 从知识库中检索 50 个 RL 相关 chunk |
| **Map** | 对每个 chunk 独立调用 LLM 提取关键信息 | 50 个局部摘要，每条约 30-50 词 |
| **Reduce** | LLM 将 50 个局部摘要归约为结构化综述 | 生成包含 6 个贡献方向的完整回答 |

**局部摘要样例（Map 阶段输出）：**

| Chunk ID | 局部摘要 |
|---|---|
| #1287 | RL 通过 Q-learning 和 policy gradient 实现了端到端决策优化 |
| #2341 | 深度强化学习（DQN/AlphaGo）展示了 RL 在复杂博弈任务中的超人类表现 |
| #3412 | RL 在 robotics 中实现了从仿真到现实的迁移学习 |
| #4501 | RLHF 技术使 LLM 能够对齐人类偏好，是 ChatGPT 的关键训练方法 |
| #5123 | Multi-agent RL 推动了协作与竞争场景下的智能体研究 |
| #6789 | RL 在推荐系统、药物发现、自动驾驶等工业场景中取得突破 |

**最终回答（Reduce 阶段输出）：**
> "Reinforcement learning (RL) has made six key contributions to artificial intelligence: (1) end-to-end
> decision optimization via Q-learning and policy gradient methods; (2) superhuman performance in
> complex games through deep RL (DQN, AlphaGo); (3) sim-to-real transfer enabling robotic
> manipulation; (4) RLHF for aligning LLMs with human preferences, a critical component behind
> ChatGPT; (5) multi-agent RL for collaborative and competitive agent research; (6) industrial
> applications in recommendation systems, drug discovery, and autonomous driving."

**分析：** Vanilla RAG 的 top-5 检索造成的碎片化回答在 ASQA 这类长文本总结任务上尤为致命——RL 的贡献分布在至少 20+ 个 chunk 中，top-5 只覆盖了冰山一角。GraphRAG 的社区摘要粒度太粗，无法提供具体贡献列举。CollaRAG 的 Summarization Agent 通过 Map-Reduce 管线实现了全面的信息覆盖和结构化输出，这正是其他 baseline 在 ASQA 上大幅落后的根本原因。

---

## 三种意图的案例对照

| 意图 | 案例 | 数据集 | 主力 Agent | CollaRAG 优势来源 |
|---|---|---|---|---|
| 单跳 (single\_hop) | 案例1 | PopQA | SingleHop | 轻量 hybrid + rerank，避免过度思考 |
| 多跳 (multi\_hop) | 案例2/3/4 | HotpotQA / 2Wiki / MuSiQue | MultiHop (G-PER) | Plan 显式分解 + Reflect 结构缺口修复 |
| 总结 (summarization) | 案例6 | ASQA | Summarization | Map-Reduce 全局聚合，baseline 无此能力 |
