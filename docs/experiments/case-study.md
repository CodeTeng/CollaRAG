# Case Study

## 案例1：多跳桥接查询（HotpotQA）

**Query:** "What is the birth city of the director of the film 'Inception'?"

**意图分类:** bridge（桥接推理）

**各方法对比：**

| 方法 | 检索结果质量 | 回答 | 正确性 |
|---|---|---|---|
| Vanilla RAG | 仅检索到 Inception 电影信息，未桥接到导演出生地 | "Inception is a 2010 science fiction film" | ✗ |
| RAPTOR | 通过层次聚合检索到导演信息，但出生城市证据不完整 | "Christopher Nolan" | ✗ |
| HippoRAG | 通过个性化 PageRank 关联到 Nolan 条目，找到出生地 | "London" | ✓ |
| HippoRAG2 | 增强版 PageRank + 多跳路径聚合，找到出生地并补充国家信息 | "London, England" | ✓ |
| **CollaRAG** | 意图识别为 bridge → 图扩展策略，第一跳定位 Nolan，第二跳扩展到出生地信息 | "London, England" | ✓ |

**分析：** CollaRAG 通过意图分类直接选择桥接策略，避免了 Vanilla RAG 的证据链断裂问题。与 HippoRAG 相比，CollaRAG 的回答更完整（包含国家信息）。

---

## 案例2：对比查询（2WikiMultiHopQA）

**Query:** "Which film was released earlier, 'The Godfather' or 'Chinatown'?"

**意图分类:** comparison（对比）

**各方法对比：**

| 方法 | 检索策略 | 回答 | 正确性 |
|---|---|---|---|
| Vanilla RAG | 单次检索，仅命中一部电影 | "The Godfather was released in 1972" | ✗（未完成对比） |
| GraphRAG | 社区摘要包含两部电影，但上下文过长导致关键信息被稀释 | "The Godfather" | ✓ |
| HippoRAG | 分别命中两部电影条目，但未显式执行对比 | "The Godfather was released in 1972" | ✓ |
| HippoRAG2 | 增强检索覆盖两部电影，但仍缺乏显式对比推理 | "The Godfather (1972) and Chinatown (1974)" | ✓ |
| **CollaRAG** | 意图识别为 comparison → 并行检索两个实体，显式对比发布日期 | "The Godfather (1972) was released earlier than Chinatown (1974)" | ✓ |

**分析：** CollaRAG 的对比策略确保了两个比较对象的信息被完整检索并显式对比，回答中包含了完整的推理过程。

---

## 案例3：动态编排 + 反思回路（MuSiQue）

**Query:** "In which country is the headquarters of the company that developed the search engine used by the university where Alan Turing studied?"

**意图分类:** bridge（多跳桥接，3跳）

**CollaRAG 执行过程：**

| 步骤 | 动作 | 检索结果 | 质量评估 |
|---|---|---|---|
| 第1跳 | 图扩展检索 "Alan Turing" + "university" | King's College, Cambridge / Princeton | quality=0.72 |
| 第2跳 | 图扩展检索 "Cambridge" + "search engine" | 无直接关联 | quality=0.31 |
| **反思触发** | quality < τ(0.5)，LLM 重新规划 | 改为检索 "university search engine provider" | — |
| 第2跳(重试) | 文本检索 "Cambridge university default search engine" | Google (via university IT policy) | quality=0.68 |
| 第3跳 | 图扩展 "Google" + "headquarters country" | United States | quality=0.85 |
| 最终回答 | — | "United States" | ✓ |

**对比：** HippoRAG 在第2跳卡住，最终返回 "United Kingdom"（误将大学所在国当作答案）。HippoRAG2 通过增强的路径探索在第2跳找到了部分关联，但仍未能完成完整的三跳推理，返回 "Cambridge"（停留在中间节点）。静态路由（w/o Dynamic）无法在第2跳失败后切换检索策略，返回空答案。

---

## 案例4：经验记忆复用

**Query A (先处理):** "Who is the spouse of the president of the country where the Eiffel Tower is located?"

- 意图: bridge → 图扩展策略
- 执行路径: Eiffel Tower → France → President (Macron) → Spouse (Brigitte Macron) ✓
- **经验记忆写入:** 成功路径 `{intent: bridge, tools: [graph_expand, graph_expand, graph_expand], hot_count: 1}`

**Query B (后处理):** "Who is the spouse of the prime minister of the country where Big Ben is located?"

- 意图: bridge（与 Query A 同类型）
- **经验记忆命中:** 召回 Query A 的成功路径，直接复用三跳图扩展策略
- 执行路径: Big Ben → UK → Prime Minister → Spouse ✓
- 跳过了 LLM 策略规划步骤，减少 1 次 LLM 调用

**对比：** w/o Memory 变体处理 Query B 时需要重新由 LLM 规划策略（额外 ~200 token 开销），且在第2跳尝试了文本检索（效率较低）后才切换到图扩展。

---

## 案例5：意图误分类的鲁棒性（Dual-Path 兜底）

**Query:** "List the Nobel Prize winners from the same university as Marie Curie."

**LLM 分类:** aggregation（聚合） — 误判，实际应为 set（集合检索）

**Dual-Path 处理：**

| 路径 | 意图判断 | 策略 | 结果 |
|---|---|---|---|
| LLM 路径 | aggregation | 社区摘要聚合 | 检索到 University of Paris 的概括性描述，但缺少具体获奖者列表 |
| 规则路径 | set（关键词匹配 "list"） | 多实体并行检索 | 检索到多位获奖者信息 |
| **融合决策** | 两路不一致 → 取规则路径（更保守） | set 策略 | ✓ |

**分析：** Dual-Path 机制在 LLM 意图分类出错时提供了规则兜底，避免了策略错配。w/o Dual-Path 变体直接使用 LLM 的错误分类，导致该查询回答不完整。
CASEOF; __aone_exit=$?; pwd -P > '/var/folders/cc/ndwfw3616fb852qjpd573r5r0000gn/T/aone-copilot-cwd-1780895687947-zllsp4aaefc.txt' 2>/dev/null; exit $__aone_exit