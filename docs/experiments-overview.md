# CollaRAG 实验体系总览

## 论文概览

**论文标题**: CollaRAG: Graph-Grounded Division of Labor for Multi-Agent Retrieval

**核心思想**: 用"分工协作"替代传统 RAG 的"一刀切"检索流水线——不同类型的查询由不同的专家 Agent 处理。

---

## 四大创新点

### 1. Intent-Aware Query Triage (意图感知查询分诊)

- 两级级联分类器（正则 + LLM 语义分类 + 关键词回退）
- 将查询路由到 5 种处理模式（greeting / single_hop / multi_hop / summarization / other）
- 不同于先前只切换检索深度的自适应方法，这里是 **Agent 级别**的分工

### 2. G-PER: Graph-Grounded Plan-Execute-Reflect (图谱锚定的规划-执行-反思)

- **Plan**: 生成图操作（link/traverse/attribute/aggregate），经 KG schema 校验
- **Execute**: 前沿式共享预算执行，简单步骤可把预算捐给困难步骤
- **Reflect**: 结构化完整性检查（typed gaps: missing_attribute/path/relation）+ 对抗性探针（在 KG 中搜索矛盾关系）
- 优雅退化：无图时退化为 text-only PER → flat ReAct

### 3. Graph-Grounded Dual-Layer Memory (图谱锚定双层记忆)

- **共享层**: 基于实体邻域重叠（Jaccard）的证据子图缓存，而非文本哈希匹配
- **私有层**: 图规划模板 + 类型化 gap lessons（来自 G-PER 的 Reflect），按 Agent 隔离
- 将结构化学习与缓存命中分离

### 4. Graph-Conditioned Capability Specialization (图谱条件化能力特化)

- 根据查询在 KG 上的"结构足迹"(hitRate, hasPath, density) 动态调整 Agent 可用工具
- 图谱贫乏区域自动撤回图遍历工具、增加文本回退
- 将静态工具权限矩阵变为自适应的图条件化能力门控

---

## 实验列表

### 1. 主实验 (Table 1) — "系统整体有效吗？"

**目的**: 证明 CollaRAG 作为一个完整系统，在多种查询类型上都优于现有方法。

**设计逻辑**:

- 6 个数据集覆盖 3 种意图（single-hop / multi-hop / summarization），直接对应"分工假设"——如果分工有效，那么在每种类型上都应比单一流水线强
- 2 个 backbone (8B / 32B) 验证架构收益不会被更强模型"吸收"
- 9 个 baseline 覆盖 4 个流派（LLM-only / text / graph / hybrid），排除"只是某类方法弱"的解释

**结果**: 所有数据集、所有 backbone 上均 SOTA，最大增益在 ASQA (+7.2 F1, 8B)

**意义**: 这是论文的"existence proof"——证明分工架构确实全面优于单一架构，尤其 ASQA 上 +7.2 F1 的最大增益直接印证了核心假设（没有 baseline 有专门的摘要聚合管线）。

---

### 2. 消融实验 (Table 2) — "每个组件贡献了多少？"

**目的**: 逐一移除四大创新点，量化每个组件的独立贡献，并排定重要性。

**设计逻辑**:

| 消融变体 | 验证什么 |
|----------|----------|
| w/o Query Triage | 路由本身的价值（退化为只用 SingleHop） |
| w/o G-PER | 图谱锚定推理 vs. flat ReAct |
| Graph-poor (no KG) | 图谱锚定 vs. 纯文本 PER（隔离图的贡献） |
| w/o Dual-Layer Memory | 记忆系统的累积学习价值 |
| w/o Capability Specialization | 动态工具门控 vs. 静态工具矩阵 |

**意义**:

- 确认 Query Triage 是最关键组件（-11.7 EM），这是对"分工假设"最强的实验证据
- 三级退化链 (G-PER → text-PER → flat ReAct) 让审稿人能分辨"图谱锚定"和"规划-反思结构"各自的贡献，避免混淆
- 防止审稿人质疑"是不是某一个组件在 carry 全局"

#### Graph-poor (no KG) 消融详解

**含义**: 只移除知识图谱，但保留 Plan-Execute-Reflect 的推理结构。

各阶段的退化情况：

| 阶段 | 有图（G-PER） | 无图（Graph-poor / text-PER） |
|------|--------------|-------------------------------|
| **Plan** | 生成图操作，经 KG schema 校验 | 退化为自由文本子查询（无 schema 可校验） |
| **Execute** | 图扩展（expand neighbors/path）作为前沿原语 | 只用文本检索（dense + BM25） |
| **Reflect** | 结构完整性检查（typed gaps）+ 对抗性探针（搜 KG 矛盾） | 退化为标量质量门（q̂ < τ 就再搜一轮） |

**三级退化链**:

```
G-PER (完整)  →  text-only PER (Graph-poor)  →  flat ReAct (w/o G-PER)
   ↓                     ↓                           ↓
图谱锚定+结构推理     纯文本结构推理              无结构推理
```

**以 2WikiMultiHopQA (Qwen3-8B, EM) 为例**:

| 变体 | EM | 相比完整系统 |
|------|-----|-------------|
| CollaRAG (G-PER) | .602 | — |
| Graph-poor (text-PER) | .587 | -1.5 |
| w/o G-PER (flat ReAct) | .511 | -9.1 |

- **图谱锚定**贡献了 .602 - .587 = **1.5 EM**
- **Plan-Reflect 结构**贡献了 .587 - .511 = **7.6 EM**

**意义**:

1. **因果归因**: 精确分离了"图谱"和"推理结构"两个变量的独立贡献
2. **验证优雅退化**: 即使完全没有知识图谱，系统也只损失 1.5 EM，不会崩溃
3. **支撑 graceful degradation 声明**: G-PER 在图谱缺失时退化为 text-only PER 的实验验证
4. **隐含结论**: 大部分增益来自 Plan-Reflect 结构本身（7.6 EM），图谱锚定是锦上添花（1.5 EM），说明系统不强依赖于高质量图谱

---

### 3. 检索质量与忠实度 (Table 3) — "答案好是因为检索好，还是生成好？"

**目的**: 将答案质量的提升归因到检索阶段，证明是"找到了更好的证据"而非"LLM 编得更好"。

**设计逻辑**:

- Recall@10 衡量检索覆盖率
- Faithfulness / Hallucination / Evidence Support 衡量生成是否忠实于检索结果
- 用独立 judge 模型 (DeepSeek-V4-flash) 评估，避免自我评估偏差

**意义**: 回应潜在质疑——"EM/F1 高可能只是 LLM 更会猜答案"。数据表明 Recall@10 高出 6-7 个点，幻觉率降到 16.7%，说明提升源于更好的证据获取。

---

### 4. 路由反事实 (Table 4) — "是路由有效，还是某个 Agent 太强？"

**目的**: 排除"一个强 Agent 单打独斗就够了"的替代解释。

**设计逻辑**:

- All → SingleHop: 只用最简单的 Agent
- All → MultiHop: 只用最强的 Agent
- Random: 随机分配
- 在 700 query 混合意图集上测试

**意义**:

- 即使全用最强的 MultiHop Agent，single-hop 查询反而变差（过度推理注入噪声）
- 没有任何单一 Agent 能覆盖所有类型 → "正确路由"本身就是性能杠杆
- 这是对 division-of-labor 假设最直接的因果验证

---

### 5. 意图分类精度 (Table 5) — "路由器本身可靠吗？"

**目的**: 量化 Query Triage 模块的分类准确率，因为消融实验已证明路由是最关键组件。

**设计逻辑**:

- 430 query 人工标注集，分层采样各意图类型
- 报告 per-class P/R/F1，不只报告 overall accuracy

**意义**:

- 89.3% 准确率、0.909 macro-F1，说明路由器本身足够可靠
- 暴露了具体的错误模式（single_hop vs. multi_hop 边界），给未来改进指明方向
- 审稿人必然会问"如果路由错了怎么办"，这个实验提前回答

---

### 6. 路由分布 (Fig 4) — "路由行为是否符合预期？"

**目的**: 验证路由器在各数据集上的分流比例是否合理（sanity check）。

**设计逻辑**:

- 可视化每个数据集的路由去向分布
- 预期：NQ/PopQA 主要走 SingleHop，HotpotQA/2Wiki/MuSiQue 主要走 MultiHop，ASQA 主要走 Summarization

**意义**:

- 证明系统行为可解释——路由分布确实与数据集的设计意图对齐
- 量化了"off-diagonal"比例（5-12%），与 Table 5 的分类精度一致
- 解释了主实验中各数据集增益大小的来源（ASQA 上 76.8% 走 Summarization → +7.2 F1）

---

### 7. 多跳深度边界 (Fig 5) — "系统的能力边界在哪里？"

**目的**: 暴露系统的局限性——随着推理深度增加，性能如何衰减。

**设计逻辑**:

- MuSiQue 按 2/3/4-hop 分层
- 对比 CollaRAG vs. HippoRAG2 在各深度的表现

**意义**:

- 诚实展示 limitation：4-hop 时优势从 +3.0 收窄到 +1.5 EM
- 解释了原因：反思预算 k_max=2 不足以修复最深的证据链
- 给 Future Work 提供明确方向（增加反思轮次或层次化分解）
- 审稿人喜欢看到作者清楚自己方法的边界

---

### 8. 记忆积累曲线 (Fig 6) — "记忆是真正在学习，还是只是缓存？"

**目的**: 证明双层记忆的"累积学习"效应是真实的结构化学习，而非简单的问题重复。

**设计逻辑**:

- 按时间顺序处理 HotpotQA 的 500 个 query
- 跟踪 EM/F1 随查询量增长的变化曲线
- 报告 overlap-hit rate

**意义**:

- 冷启动 0.422 → 热启动 0.520（+23.2% 相对提升），证明记忆有效
- 8.5% overlap-hit rate 说明是实体邻域重叠驱动的结构化复用，不是问题文本匹配
- 300 query 后趋于平坦 → 容量设计合理（B_max=200）
- 回应了"在不重复问题的基准上，记忆有什么用？"这个潜在质疑

---

### 9. 超参敏感性 (Fig 7) — "系统是否鲁棒？"

**目的**: 证明系统不依赖极端调参，默认值附近性能平坦。

**设计逻辑**:

- 逐一扫描 5 个核心超参（plan 长度 n_max、质量阈值 τ、反思次数 k_max、每步迭代 j_max、top-k）
- 固定其他参数，只变一个

**意义**:

- 扰动一步最多损失约 2 EM（通常 <1），说明系统稳健
- 暴露了有趣的 trade-off：plan 过长会导致幻觉步骤（valid-plan rate 从 96.2% 降到 88.1%）
- 反思轮次的收益递减清晰（0→1: +2.6, 1→2: +0.5, 2→3: +0.1），证明 k_max=2 是最优性价比点
- 防止审稿人质疑"结果是不是精心调出来的"

---

## 实验体系的整体逻辑

```
主实验 ──→ "有效吗？"（全面 SOTA）
   │
消融实验 ──→ "每个组件贡献多少？"（路由最重要）
   │
   ├─ 路由反事实 ──→ "是路由有效 vs. Agent 强？"（因果验证）
   ├─ 意图分类精度 ──→ "路由器本身靠谱吗？"（模块验证）
   ├─ 路由分布 ──→ "行为可解释吗？"（sanity check）
   │
   ├─ 检索质量/忠实度 ──→ "提升来源是什么？"（归因分析）
   ├─ 记忆积累 ──→ "记忆真的在学习吗？"（机制验证）
   │
   ├─ 多跳深度边界 ──→ "极限在哪？"（limitation）
   └─ 超参敏感性 ──→ "鲁棒吗？"（工程可靠性）
```

**设计核心逻辑**: 先证明有效 → 再解释为什么有效 → 最后坦诚边界在哪，层层递进地回应审稿人可能提出的每一个质疑。
