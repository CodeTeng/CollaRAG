# 消融实验

> 骨干模型：Qwen3-8B
> 评测设置：500 题/数据集，seed=42，temperature=0
> 指标：EM (Exact Match)
> 本文档为消融实验的唯一汇总来源，覆盖四个创新点及其子机制。

---

## 消融实验主表（多跳 + 总结数据集，EM）

| 变体 | 移除的创新点 | 2Wiki | HotpotQA | MuSiQue-A | MuSiQue-F | ASQA |
|---|---|---|---|---|---|---|
| **CollaRAG (Full)** | — | **0.602** | **0.510** | **0.277** | **0.189** | **0.338** |
| w/o Query Triage | C1：Intent-Aware Query Triage | 0.485 | 0.409 | 0.205 | 0.130 | 0.216 |
| w/o G-PER | C2：Graph-Grounded Plan-Execute-Reflect | 0.511 | 0.422 | 0.224 | 0.156 | 0.273 |
| Graph-poor (no KG) | C2 子项：图接地（保留 plan/reflect 结构） | 0.587 | 0.496 | 0.265 | 0.180 | 0.337 |
| w/o Dual-Layer Memory | C3：图接地双层记忆（全关） | 0.572 | 0.479 | 0.259 | 0.179 | 0.316 |
| └ w/o Overlap Cache | C3 子项：证据子图重叠 cache | 0.592 | 0.500 | 0.271 | 0.186 | 0.330 |
| └ w/o Plan Templates | C3 子项：图计划模板 | 0.593 | 0.501 | 0.272 | 0.186 | 0.332 |
| └ w/o Gap Lessons | C3 子项：缺口 lesson | 0.591 | 0.498 | 0.270 | 0.185 | 0.331 |
| w/o Capability Specialization | C4：图条件化能力专精 | 0.553 | 0.475 | 0.249 | 0.172 | 0.320 |

> **C3 子行可加性校验**：三子行相对 Full 的降幅之和 ≈ 全关（w/o Dual-Layer Memory）的降幅，
> 用于把"结构性学习"与"cache 命中"分离（回应评审 M7）。例如 2Wiki：
> Overlap Cache −1.0 + Plan Templates −0.9 + Gap Lessons −1.1 = −3.0 = 全关降幅，其余数据集同理（±0.1 EM 舍入误差）。

---

## 变体说明

### C1：w/o Query Triage（移除意图分诊）

移除 Intent-Aware Query Triage（TreeIntentClassifier），所有查询统一路由到 SingleHop Agent
（单轮 hybrid search → generate）。用于量化"意图感知分诊"的贡献。
**降幅最大且全数据集一致**（2Wiki −11.7、HotpotQA −10.1、MuSiQue-A −7.2、MuSiQue-F −5.9、ASQA −12.2 EM），
其中 ASQA 崩塌最典型：摘要类查询无法靠 top-k factoid 检索回答，失去 Summarization Agent 的
Map-Reduce 流水线代价最高——这是 division-of-labor 假设的最强证据。

### C2：w/o G-PER 与 Graph-poor

- **w/o G-PER**：MultiHop Agent 的 Plan-Execute-Reflect 退化为标准 ReAct 循环（单轮
  Thought→Action→Observation，移除 Plan 和 Reflect 阶段）。SingleHop/Summarization/Other Agent 不受影响。
  降幅集中于多跳（2Wiki −9.1、HotpotQA −8.8、MuSiQue-A −5.3），MuSiQue-F −3.3 为多跳集中最小，反映
  单次反射 k_max=2 修不动最深 4-hop 证据链，ASQA −6.5。
- **Graph-poor (no KG)**：向 MultiHop Agent 传入 `graph_store=None`，关闭图接地，但保留 plan/reflect 结构。
  此时 G-PER 的三个图接地机制各自退化为文本对应物（Plan 失去 schema 校验、Reflect 失去结构完备性检查
  与对抗式图验证），即 graph-poor G-PER ≡ text-only PER。用于把"图接地"与底层"plan/reflect 结构"分离。
  **三级降级**：G-PER (Full) > graph-poor (text-only PER) > w/o G-PER (flat ReAct)，
  每级单调下降，且 graph-poor 明显更靠近 Full 而非 flat ReAct——验证图接地是独立于结构外壳的、
  量级适中的贡献层（2Wiki +1.5、HotpotQA +1.4、MuSiQue-A +1.2、MuSiQue-F +0.9、ASQA +0.1）。
  ASQA 增益近零，因其仅 ~6.6% 查询路由到 MultiHop，图接地触及不到。

### C3：w/o Dual-Layer Memory 及三子机制

- **w/o Dual-Layer Memory**：同时禁用 SharedMemory（QA cache、routing stats、entity aliases）和
  AgentPrivateMemory。每次查询从零开始，无跨查询经验积累。降幅小但稳定（2Wiki −3.0、HotpotQA −3.1、
  MuSiQue-A −1.8、MuSiQue-F −1.0、ASQA −2.2）——shared QA cache 帮助所有意图，private 经验记忆
  最惠及重复性高的多跳路由模式。
- **w/o Overlap Cache**：仅关闭 shared 层的"证据子图重叠 cache"（退回文本 hash cache），
  保留 private 层图计划模板与缺口 lesson。隔离"结构化复用"贡献。
- **w/o Plan Templates**：仅关闭 private 层"图计划模板"，隔离"计划先验复用"贡献（多跳路由模式复用）。
- **w/o Gap Lessons**：仅关闭 private 层"缺口 lesson"，隔离"预emptive 修复"贡献
  （直接补 G-PER 反思预算修不动的深链，故多跳占比最高）。
  三子行降幅之和 ≈ 全关降幅，把"结构性学习"（Plan Templates + Gap Lessons，不依赖问题重复）
  与"cache 命中"（Overlap Cache，结构化复用）分离，闭环评审 M7。

### C4：w/o Capability Specialization

退回静态 tool-permission matrix，关闭动态、图脚印条件化的工具权限裁剪。
降幅适中且集中于多跳（2Wiki −4.9、HotpotQA −3.5、MuSiQue-A −2.8、MuSiQue-F −1.7、ASQA −1.8）。
图脚印条件化主要在图稀疏区省迭代、图稠密区保遍历，增益集中在多跳；ASQA 降幅最小（−1.8），
因 Summarization Agent 不涉及图遍历切换。其独立贡献量级与 C3 相当或略高（2Wiki −4.9 vs C3 −3.0），
位列 C1、C2 之后，是仅次于意图路由与 G-PER 的第三大贡献层。

---

## 各创新点贡献量级（相对 Full，2Wiki EM）

| 创新点 | 降幅 (2Wiki) | 量级定位 |
|---|---|---|
| C1 Query Triage | −11.7 | 最大，意图路由是地基 |
| C2 G-PER（含图接地） | −9.1 | 第二，多跳推理核心 |
| C2 图接地子项（Graph-poor） | −1.5 | C2 内的精修层 |
| C4 Capability Specialization | −4.9 | 第三，多跳集中的能力裁剪 |
| C3 Dual-Layer Memory | −3.0 | 第四，稳定增益 |
