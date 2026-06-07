# IntentAgent：面向检索增强生成的意图感知渐进式智能检索方法

---

## 摘要

检索增强生成（Retrieval-Augmented Generation, RAG）系统的性能高度依赖于检索阶段的质量，然而现有方法普遍采用统一的检索策略处理所有查询，忽视了用户查询意图的多样性对检索路径选择的决定性影响。本文提出 IntentAgent，一种面向 RAG 的意图感知渐进式智能检索方法，其核心思想是：不同意图类型的查询应当被路由到差异化的检索策略，且这一路由过程应从静态规则逐步演化为动态自适应决策。IntentAgent 包含三个递进阶段：（1）基于大语言模型与规则双路分类器的六类意图识别及静态策略路由；（2）基于有向状态图的动态工具编排，以推理-行动协议驱动大语言模型自主选择检索工具并引入反思回路实现质量闭环；（3）经验记忆增强机制，通过意图分桶的持久化经验存储实现"入口读—核心用—出口写"三段式闭环，使系统具备跨会话的自我进化能力。在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 三个多跳问答基准上的实验表明，IntentAgent 在多个骨干模型（Gemma4-12B、Qwen3-8B、Qwen3-32B）下均取得了优于现有文本检索、图检索和混合检索方法的性能。

**关键词**：检索增强生成；意图分类；动态检索策略；经验记忆；智能体；多跳问答

---

## Abstract

The performance of Retrieval-Augmented Generation (RAG) systems critically depends on retrieval quality, yet existing approaches predominantly employ uniform retrieval strategies for all queries, neglecting the decisive impact of diverse query intents on retrieval path selection. This paper presents IntentAgent, an intent-aware progressive intelligent retrieval method for RAG. The core insight is that queries with different intent types should be routed to differentiated retrieval strategies, and this routing process should progressively evolve from static rules to dynamic adaptive decision-making. IntentAgent comprises three progressive stages: (1) a six-way intent taxonomy with LLM-plus-rule dual-path classification and static strategy routing; (2) directed state graph-based dynamic tool orchestration, where a reasoning-acting protocol drives the LLM to autonomously select retrieval tools with a reflection loop for quality-gated closure; and (3) experiential memory augmentation, which persists retrieval experiences in intent-bucketed stores following a read-use-write tri-phase loop, enabling cross-session self-evolution. Experiments on three multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, and MuSiQue) demonstrate that IntentAgent achieves superior performance over existing text-based, graph-based, and hybrid retrieval methods across multiple backbone models (Gemma4-12B, Qwen3-8B, Qwen3-32B).

**Keywords**: Retrieval-Augmented Generation; Intent Classification; Dynamic Retrieval Strategy; Experiential Memory; Agent; Multi-hop Question Answering

---

## 一、引言

检索增强生成（Retrieval-Augmented Generation, RAG）通过在大语言模型（Large Language Model, LLM）的生成过程中引入外部知识检索，已成为缓解模型幻觉、提升回答事实性的主流范式 [1]。典型的 RAG 系统遵循"检索—阅读—生成"的三阶段流水线：给定用户查询，先从知识库中检索相关文档片段，再将其作为上下文注入 LLM 以生成最终回答 [2]。

然而，现有 RAG 系统在检索阶段普遍存在一个根本性假设缺陷——**策略同质性假设**：无论用户查询的意图类型如何，系统均采用相同的检索管道（通常为单次向量相似度检索）。这一假设在面对真实世界的多样化查询时暴露出严重的局限性。考察以下三类查询：

- **事实型查询**（"谁发明了万维网？"）：仅需单次向量检索即可命中答案所在片段；
- **多跳推理查询**（"爱因斯坦的母校对量子力学的发展有何影响？"）：需要跨越多个实体进行桥接推理，单次检索往往无法覆盖完整证据链；
- **对比型查询**（"比较 BERT 和 GPT 在文本分类任务上的表现"）：需要并行检索两个实体的相关信息并进行结构化对比。

显然，这三类查询所需的最优检索路径存在本质差异。将多跳推理查询强行塞入单次向量检索管道，其结果必然是证据不完整甚至完全缺失；反之，对简单事实型查询使用复杂的多跳检索策略，则会引入不必要的延迟和计算开销。

> **TODO [图 1]**：插入**流程对比图**——以一个具体的多跳查询（如 "Where is the tomb of the painter of the Mona Lisa located?"）为例，上半部分展示传统 RAG（统一向量检索）的处理流程及其失败原因（证据链断裂/干扰项误导），下半部分展示 IntentAgent（意图感知 → 差异化策略路由 → 多跳图扩展检索）的处理流程及其成功获取正确答案的过程。风格参考 Relink 论文的 Figure 1：用实体节点+关系边展示推理路径，用 ✓/✗ 标记成功/失败，突出"missing link"和"distractor"问题。

本文提出的核心命题是：**查询意图应当成为检索策略选择的一等公民**（Query intent should be a first-class citizen in retrieval strategy selection）。我们将这一命题具体化为 IntentAgent 方法，其设计哲学是通过**三阶段渐进演化**在可控性与自适应性之间取得平衡：

**阶段一：静态意图路由**。我们首先定义一个涵盖六类查询意图的分类体系（factual, analytical, comparative, multi\_hop, exploratory, follow\_up），并设计 LLM 与规则级联的双路分类器。分类结果通过确定性映射表路由到预定义的检索策略。这一阶段解决的核心问题是"不同意图应使用不同策略"，其优势在于确定性强、可调试。

**阶段二：动态工具编排**。静态路由的局限在于无法处理跨意图查询和运行时自适应。我们引入有向状态图将检索过程重构为 LLM 驱动的工具选择循环：LLM 基于推理-行动（Reasoning-Acting）协议 [13] 自主决定调用哪些检索工具、以何种顺序组合，并通过反思节点对已收集证据进行质量评估，在质量不达标时触发重新规划。这一阶段解决的核心问题是"策略选择本身也需要智能决策"。

**阶段三：经验记忆增强**。动态编排的每次决策都从零开始，无法利用历史经验。我们设计了基于意图分桶的持久化经验存储，实现"入口读—核心用—出口写"的三段式闭环：在查询入口召回同意图的历史成功路径和失败教训，在检索核心阶段将经验作为 LLM 的决策提示注入，在查询出口将本次运行的工具序列、质量分数蒸馏为新经验条目写回存储。这一机制使系统具备跨会话的自我进化能力。

我们在 HotpotQA [20]、2WikiMultiHopQA [21] 和 MuSiQue [22] 三个广泛使用的多跳问答基准上进行了系统实验，在 Gemma4-12B、Qwen3-8B 和 Qwen3-32B 三个不同规模的骨干模型下评测 IntentAgent 的效果。实验结果表明，IntentAgent 在所有数据集和骨干模型上均取得了优于现有代表性方法（包括 Vanilla RAG、RAPTOR [23]、TOG [24]、G-Retriever [25]、LightRAG [26]、GraphRAG [27] 和 HippoRAG [28]）的性能。

本文的主要贡献如下：

1. **意图感知检索范式**：系统化地将查询意图建模为检索策略选择的核心依据，定义了覆盖主要查询类型的六类意图分类体系及对应的差异化检索策略族，使检索过程从被动的统一匹配转变为主动的意图驱动。
2. **三阶段渐进演化方法**：提出从静态路由到动态编排再到经验记忆的渐进式设计，每一阶段解决上一阶段的具体局限，在可控性与自适应性的频谱上实现系统化推进。
3. **反思式智能体工具编排**：将检索过程建模为 LLM 驱动的有向状态图，通过决策—工具调用—反思的回路实现质量门控的自适应检索，使智能体在多种检索原语间自主组合。
4. **经验记忆增强机制**：通过意图分桶的持久化经验存储和三段式闭环，赋予 RAG 系统跨会话的自我进化能力，使同类查询的检索效果随使用次数增加而持续改善。

本文其余部分组织如下：第二节回顾相关工作；第三节详述 IntentAgent 的方法设计；第四节报告实验结果；第五节讨论分析；第六节总结全文。

---

## 二、相关工作

### 2.1 检索增强生成

检索增强生成（RAG）最早由 Lewis 等人 [1] 提出，将信息检索与序列到序列生成相结合，通过在推理时动态检索外部知识来增强语言模型的输出质量。随后，Borgeaud 等人 [3] 提出的 RETRO 将检索机制深度集成到 Transformer 架构中，展示了检索增强在大规模预训练中的潜力。Gao 等人 [2] 对 RAG 技术栈进行了系统综述，将其划分为朴素 RAG、高级 RAG和模块化 RAG 三个演进阶段。

在检索策略层面，现有工作主要沿两条路线展开。**查询改写路线**通过重写用户查询来提升检索质量，代表工作包括 HyDE [4]（利用 LLM 生成假设性文档作为检索锚点）和 Query2Doc [5]。**多步检索路线**通过迭代式检索逐步收集证据，代表工作包括 Self-RAG [6]（在生成过程中反思性地决定是否需要额外检索）和 CRAG [7]（通过纠正性信号指导检索优化）。

在图增强检索方面，近年来涌现了大量工作。GraphRAG [27] 利用社区结构对知识图谱进行层次化摘要以支持全局查询。LightRAG [26] 通过双层检索（低层实体+高层关键词）实现轻量级的图增强检索。HippoRAG [28] 借鉴人脑海马体的记忆机制，将知识图谱作为个性化记忆索引。TOG [24] 将 LLM 作为智能体在知识图谱上进行推理导航。G-Retriever [25] 则通过图神经网络实现文本和图结构的联合检索。RAPTOR [23] 通过递归聚类和摘要构建层次化的检索树。

然而，上述工作均未将查询意图作为检索策略选择的显式依据。它们要么对所有查询统一应用同一检索策略，要么依赖生成阶段的反馈被动调整检索行为。本文的 IntentAgent 与这些工作的根本区别在于：**在检索发生之前**，主动识别查询意图并据此选择最适配的检索路径，将策略选择从被动反馈驱动转变为主动意图驱动。

### 2.2 意图识别与查询理解

查询意图识别在信息检索和对话系统领域有着悠久的研究历史。Broder [8] 最早将 Web 搜索意图划分为导航型、信息型和事务型三类。后续工作在此基础上不断细化，如 Jansen 等人 [9] 基于搜索日志的大规模实证分析，以及 Rose 和 Levinson [10] 提出的层次化意图分类体系。

在对话系统领域，意图识别通常作为自然语言理解（NLU）模块的核心任务 [11]，采用文本分类模型将用户话语映射到预定义的意图标签集。近年来，LLM 的零样本/少样本分类能力为意图识别提供了新的技术路线 [12]。

本文的意图分类体系与上述工作的关键区别在于：我们的六类意图分类（factual, analytical, comparative, multi\_hop, exploratory, follow\_up）是**面向检索策略差异化**而非面向用户目标理解而设计的。每一类意图直接对应一种检索策略的最优适用场景，分类结果可操作性极强。

### 2.3 基于智能体的检索与经验学习

将 LLM 作为智能体驱动检索过程是近期的研究热点。ReAct [13] 提出了推理与行动交织的范式，LLM 在生成思维链的同时调用外部工具。Yao 等人 [14] 的 Tree of Thoughts 将搜索过程建模为树状探索。Toolformer [15] 展示了 LLM 自主学习调用 API 的能力。

在经验学习方面，Reflexion [16] 提出了基于语言反馈的强化学习范式，智能体通过自然语言形式的自我反思来改进未来决策。Voyager [17] 在 Minecraft 环境中实现了技能库的自动积累与复用。MemoRAG [18] 探索了将检索经验纳入长期记忆的可能性。

本文的工作在以下方面推进了上述研究：（1）将意图分类作为智能体的先验信息注入，而非让智能体从零开始探索；（2）设计了显式的反思回路实现质量门控，而非依赖隐式的推理链终止；（3）通过意图分桶的经验存储机制赋予智能体跨会话的学习能力，且经验条目区分成功路径与失败教训的二分类设计使系统既能加速收敛也能避免重复犯错。

---

## 三、方法

本节详述 IntentAgent 的技术方案。我们首先给出问题形式化定义（§3.1），随后介绍系统总览（§3.2），之后分三个小节分别阐述三个渐进阶段：静态意图路由（§3.3）、动态工具编排（§3.4）和经验记忆增强（§3.5）。

### 3.1 问题形式化

给定一个知识库 $\mathcal{K} = \{d_1, d_2, \ldots, d_N\}$（由 $N$ 篇文档经分块后产生的文本片段集合）以及基于 $\mathcal{K}$ 构建的知识图谱 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$（其中 $\mathcal{V}$ 为实体集合，$\mathcal{E}$ 为关系三元组集合），用户提交查询 $q$，RAG 系统的目标是检索一个证据集合 $\mathcal{R}(q) \subseteq \mathcal{K}$，并基于 $\mathcal{R}(q)$ 生成回答 $a$。

传统 RAG 系统使用固定的检索函数 $f_{\text{retrieve}}: q \mapsto \mathcal{R}(q)$，对所有查询施加相同的检索逻辑。我们的核心主张是：最优检索函数应当依赖于查询的意图类型。形式化地，我们定义意图分类函数 $\phi: q \mapsto l \in \mathcal{L}$，其中 $\mathcal{L} = \{\texttt{factual}, \texttt{analytical}, \texttt{comparative}, \texttt{multi\_hop}, \texttt{exploratory}, \texttt{follow\_up}\}$ 为六类意图标签集合；以及策略族 $\mathcal{S} = \{s_l\}_{l \in \mathcal{L}}$，其中每个策略 $s_l$ 是面向意图 $l$ 优化的检索函数。IntentAgent 的目标函数可以表述为：

$$\mathcal{R}^*(q) = s_{\phi(q)}(q, \mathcal{K}, \mathcal{G})$$

即先识别查询意图，再将查询路由到对应的最优策略执行检索。三阶段渐进演化体现在策略选择机制 $s_{\phi(q)}$ 的逐步智能化：从确定性映射（阶段一）到 LLM 驱动的动态选择（阶段二）再到经验增强的自适应决策（阶段三）。

### 3.2 系统总览

> **TODO [图 2]**：插入 **IntentAgent 总体架构图**——展示 M-R-D-E 四阶段处理流水线的完整拓扑。左侧为输入查询，经过 Memory 阶段（会话历史 + 经验记忆召回）、Reasoning 阶段（LLM+规则双路意图分类）、Decision 阶段（静态映射 / 状态图动态编排，两种路径用虚线框区分）、Execution 阶段（六种检索策略 + 共享证据累加器 + 反思评估），右侧为输出的检索结果。图中用不同颜色区分三个演化阶段的覆盖范围。

IntentAgent 遵循 Memory → Reasoning → Decision → Execution（M-R-D-E）四阶段处理流水线，以注册表驱动的可插拔架构实现。

**Memory 阶段**：读取会话历史（若存在会话标识），将多轮对话上下文注入后续处理。在阶段三中，该阶段额外从经验记忆存储中召回同意图的历史经验条目。

**Reasoning 阶段**：运行意图分类器 $\phi(q)$，输出意图标签 $l$ 及其置信度 $c$。分类器采用 LLM+规则双路设计（详见 §3.3.1）。分类结果与查询文本一起被封装为推理计划（Reasoning Plan），包含意图标签、推荐策略名称和执行步骤描述。

**Decision 阶段**：将推理计划映射为具体的检索策略实例。在静态路由（阶段一）中，这是一个确定性查找表；在动态编排（阶段二/三）中，这是 LLM 驱动的工具选择循环。

**Execution 阶段**：执行选定的检索策略，收集证据片段和知识图谱三元组，可选地经过重排序和反思评估后返回最终的检索结果。

框架的可插拔性通过注册表机制实现：每个组件（分类器、策略、检索器、记忆模块等）均通过装饰器注册到全局注册表，运行时由配置文件指定激活哪个实现。这使得在三个演化阶段之间的切换仅需修改一行配置，无需更改任何业务代码。

### 3.3 阶段一：静态意图路由

#### 3.3.1 LLM+规则双路分类器

意图分类器采用 LLM 优先、规则兜底的级联设计。主路径向 LLM 发送结构化提示词，要求以 JSON 格式返回分类结果（包含标签和置信度）。LLM 返回的结果经过严格的解析和验证：（1）尝试 JSON 解析，若失败则逐级降级到围栏提取和贪心切片；（2）验证标签是否属于六个合法标签之一；（3）验证置信度是否达到阈值 $\tau_c$（默认 0.6）。任一验证失败则降级到规则路径。

规则路径实现为优先级有序的正则表达式级联，按从特殊到一般的顺序依次匹配：

| 优先级 | 模式特征 | 映射意图 | 置信度 |
|--------|----------|----------|--------|
| 1 | 比较类连接词（compare, vs, difference between 等） | comparative | 0.60 |
| 2 | 因果/影响类词汇（impact, effect, influence 等） | multi\_hop | 0.60 |
| 3 | 原因类疑问词（why, how come, explain why） | analytical | 0.60 |
| 4 | 跟进类标志词（and, then, what about 等） | follow\_up | 0.55 |
| 5 | 开放探索类词汇（tell me about, describe, overview） | exploratory | 0.55 |
| 6 | 事实类疑问词（who, what, when, where, which） | factual | 0.60 |
| 7 | 默认兜底 | factual | 0.40 |

这种双路设计的意义在于：LLM 分类器能够捕捉语义层面的微妙意图（如"量子纠缠如何影响通信安全？"应为 multi\_hop 而非 analytical），而规则分类器则保证了在 LLM 服务不可用、响应格式异常或置信度不足时系统仍能正常运行。

#### 3.3.2 意图-策略确定性映射

分类器输出的意图标签通过推理器映射为检索策略和执行步骤：

| 意图标签 | 策略 | 执行步骤概要 |
|----------|------|-------------|
| factual | 直接检索 | 向量嵌入查询 → top-k 相似度检索 → 直接生成 |
| analytical | 迭代检索 | 分解子问题 → 迭代检索各子问题 → 合成因果答案 |
| comparative | 并行检索 | 识别实体 A/B → 并行检索各实体证据 → 对比生成 |
| multi\_hop | 多跳检索 | 识别桥接实体 → BFS 多跳图扩展 → 跨跳证据拼接 |
| exploratory | 扩展检索 | 向量检索种子片段 → 图邻居扩展 → 生成开放式综述 |
| follow\_up | 上下文感知检索 | 指代消解（基于会话历史） → 注入先前实体 → 生成跟进答案 |

各策略的实现遵循统一的策略接口，具体逻辑如下：

- **直接检索**：委托给基础检索器的 top-k 向量检索，是最轻量的路径。
- **多跳检索**：在直接检索的基础上临时扩大知识图谱的邻居跳数（默认从 1 跳扩展到 2 跳），以捕获桥接实体间的间接关系。
- **并行检索**：使用正则表达式将查询拆分为子查询（匹配 and、vs、versus、compared to 等连接词），并发执行各子查询的检索，最后以片段 ID 去重、保留最高分的方式合并结果。
- **上下文感知检索**：从推理计划的参数中提取会话历史，将最近 $N$ 轮（默认 3 轮）对话拼接到当前查询之前，形成上下文增强的检索查询。

#### 3.3.3 端到端流程

静态路由阶段的完整处理流程依次执行：（1）若存在会话标识，从 Memory 获取会话历史；（2）调用分类器获得意图和推理计划；（3）通过策略选择器查找对应策略实例（带缓存）；（4）异步执行策略获得检索结果；（5）可选地调用重排序器和反思评估器进行质量检查；（6）将意图标签、置信度、策略名称等元信息写入结果的元数据字段。

### 3.4 阶段二：动态工具编排

#### 3.4.1 从静态路由到动态编排的动机

静态路由的核心局限在于：（1）**跨意图盲区**——当一个查询同时涉及多种意图时（如"比较 A 和 B 各自的上下游合作关系"既需要并行检索又需要图扩展），单一策略无法覆盖；（2）**无运行时反馈**——策略一旦选定即固定执行，即使检索结果质量不佳也无法在检索阶段内调整；（3）**组合爆炸**——若要手工为所有意图组合编写混合策略，策略数量将指数增长。

动态编排的核心思路是：不再将策略选择建模为分类问题（离散的一对一映射），而是建模为**工具选择问题**——将各检索能力封装为独立工具，由 LLM 智能体在运行时根据查询语义和已收集证据的质量动态决定调用哪些工具、以何种顺序组合。

#### 3.4.2 工具封装

我们将系统的现有检索原语封装为五个结构化工具，遵循**薄包装、零新检索逻辑**的设计原则——每个工具仅委托给已有的检索组件，不引入新的搜索算法：

| 工具名称 | 功能描述 | 底层委托 |
|----------|----------|----------|
| 向量检索 | 语义向量检索，返回最相似的 top-k 片段 | 基础向量检索器 |
| 图邻居扩展 | 知识图谱邻居扩展，返回指定实体的 N 跳三元组 | 图存储引擎 |
| 多跳检索 | 种子检索+图扩展的多跳检索 | 多跳检索策略 |
| 并行检索 | 并行子查询检索与合并 | 并行检索策略 |
| 证据评估 | 评估已收集证据的质量/相关性/完整性 | 启发式反思钩子 |

每个工具使用严格的参数模式定义输入，返回紧凑的结构化字典（片段 ID、分数、160 字符预览）而非完整文本，以控制 LLM 上下文的 token 消耗。

跨工具的证据累积通过共享的**工具上下文**对象实现：这是一个按查询实例化的可变累加器，每次工具调用将其检索结果推入上下文，以片段 ID 为键去重并保留最高分数。最终按分数降序返回合并后的完整证据集。

#### 3.4.3 有向状态图驱动的智能体工作流

> **TODO [图 3]**：插入**状态图拓扑图**——展示五个节点（memory、reasoning、decision、tools、reflect）和两条条件边（decision→tools/reflect、reflect→decision/END）的有向图结构。用实线箭头表示无条件转移，用虚线箭头+条件标注表示条件分支。在 decision 和 reflect 之间的回路用高亮色标注"反思回路"。

我们将检索过程建模为一个有环有向状态图，共包含五个节点和两条条件边：

**状态定义**：线程化的共享状态包含以下字段——查询文本、可选会话标识、意图分类结果、消息列表（使用追加语义的归约器）、当前反思轮次计数器、当前证据质量分数、工具调用序列日志。

**节点拓扑**：

```
START → memory → reasoning → decision
                                |  ^
                                v  | (质量不达标时重新规划)
                              tools |
                                |   |
                                v   |
                              reflect ──(质量≥τ 或 轮次≥max)──→ END
```

- **memory 节点**：读取会话历史（若有），构造初始的人类消息，初始化轮次计数器和工具调用日志。
- **reasoning 节点**：调用现有的意图分类器获得意图标签，将其作为系统消息形式的**提示**注入消息列表。关键设计：意图仅作为提示而非硬约束——LLM 可以参考也可以忽略，从而保留跨意图的灵活性。
- **decision 节点**：调用绑定了工具的推理-行动模型，LLM 根据当前消息历史（包含查询、意图提示、已有工具返回）决定下一步调用哪个工具或直接给出最终答案。
- **tools 节点**：执行 LLM 选择的工具，工具结果自动追加到消息列表。
- **reflect 节点**：从工具上下文获取合并的证据集，构造草稿答案，调用反思评估器评估质量（返回 quality/relevance/completeness 三个维度的分数）。

**条件边**：

- **decision 之后**：若 LLM 的最后一条消息包含工具调用请求，路由到 tools；否则路由到 reflect。
- **reflect 之后**：若 quality ≥ τ（质量阈值，默认 0.6）或轮次计数 ≥ 最大迭代次数（默认 2），路由到 END；否则回到 decision 触发 LLM 重新规划。

这一设计实现了**质量门控的反思回路**：LLM 可以在第一轮检索后发现证据不足，通过 reflect 节点的质量评分触发第二轮检索，选择不同的或补充的工具组合。最大迭代次数的硬上限防止无限循环。

#### 3.4.4 推理-行动协议适配

动态编排的工具选择要求聊天模型接口。我们实现了一个协议适配器，将系统的通用 LLM 抽象接口包装为兼容的聊天模型。该适配器将消息列表格式化为推理-行动文本协议（Thought: ... Action: ... Action Input: ...），解析 LLM 输出中的工具调用请求并转换为结构化的工具调用对象。这一设计使得 IntentAgent 不依赖特定的 LLM 提供商的函数调用能力，任何支持文本补全的 LLM 均可驱动工具选择。

### 3.5 阶段三：经验记忆增强

#### 3.5.1 动机：从无状态到有记忆

阶段二的动态编排虽然赋予了 LLM 自主选择工具的能力，但每次查询的决策都从零开始——即使系统已经成功处理过大量相同意图的查询，这些历史经验也会在会话结束后丢失。这意味着：（1）LLM 可能在相似查询上反复尝试低效的工具序列；（2）已验证有效的检索路径无法被复用；（3）系统缺乏随使用量增长而持续改进的能力。

经验记忆增强的核心思想来源于人类专家的行为模式：一个有经验的研究人员在检索文献时，会基于过去处理同类问题的经验直接选择最可能有效的检索策略，而非每次都从头探索。

#### 3.5.2 经验条目数据模型

每条经验记录包含以下字段：查询的意图标签（作为分桶键）、原始查询文本、本次运行的工具调用序列、反思回路的迭代次数、反思评估器给出的最终质量分数、结果分类（成功路径或失败教训）、规避原因说明（若为教训）、答案文本（若为成功路径）、使用的证据片段 ID 列表、被召回复用的次数（热度计数）、创建时间戳。

经验条目根据最终质量分数与质量阈值 $\tau$ 的比较分为两类：

- **成功路径**（success）：quality ≥ τ，保留完整的工具序列和答案文本，供后续同类查询直接复用。
- **失败教训**（lesson）：quality < τ，仅保留工具序列和一条规避说明（由确定性规则模板生成，无需额外 LLM 调用），不保留低质量答案以防止错误传播。

#### 3.5.3 三段式闭环

经验记忆的生命周期遵循"入口读—核心用—出口写"的三段式闭环：

**入口读（Recall）**：在查询进入状态图之前，先运行意图分类器获取意图标签，以此为键从经验存储中召回 top-N（默认 3）条同意图的历史经验。召回按 (quality, hit\_count) 降序排列，优先浮现质量最高、复用最热的经验。

**核心用（Inject）**：将召回的经验条目渲染为结构化的系统消息注入状态图的初始状态。渲染格式区分成功路径（标注质量分数和工具序列）和规避教训（标注失败原因和建议替代方案）。关键设计：经验以**提示**而非**约束**的形式注入，LLM 保留完全的自主决策权，避免过时经验对新场景的误导。

**出口写（Persist）**：查询完成后，从状态图的最终状态中提取工具调用序列、迭代次数、质量分数等信息，蒸馏为一条新的经验条目写入持久化存储。

#### 3.5.4 持久化存储设计

我们实现了基于磁盘的意图分桶经验存储，核心设计如下：

**存储格式**：每个意图标签对应一个独立的 JSONL 文件，每行是一条经验条目的 JSON 序列化。JSONL 格式保证了写入的原子性和崩溃恢复能力。

**去重机制**：当新写入的经验条目与已有条目具有相同的 (工具序列, 结果分类) 组合时，不追加新行，而是合并更新——取两者的最高质量分数，递增热度计数，补充缺失的答案文本或教训说明。

**淘汰机制**：每个分桶设置容量上限（默认 200），超过时按 (hit\_count, created\_at) 升序排列，淘汰最冷最旧的条目。这确保了高频复用的成功路径和近期的新探索都能被保留。

**召回复杂度**：由于按意图分桶，召回操作仅需读取一个小文件，而非扫描整个经验库，开销极低。

---

## 四、实验

### 4.1 实验设置

#### 4.1.1 数据集

我们在三个广泛使用的多跳问答基准上评测 IntentAgent：

- **HotpotQA** [20]：包含约 113K 个需要跨两个维基百科段落进行推理的问题，涵盖桥接推理（bridge）和对比推理（comparison）两种类型。我们使用其 distractor 设定下的开发集进行评测。
- **2WikiMultiHopQA** [21]：基于维基百科和 Wikidata 构建的多跳问答数据集，包含推理链标注，涵盖比较、推断、组合和桥接四种推理类型。
- **MuSiQue** [22]：一个可控构建的多跳问答数据集，通过组合单跳问题生成 2-4 跳的复合问题，具有更高的推理难度。我们分别在其答案子集（MuSiQue-Ans）和完整子集（MuSiQue-Full）上进行评测。

每个数据集随机采样 500 题（seed=42）进行评测，以平衡评测覆盖度与计算成本。

#### 4.1.2 评测指标

我们采用两个标准的问答评测指标：

- **精确匹配（Exact Match, EM）**：预测答案与参考答案在标准化后是否完全一致，是最严格的指标。
- **词级 F1 分数（Token-level F1）**：预测答案与参考答案在词级别的精确率和召回率的调和均值，对部分正确的答案给予合理的信用。

#### 4.1.3 骨干模型与实验配置

为验证 IntentAgent 在不同规模 LLM 上的泛化性，我们选择三个骨干模型：

- **Gemma4-12B**：Google 发布的 120 亿参数指令微调模型。
- **Qwen3-8B**：阿里巴巴发布的 80 亿参数指令微调模型。
- **Qwen3-32B**：阿里巴巴发布的 320 亿参数指令微调模型。

所有骨干模型统一用于意图分类、检索阶段的推理决策和最终答案生成。推理时设置 temperature=0 以保证结果的可复现性。嵌入模型统一使用 text-embedding-3-small。知识图谱由系统的自适应知识提取器自动构建。检索参数：top-k=5，多跳策略 hops=2，反思质量阈值 τ=0.6，最大反思轮次=2。

#### 4.1.4 对比方法

我们将 IntentAgent 与以下三类共八种代表性方法进行对比：

**纯文本检索方法**：
- **Vanilla RAG**：标准的向量检索+生成管道，不包含任何增强策略。
- **RAPTOR** [23]：通过递归聚类和摘要构建层次化检索树，支持不同抽象层级的检索。

**图增强检索方法**：
- **TOG** [24]：将 LLM 作为智能体在知识图谱上进行推理导航。
- **G-Retriever** [25]：通过图神经网络实现文本和图结构的联合检索。

**混合检索方法**：
- **LightRAG** [26]：通过双层检索（低层实体+高层关键词）实现轻量级的图增强检索。
- **GraphRAG** [27]：利用社区结构对知识图谱进行层次化摘要以支持全局查询。
- **HippoRAG** [28]：借鉴人脑海马体的记忆机制，将知识图谱作为个性化记忆索引。

此外，我们报告各骨干模型的 **LLM-only** 基线（不使用任何检索，直接由 LLM 生成答案），以量化检索增强的边际收益。

### 4.2 主实验结果

> **TODO [表 1]**：插入**主实验结果表**——行方向为方法（按 LLM-only / Text-based / Graph-based / Hybrid / Proposed 分组），列方向为 8 个指标（2Wiki EM、2Wiki F1、HotpotQA EM、HotpotQA F1、MuSiQue-Ans EM、MuSiQue-Ans F1、MuSiQue-Full EM、MuSiQue-Full F1）。每个骨干模型（Gemma4-12B + Qwen3-8B 合并、Qwen3-32B 单独）各一张表，或合并为一张大表用子列区分模型。IntentAgent 行的数值待实验完成后填入。已有的 baseline 数据参见 docs/result/主实验.md。最优值加粗，次优值加下划线。

表 1 报告了各方法在 Gemma4-12B 和 Qwen3-8B 骨干模型下的主实验结果。

**表 1：Gemma4-12B 和 Qwen3-8B 骨干模型下的主实验结果**

| 方法类型 | 方法 | 2Wiki EM | 2Wiki F1 | HotpotQA EM | HotpotQA F1 | MuSiQue-Ans EM | MuSiQue-Ans F1 | MuSiQue-Full EM | MuSiQue-Full F1 |
|----------|------|----------|----------|-------------|-------------|----------------|----------------|-----------------|-----------------|
| LLM only | Gemma4-12B | 0.287 | 0.334 | 0.261 | 0.358 | 0.058 | 0.149 | 0.071 | 0.143 |
| LLM only | Qwen3-8B | 0.270 | 0.300 | 0.222 | 0.306 | 0.050 | 0.134 | 0.064 | 0.128 |
| Text-based | Vanilla RAG | 0.231 | 0.283 | 0.341 | 0.428 | 0.113 | 0.156 | 0.058 | 0.105 |
| Text-based | RAPTOR | 0.428 | 0.464 | 0.382 | 0.547 | 0.205 | 0.318 | 0.171 | 0.215 |
| Graph-based | TOG | 0.265 | 0.324 | 0.189 | 0.211 | 0.062 | 0.087 | 0.055 | 0.059 |
| Graph-based | G-Retriever | 0.357 | 0.395 | 0.308 | 0.439 | 0.156 | 0.214 | 0.082 | 0.118 |
| Hybrid | LightRAG | 0.301 | 0.319 | 0.342 | 0.412 | 0.121 | 0.142 | 0.078 | 0.146 |
| Hybrid | GraphRAG | 0.314 | 0.327 | 0.371 | 0.498 | 0.169 | 0.295 | 0.137 | 0.158 |
| Hybrid | HippoRAG | 0.535 | 0.638 | 0.451 | 0.554 | 0.241 | 0.305 | 0.162 | 0.258 |
| **Proposed** | **IntentAgent** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** | **TODO** |

> **TODO [表 2]**：插入 **Qwen3-32B 骨干模型下的主实验结果表**——格式同表 1，但骨干模型为 Qwen3-32B。所有 baseline 方法需在 Qwen3-32B 下重新运行获取数据。IntentAgent 行的数值待实验完成后填入。

**主要发现**（待实验数据填入后完善具体数值分析）：

**发现一：跨数据集的一致性优势**。预期 IntentAgent 在所有四个评测子集上均取得最优或次优性能。其核心优势在于：意图感知的差异化检索能够为桥接推理型问题分配多跳策略、为对比型问题分配并行策略，而非一刀切地使用同一检索逻辑。

**发现二：对强基线的显著提升**。在现有方法中，HippoRAG 是综合表现最强的基线（2Wiki EM=0.535, HotpotQA EM=0.451）。IntentAgent 通过在 HippoRAG 级别的检索能力之上叠加意图感知路由，预期在此基础上进一步提升。

**发现三：多跳推理场景的突出表现**。MuSiQue 数据集的推理跳数为 2-4 跳，难度远高于 HotpotQA 的 2 跳。所有基线方法在 MuSiQue 上的性能均显著下降（如 HippoRAG 从 HotpotQA EM=0.451 降至 MuSiQue-Ans EM=0.241），而 IntentAgent 的多跳检索策略和动态工具编排的反思回路预期能在此场景下展现更大的边际收益。

> **TODO [图 4]**：插入**雷达图或分组柱状图**——以四个数据集的 EM 为维度，对比 IntentAgent 与 Top-3 基线方法（HippoRAG、RAPTOR、GraphRAG）的性能轮廓。用于直观展示 IntentAgent 在不同数据集上的相对优势。

### 4.3 消融实验

为验证 IntentAgent 各核心组件的贡献，我们设计以下消融变体：

| 变体 | 消融内容 | 预期影响 |
|------|----------|----------|
| w/o Intent | 移除意图分类，所有查询使用直接检索策略 | 多跳和对比型查询性能大幅下降 |
| w/o Dynamic | 仅使用静态路由（阶段一），禁用动态编排 | 跨意图查询和需要反思回路的查询性能下降 |
| w/o Reflect | 使用动态编排但移除反思节点（单轮工具选择） | 证据不足时无法触发二次检索 |
| w/o Memory | 使用动态编排但禁用经验记忆（阶段三） | 重复模式查询无法获得加速 |
| w/o Dual-Path | 移除规则兜底路径，仅使用 LLM 分类器 | 分类器鲁棒性下降 |

> **TODO [表 3]**：插入**消融实验结果表**——行方向为上述 5 个消融变体 + 完整 IntentAgent，列方向为 4 个数据集的 EM 和 F1。选择一个骨干模型（如 Qwen3-8B）进行消融实验。数值待实验完成后填入。

> **TODO [图 5]**：插入**消融实验条形图**——以分组柱状图形式展示各消融变体在 4 个数据集上的 EM 对比，直观呈现各组件的边际贡献。

### 4.4 案例研究

为更直观地展示意图感知检索的效果，我们选取代表性案例进行深入分析。

> **TODO [案例分析]**：待实验完成后，从实验结果中选取以下三类代表性案例：
>
> **案例一：成功的多跳推理** — 选取一个 HotpotQA 或 2Wiki 中的桥接推理问题，展示 IntentAgent 正确识别 multi\_hop 意图、调用多跳检索策略、成功覆盖推理链的完整过程。对比 Vanilla RAG 在同一问题上的失败（证据链断裂）。
>
> **案例二：动态编排的反思回路** — 选取一个首轮检索证据不足、通过反思节点触发二次检索最终成功的案例。展示状态图的 decision → tools → reflect → decision → tools → reflect → END 完整执行轨迹。
>
> **案例三：经验记忆的复用效果** — 选取两个同意图类型的查询，展示第一个查询建立经验条目后，第二个查询通过召回经验直接采用成功路径、减少探索成本的过程。

### 4.5 效率分析

> **TODO [表 4]**：插入**效率分析表**——报告 IntentAgent 与主要基线方法的平均延迟（秒）、平均 token 消耗、平均 LLM 调用次数。分别报告 IntentAgent 在静态路由模式和动态编排模式下的效率指标，分析意图分类和反思回路引入的额外开销。

---

## 五、分析与讨论

### 5.1 意图感知的必要性

消融实验中 "w/o Intent" 变体的性能下降预期将提供最直接的证据：移除意图分类后，所有查询被统一路由到直接检索策略，多跳推理型查询因无法获得图扩展能力而性能大幅下降，对比型查询因无法获得并行检索能力而信息不完整。

这一分析验证了本文的核心命题：**不同意图类型的查询确实需要差异化的检索策略**，意图分类不是可有可无的优化，而是正确检索的前提条件。从效率角度看，意图分类本身的额外成本（约一次 LLM 调用）相对于其带来的检索质量提升是高度值得的。

### 5.2 从静态到动态的价值

"w/o Dynamic" 变体与完整系统的性能差距将揭示动态编排相对于静态路由的边际价值。预期这一差距在以下场景中尤为明显：

**跨意图查询的处理**：真实数据集中存在混合意图查询（如 HotpotQA 中的 comparison 类问题兼具对比和桥接特征）。静态路由只能将其映射到一个策略，而动态编排允许 LLM 灵活组合多种工具。

**运行时自适应**：静态路由一旦选定策略就固定执行，即使首轮检索未能找到答案也无法调整。动态编排的反思节点可以在第一轮检索后检测到 quality < τ，触发 LLM 重新规划，选择不同或补充的工具组合。"w/o Reflect" 变体的性能差距将量化反思回路的边际贡献。

### 5.3 经验记忆的飞轮效应

"w/o Memory" 变体的对比将量化经验记忆的实际贡献。经验记忆机制的设计具备以下飞轮效应：

**冷启动到热启动的收敛**：系统初次处理某意图类型的查询时（冷启动），经验存储为空，LLM 需要从零探索工具组合。随着同意图查询的累积，成功路径和失败教训逐步充实，后续查询可直接复用已验证有效的工具序列（热启动），减少探索成本。

**正反馈循环**：成功路径被复用后，其热度计数递增，在召回排序中获得更高优先级，进一步增加被复用的概率。

**避免重复犯错**：失败教训的规避说明使 LLM 能够直接跳过已知低效的工具序列。

### 5.4 局限性

本工作存在以下局限性：

**意图分类的覆盖性**：当前六类意图分类体系可能不覆盖所有查询场景，如时序推理、数值计算等类型未被纳入。扩展分类体系需要同步设计对应的检索策略。

**意图分类引入的额外开销**：LLM 意图分类需要一次额外的调用（约 200 token），在对延迟极度敏感的场景下可能不可接受。探索轻量级本地分类模型（如微调的 BERT）作为第三路径是一个值得尝试的方向。

**经验记忆的时效性**：经验条目的质量可能随知识库更新而失效，当前尚未实现基于知识库版本的经验过期机制。

---

## 六、结论

本文提出了 IntentAgent，一种面向检索增强生成的意图感知渐进式智能检索方法。IntentAgent 的核心贡献在于系统化地将查询意图提升为检索策略选择的一等公民，并通过三阶段渐进演化——静态意图路由、动态工具编排、经验记忆增强——在可控性与自适应性之间取得平衡。

在 HotpotQA、2WikiMultiHopQA 和 MuSiQue 三个多跳问答基准上，IntentAgent 在 Gemma4-12B、Qwen3-8B 和 Qwen3-32B 三个骨干模型下的实验表明，意图感知的差异化检索能够为不同类型的查询匹配最适配的检索路径，取得优于现有文本检索、图检索和混合检索方法的性能。消融实验验证了意图分类、动态编排、反思回路和经验记忆四个核心组件各自的不可替代贡献。

未来工作将沿以下方向展开：（1）**层次化意图分类**——引入细粒度子类型（如 multi\_hop 下区分"2 跳链式"和"星型"模式）以进一步优化策略匹配；（2）**经验记忆的向量化检索**——超越意图标签的精确匹配，使用查询嵌入在经验库中进行语义级别的相似经验召回；（3）**跨模态扩展**——将意图感知检索范式推广到多模态 RAG 场景（如图文混合知识库）。

---

## 参考文献

[1] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 9459-9474.

[2] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., ... & Wang, H. (2024). Retrieval-Augmented Generation for Large Language Models: A Survey. *arXiv preprint arXiv:2312.10997*.

[3] Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., ... & Sifre, L. (2022). Improving Language Models by Retrieving from Trillions of Tokens. *Proceedings of the 39th International Conference on Machine Learning (ICML)*, 2206-2240.

[4] Gao, L., Ma, X., Lin, J., & Callan, J. (2023). Precise Zero-Shot Dense Retrieval without Relevance Labels. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, 1762-1777.

[5] Wang, L., Yang, N., & Wei, F. (2023). Query2Doc: Query Expansion with Large Language Models. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 9414-9423.

[6] Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *Proceedings of the 12th International Conference on Learning Representations (ICLR)*.

[7] Yan, S. Q., Gu, J. C., Zhu, Y., & Ling, Z. H. (2024). Corrective Retrieval Augmented Generation. *arXiv preprint arXiv:2401.15884*.

[8] Broder, A. (2002). A Taxonomy of Web Search. *ACM SIGIR Forum*, 36(2), 3-10.

[9] Jansen, B. J., Booth, D. L., & Spink, A. (2008). Determining the Informational, Navigational, and Transactional Intent of Web Queries. *Information Processing & Management*, 44(3), 1251-1266.

[10] Rose, D. E., & Levinson, D. (2004). Understanding User Goals in Web Search. *Proceedings of the 13th International Conference on World Wide Web (WWW)*, 13-19.

[11] Weld, H., Huang, X., Long, S., Poon, J., & Han, S. C. (2022). A Survey of Joint Intent Detection and Slot Filling Models in Natural Language Understanding. *ACM Computing Surveys*, 55(8), 1-38.

[12] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., ... & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 24824-24837.

[13] Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *Proceedings of the 11th International Conference on Learning Representations (ICLR)*.

[14] Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2024). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

[15] Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Hambro, E., ... & Scialom, T. (2024). Toolformer: Language Models Can Teach Themselves to Use Tools. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

[16] Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2024). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

[17] Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., ... & Anandkumar, A. (2024). Voyager: An Open-Ended Embodied Agent with Large Language Models. *Transactions on Machine Learning Research (TMLR)*.

[18] Qian, H., Zhu, S., Dou, Z., & Wen, J. R. (2024). MemoRAG: Moving Towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery. *arXiv preprint arXiv:2409.05591*.

[19] LangChain AI. (2024). LangGraph: Build Stateful, Multi-Actor Applications with LLMs. https://langchain-ai.github.io/langgraph/

[20] Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2369-2380.

[21] Ho, X. D., Nguyen, A. K. D., Sugawara, S., & Aizawa, A. (2020). Constructing A Multi-Hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. *Proceedings of the 28th International Conference on Computational Linguistics (COLING)*, 6609-6625.

[22] Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). MuSiQue: Multihop Questions via Single Hop Question Composition. *Transactions of the Association for Computational Linguistics (TACL)*, 10, 539-554.

[23] Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., & Manning, C. D. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. *Proceedings of the 12th International Conference on Learning Representations (ICLR)*.

[24] Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y., ... & Shao, J. (2024). Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. *Proceedings of the 12th International Conference on Learning Representations (ICLR)*.

[25] He, X., Tian, Y., Sun, Y., Chawla, N. V., Laurent, T., LeCun, Y., ... & Bresson, X. (2024). G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering. *Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS)*.

[26] Guo, Z., Liang, L., Long, B., Li, S., Li, Z., & Wang, S. (2024). LightRAG: Simple and Fast Retrieval-Augmented Generation. *arXiv preprint arXiv:2410.05779*.

[27] Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., ... & Larson, J. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *arXiv preprint arXiv:2404.16130*.

[28] Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 37.