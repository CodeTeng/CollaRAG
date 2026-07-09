# 创新点一：AdaGraph —— 知识构建阶段的双重增强

> 本文档基于 `docs/plan.md` §核心功能 第 2 条与 `src/chimera_rag/plugins/adagraph/` 实际实装整理。
> **作用阶段**：Ingestion（知识构建）
> **代号对应**（plan.md §产品概述）：`AdaGraph` = 原设计材料中的 `DEKER`
> **启用方式**：`configs/*.yaml` 中 `plugins.adagraph.enabled: true`，并把 `ingestion.chunker/extractor/pruner.active` 切到 `adagraph.*`

---

## 1. 插件定位

AdaGraph 占据 `IngestionPipeline` 的**三个插槽**，分别对应 plan.md §核心功能 第 2 条中列出的三个子模块：

| 子模块 | 插槽 | 注册名 | 代码文件 | 实装状态（plan.md §核心功能 第 2 条） |
|---|---|---|---|---|
| 动态粒度分块器 | `chunker` | `adagraph.dynamic` | `plugins/adagraph/dynamic_chunker.py` | **自适应公式已实装** |
| 分层四级三元组提取器 | `extractor` | `adagraph.layered` | `plugins/adagraph/layered_extractor.py` | **L1 已实装；L2-L4 TODO** |
| 双重冗余剪枝器 | `pruner` | `adagraph.dual_redundancy` | `plugins/adagraph/dual_pruner.py` | **相似性已实装；传递性 TODO** |

开关方式（摘自 `configs/full.yaml`）：

```yaml
ingestion:
  chunker:
    active: adagraph.dynamic
  extractor:
    active: adagraph.layered
  pruner:
    active: adagraph.dual_redundancy

plugins:
  adagraph:
    enabled: true
    dynamic_chunker:   { min_chunk_size: 120, max_chunk_size: 500, base_chunk_size: 300 }
    layered_extractor: { layers_enabled: [L1] }
    dual_pruner:       { similarity_threshold: 0.85, enable_transitive: false }
```

关闭时 `enabled: false` 并把三个 `active` 改回 `defaults.fixed / defaults.simple_llm / defaults.noop` 即可，无需改代码——这是 plan.md §实施策略"主路径"明确要求的契约。

---

## 2. 子模块一：DynamicChunker（动态粒度分块器）

### 2.1 要解决什么问题

裸 KG-RAG 基线使用的 `FixedSizeChunker` 采用固定窗口（token 固定窗口 + overlap，plan.md §目录结构 第 195 行）。它对所有文本施加同一粒度，于是：

- 实体密集、语义复杂的段落被切得太粗，一个 chunk 挤进太多事实 → 下游抽取时三元组彼此干扰；
- 叙事稀疏、信息密度低的段落被切得太细 → 增加 embedding / LLM 调用成本，对召回无增益。

AdaGraph 的解法是让**块大小成为文本自身属性的函数**，无需人工标注。

### 2.2 自适应公式

`plugins/adagraph/dynamic_chunker.py::compute_adaptive_chunk_size` 实装的是 plan.md §目录结构 第 234 行所要求的公式。给定文本 $t$：

$$
\text{score}(t) = w_c \cdot \text{complexity}(t) + w_d \cdot \text{density}(t), \quad w_c + w_d = 1
$$

$$
\text{size}(t) = \text{clip}_{[\min,\max]}\Big(\text{base} - (\text{score}(t) - 0.5)\cdot(\max - \min)\Big)
$$

三条直觉：

- `score = 0.5` → `size = base`，与 `FixedSizeChunker` 等价；
- `score → 1`（高复杂 / 高密度）→ `size → min`，切得更细；
- `score → 0`（低复杂 / 低密度）→ `size → max`，切得更粗。

默认权重 `w_c = w_d = 0.5`，默认 `(min, base, max) = (100, 300, 500)`。公式被刻意写成一个纯函数暴露在模块顶层，便于 `tests/test_adagraph_dynamic_chunker.py`（plan.md §目录结构 第 346 行）在 score=0/0.5/1 三个边界上做独立单测。

### 2.3 complexity 与 density 的零依赖估计

plan.md §目录结构 第 234 行允许这两个指标"stub 为 0.5"或用真实方法实装。当前代码选择了**零依赖的启发式**：

**complexity**（`_estimate_complexity`）— 三项加权：
- 词汇多样性（type-token ratio）× 0.5
- 句长变异系数（$\sigma/\mu$）× 0.3
- 平均词长归一化 `(avg_len-3)/5` 截断到 `[0,1]` × 0.2

**density**（`_estimate_entity_density`）— 首字母大写 token 比例，扣除句首大写以消除假阳性。

两者都截断到 `[0, 1]`，符合公式前提。生产部署时若装了 `extras[spacy]`，可以把这两个函数替换为 NER / 依存分析版本——`DynamicChunker.chunk()` 的公共接口完全不变。

### 2.4 句子级装箱

`chunk()` 在拿到自适应 size 后，按 `[.!?]` 切句，然后顺序 pack 句子到 buffer，一旦再加超过 size 就落一个 Chunk。这比字符硬切多一步保护：**不会割裂句子**，从而给后续 L4 跨句抽取保留了可用的句边界。

每个 Chunk 的 `metadata` 会写回本次决策的四个数字：

```python
{"adaptive_chunk_size": int,
 "complexity": float,
 "density": float,
 "score": float}
```

这个"审计信息"同时服务于 Web UI 的追踪面板与离线消融分析。

### 2.5 参数兼容性

构造函数除了 `min_chunk_size / max_chunk_size / base_chunk_size / complexity_weight / density_weight`，还额外接受 `chunk_size`——当 yaml 写的是 `defaults.fixed` 风格的 `chunk_size: 300` 时，`DynamicChunker` 仍能被构造（等价于 `base_chunk_size=300`）。配合 `ChimeraRAG._filter_params` 的签名过滤，`fixed ↔ dynamic` 热切换不会因为参数多/少而报 `TypeError`。

---

## 3. 子模块二：LayeredTripleExtractor（分层四级三元组提取器）

### 3.1 为什么要分层

plan.md §核心功能 第 2 条明确把抽取拆成 4 层：

| 层次 | 目标 | 当前实装 |
|---|---|---|
| **L1 — explicit** | 文本字面写出的 (s, p, o) | ✅ 复用 `defaults/extractor.py::SimpleLLMExtractor` 的单 prompt 抽取 + JSON 解析 |
| **L2 — implicit** | 共指、因果、时序等隐式关系 | 🚧 接口存在，返回空列表 + debug 日志（plan.md §目录结构 第 235 行所标 TODO） |
| **L3 — schema** | 受实体类型 schema 约束的补全关系 | 🚧 接口存在，返回空列表 + debug 日志 |
| **L4 — cross-sentence** | 需要跨句融合多处证据的事实 | 🚧 接口存在，返回空列表 + debug 日志 |

每个三元组的 `Triple.layer` 字段（`core/types.py` 中的 `Literal["L1","L2","L3","L4"]`）会被写回，供下游按层筛选。

### 3.2 分层开关

```python
LayeredTripleExtractor(
    llm=llm,
    layers_enabled=["L1"],       # 默认；也可 ["L1","L2"] 等任意组合
    max_triples_per_chunk=30,    # 单个 chunk 的三元组数上限，防止异常输入爆图
)
```

`layers_enabled` 校验每个元素必须属于 `{L1,L2,L3,L4}`，非法值直接抛 `ValueError`——符合 plan.md §实施注意事项关于"Pydantic 严格校验"的一致风格。

### 3.3 L1 的复用意义

L1 直接委托给 `SimpleLLMExtractor`（`extractor.py::_l1 = SimpleLLMExtractor(...)`）。这意味着**当 `layers_enabled=["L1"]` 时，LayeredTripleExtractor 与 vanilla `SimpleLLMExtractor` 的抽取语义完全等价**——这是两个创新点可做公平消融对比的前提：`adagraph_only` vs `vanilla` 之间的差异完全归因于 DynamicChunker + DualRedundancyPruner 的效果，不会被 L1 与 baseline 抽取器的 prompt 差异污染。

### 3.4 未来 L2–L4 落地的切入点

`_extract_l2 / _extract_l3 / _extract_l4` 三个私有方法已经存在于类中，TDD 循环要补齐哪层只需：

1. 在 `tests/test_adagraph_layered_extractor.py` 中先写该层的失败测试；
2. 把对应的私有方法从"返回空列表 + debug 日志"改成真正的 LLM prompt + 解析；
3. 不必改 `extract()` 的循环逻辑，也不必改任何 yaml 外部字段。

---

## 4. 子模块三：DualRedundancyPruner（双重冗余剪枝器）

### 4.1 两种冗余的定义

plan.md §核心功能 第 2 条把冗余分成两类：

| 冗余类型 | 典型例子 | 当前策略 |
|---|---|---|
| **相似性冗余** | `(Einstein, born_in, Ulm)` 与 `(Albert Einstein, birthplace, Ulm, Germany)` 指代同一事实 | 三元组 token Jaccard ≥ 阈值则保留先到者 |
| **传递性冗余** | 已有 `A→B` 与 `B→C`，再保留可推导的 `A→C` 属重复 | 闭包推理后移除可推导边（TODO） |

### 4.2 相似性剪枝（已实装）

实装选择的是 **token 级 Jaccard 相似度**，而非 embedding 余弦：

$$
\text{sim}(t_1, t_2) = \frac{|S(t_1) \cap S(t_2)|}{|S(t_1) \cup S(t_2)|}
$$

其中 $S(t)$ 是对 `f"{subject} {predicate} {object}"` 做小写 + 正则 `\w+` 分词后的 token 集合。默认阈值 `0.85`。

选择 Jaccard 而非 embedding 的理由（同 plan.md §技术栈提到的"零依赖 / MVP"取向）：

- 不需要额外的 embedding 模型就能跑；
- 确定性强，单测稳定；
- 真要升级到 embedding cosine，只需替换 `_triple_signature` 一个函数，`prune()` 的流程不变。

流程是"保留先到者"的线性扫描：

```python
for t in triples:
    sig = _triple_signature(t)
    if any(_jaccard(sig, s) >= threshold for s in kept_sigs):
        continue
    kept.append(t); kept_sigs.append(sig)
```

这个策略保证输出**顺序稳定**、**跨运行可复现**，对 pytest 友好。

### 4.3 传递性剪枝（保留接口）

`enable_transitive=True` 走 `_prune_transitive` 钩子，当前版本 `return triples` 不做事但不报错。plan.md §目录结构 第 236 行标为 TODO 的主要原因是：要落地传递性剪枝必须先有**谓词传递性表**（例如 `ancestor_of / part_of` 是传递的，`wrote / met` 不是），再在 `NetworkXGraphStore` 上做 BFS 闭包。这是一个独立的 TDD 循环，当前代码为它预留了入口。

### 4.4 日志与可观测性

每次 `prune()` 会写一条 INFO 级日志：

```
dual_pruner: 2374 -> 1851 triples after similarity pass
```

这些数字可被 Web UI 的 `/api/ingest` 统计与 `/api/graph/stats` 页面消费，也会被 `main.py ingest` 子命令以 JSON 形式打印出来（`chimera_rag/cli.py::cmd_ingest`）。

---

## 5. 在 IngestionPipeline 中的接入

源码：`src/chimera_rag/core/pipeline.py::IngestionPipeline.run()`

```
documents
  → self.chunker.chunk()                         ← DynamicChunker
  → self.state.chunk_lookup[ch.chunk_id] = ch    （chunk 注册到共享状态）
  → self.embedder.encode([...])                  （批量编码）
  → self.state.vector_store.add([...])
  → for ch in chunks: self.extractor.extract()   ← LayeredTripleExtractor
  → self.pruner.prune(raw_triples)               ← DualRedundancyPruner
  → for t: self.state.graph_store.add_triple(t)

返回 {"documents": N, "chunks": M, "triples_raw": R, "triples_pruned": P}
```

**Pipeline 不知道 DynamicChunker / LayeredTripleExtractor / DualRedundancyPruner 的存在**——它只认三个 ABC（`BaseChunker / BaseTripleExtractor / BasePruner`）。这正是 plan.md §实施注意事项 第 1 条"依赖方向强约束"的具体体现。

---

## 6. 对应测试

plan.md §目录结构 第 346–348 行规划的三个测试文件均已落地：

| 测试文件 | 主要覆盖点 |
|---|---|
| `tests/test_adagraph_dynamic_chunker.py` | `compute_adaptive_chunk_size` 在 score = 0 / 0.5 / 1 三个边界；句子级打包不割句；短文本早退 |
| `tests/test_adagraph_layered_extractor.py` | `layers_enabled` 合法性；L2/L3/L4 的 TODO 兜底返回空；`max_triples_per_chunk` 截断 |
| `tests/test_adagraph_dual_pruner.py` | 阈值门控；保留先到者顺序；`enable_transitive=True` 不抛异常 |

端到端测试：`tests/test_pipeline_adagraph.py` 用 `configs/adagraph_only.yaml` 同等配置（Mock Provider 替换 DeepSeek）跑完 `ingest → query` 验证整条链路。

---

## 7. 与 CollaRAG 的正交性

AdaGraph **只写** `PipelineState.graph_store / vector_store / chunk_lookup`，不读取它们；更不会触碰查询阶段的任何插槽。这带来了两个好处：

1. **消融实验干净**：可以把 CollaRAG 关掉只比较"图好不好"的贡献，或把 AdaGraph 关掉只比较"查得聪不聪明"的贡献；
2. **协同效应可度量**：full 配置相对 `adagraph_only + colla_rag_only` 的增益超出两者加和的部分，就是真正的协同——这是答辩时"1+1>2"的定量依据。

关于 CollaRAG 侧的实装细节，见 `docs/innovation-2/colla-rag.md`。
