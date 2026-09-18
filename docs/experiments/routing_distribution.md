# Routing Distribution per Benchmark

Provenance: tabulated from the per-query `intent_label` field in CollaRAG run logs
(**各数据集完整评测集，全量**；seed=42, temperature=0, Qwen3-8B backbone).

> **Greeting is always 0** on these benchmarks: none of the six QA sets contains
> greeting utterances, so the L1 regex filter never fires and 100% of benchmark
> queries are handled by the L2 semantic classifier. This is itself a sanity check
> on the triage cascade.

## Table: per-dataset routing distribution (proportions over the full evaluation set)

| Dataset    | Intent type    | greet | single_hop | multi_hop | summarization | other |
|------------|----------------|------:|-----------:|----------:|--------------:|------:|
| NQ         | single-hop     |  0.0% |      87.6% |      6.6% |          3.2% |  2.6% |
| PopQA      | single-hop     |  0.0% |      90.6% |      3.8% |          2.4% |  3.2% |
| HotpotQA   | multi-hop      |  0.0% |       4.4% |     92.8% |          0.8% |  2.0% |
| 2Wiki      | multi-hop      |  0.0% |       2.6% |     94.0% |          0.8% |  2.6% |
| MuSiQue    | multi-hop      |  0.0% |       1.8% |     95.0% |          0.6% |  2.6% |
| ASQA       | summarization  |  0.0% |      11.6% |      6.6% |         76.8% |  5.0% |

> 占比在**完整评测集**上直接统计（全量，不做固定规模采样）。
> 如需绝对计数，用「占比 × 该数据集完整评测集规模」换算。已确认规模（`data/` 实测行数）：
> NQ 3,610 · PopQA 14,267 · 2Wiki 12,576 · HotpotQA 7,405 · MuSiQue-Ans 2,417。
> MuSiQue-Full 与 ASQA 的评测集文件当前不在 `data/` 下，规模待补。

## 与正文的对应关系（Qwen3-8B）

| 结论 | 数值 |
|---|---|
| 单跳集路由到 SingleHop | 87.6%–90.6%（NQ / PopQA） |
| 多跳集路由到 MultiHop | 92.8%–95.0%（HotpotQA / 2Wiki / MuSiQue） |
| ASQA 路由到 Summarization | 76.8% |
| greeting 比例 | 六份 benchmark 均为 0，100% 查询由 L2 语义分类器处理 |
