# Routing Distribution per Benchmark

Provenance: tabulated from the per-query `intent_label` field in CollaRAG run logs
(n=500 per dataset, seed=42, temperature=0, Qwen3-8B backbone).

> **Greeting is always 0** on these benchmarks: none of the six QA sets contains
> greeting utterances, so the L1 regex filter never fires and 100% of benchmark
> queries are handled by the L2 semantic classifier. This is itself a sanity check
> on the triage cascade.

## Table: per-dataset routing distribution (counts out of 500)

| Dataset    | Intent type    | greet | single_hop | multi_hop | summarization | other | Total |
|------------|----------------|------:|-----------:|----------:|--------------:|------:|------:|
| NQ         | single-hop     |     0 |        438 |        33 |            16 |    13 |   500 |
| PopQA      | single-hop     |     0 |        453 |        19 |            12 |    16 |   500 |
| HotpotQA   | multi-hop      |     0 |         22 |       464 |             4 |    10 |   500 |
| 2Wiki      | multi-hop      |     0 |         13 |       470 |             4 |    13 |   500 |
| MuSiQue    | multi-hop      |     0 |          9 |       475 |             3 |    13 |   500 |
| ASQA       | summarization  |     0 |         58 |        33 |           384 |    25 |   500 |
| **Overall**| —              |     0 |        993 |      1497 |           423 |    90 |  3000 |

## Percentages

| Dataset    | single_hop | multi_hop | summarization | other |
|------------|-----------:|----------:|--------------:|------:|
| NQ         |      87.6% |      6.6% |          3.2% |  2.6% |
| PopQA      |      90.6% |      3.8% |          2.4% |  3.2% |
| HotpotQA   |       4.4% |     92.8% |          0.8% |  2.0% |
| 2Wiki      |       2.6% |     94.0% |          0.8% |  2.6% |
| MuSiQue    |       1.8% |     95.0% |          0.6% |  2.6% |
| ASQA       |      11.6% |      6.6% |         76.8% |  5.0% |
| **Overall**|      33.1% |     49.9% |         14.1% |  3.0% |
