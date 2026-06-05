# 实验指南：从数据集到评测指标

本文档说明如何在 Chimera-RAG 上跑完整的实验流程：建图 → 推理 → 评测。

---

## 核心概念：三阶段工作流

```
 数据文件                    storage/                     output/
 (JSON/JSONL)                (图谱 + 向量)                 (预测 + 报告)

 ┌─────────┐   ingest    ┌──────────────┐   infer    ┌──────────────┐
 │ 语料段落 │ ─────────►  │ 知识图谱     │ ─────────► │ predictions  │
 │ QA 题目  │   建库      │ 向量索引     │   答题     │ .jsonl       │
 └─────────┘   (慢,一次)  └──────────────┘   (中)     └──────┬───────┘
                                                             │  eval
                                                             ▼
                                                      ┌──────────────┐
                                                      │ report.md    │
                                                      │ EM/F1/ROUGE  │
                                                      └──────────────┘
```

| 阶段 | 命令 | 做什么 | 调 LLM 吗 | 产物 |
|---|---|---|---|---|
| **ingest** | `main.py ingest` | 文本 → chunk → embedding → LLM 抽三元组 → 图谱 | 是（每 chunk 一次） | `storage/` 下的图谱 + 向量 |
| **infer** | `main.py infer` | 逐题：意图分类 → 检索 → 生成答案 | 是（每题至少一次） | `output/` 下的 predictions.jsonl |
| **eval** | `eval.py` | 对比 prediction vs reference，算 EM/F1/ROUGE-L | 否（纯计算） | report.md/.csv/.json |

---

## 为什么 mock_wiki 不用手动 ingest，真实数据集要？

`main.py infer` 内部有一段**自动建图**逻辑（`cli.py:262-299`）：

```python
# 如果图谱尚未注入（无持久化或 --rebuild），触发一次 ingest 并落盘
if crag.stats()["chunks"] == 0:
    docs = loader.load_documents()      # ← 全量加载，无 limit！
    crag.ingest(docs)
```

| 场景 | `load_documents()` 返回多少 | 后果 |
|---|---|---|
| `mock_wiki` | 25 篇小文档 | 自动 ingest 几秒搞定，完全无感 |
| `hotpotqa` | **66,581** 篇段落 | 自动 ingest 调 LLM 几万次，几小时起步 |
| `musique` | **48,315** 篇段落 | 同上 |

关键问题：**自动 ingest 不支持 `--limit`**，一旦触发就是全量。

所以真实数据集的正确做法是：
1. **先手动** `ingest --limit 500`（只用 500 行 QA 对应的语料建图，几分钟搞定）
2. **再** `infer --limit 500`（复用已建好的图谱答题，不再触发自动 ingest）

---

## 数据集格式

三个多跳 QA 数据集（HotpotQA / 2Wiki / MuSiQue）的 JSON/JSONL 文件是 **自包含的**——每行既有 QA 也有语料：

```json
{
  "id": "...",
  "question": "Who is the spouse of the Green performer?",
  "answer":   "Miquette Giraudy",
  "paragraphs": [
    {"title": "Grant's First Stand", "paragraph_text": "Grant's First Stand is the debut album..."},
    {"title": "...", "paragraph_text": "..."},
    ... (每行约 20 段)
  ]
}
```

- `load_examples()` 抽 question + answer → 给 infer 用
- `load_documents()` 抽 paragraphs → 给 ingest 用
- 配置里只需 `path:` 一个字段

| 数据集 | 配置方式 | 说明 |
|---|---|---|
| `hotpotqa` / `two_wiki` / `musique` | `path:` 单字段 | QA 和语料在同一文件 |
| `mock_wiki`（合成数据集） | `corpus_path:` + `qa_path:` | 语料和 QA 分开的文件 |

---

## 实验操作手册

### 前置准备

```bash
# 1) 安装依赖
uv sync --all-extras --dev

# 2) 配置 .env（本地 ollama 场景）
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:8b

# 3) 启动 ollama 4 路并发（建图加速）
bash scripts/ollama_restart_parallel.sh

# 4) 确认模型已拉取
ollama pull qwen3:8b
```

### 方式 A：完整 RAG 实验（建图 + 意图检索 + 评测）

**推荐先用 musique 跑通一遍**（最小数据集，最快验证全流程）：

```bash
# Step 1: 建图（前 500 条 QA 附带的全部 paragraph → 约数千~万个 Document → 图谱）
uv run python main.py ingest \
    --config configs/ollama_intent_only.yaml \
    --dataset musique --limit 500 --rebuild

# Step 2: 推理（500 题问答，复用已建好的图谱）
uv run python main.py infer \
    --config configs/ollama_intent_only.yaml \
    --dataset musique --limit 500

# Step 3: 评测
uv run python eval.py \
    --predictions output/musique/ollama_intent_only \
    --metrics em,f1,rouge_l
```

**跑通后扩展到 3 个数据集**：

```bash
for ds in hotpotqa two_wiki musique; do
  uv run python main.py ingest --config configs/ollama_intent_only.yaml --dataset $ds --limit 500 --rebuild
  uv run python main.py infer  --config configs/ollama_intent_only.yaml --dataset $ds --limit 500
  uv run python eval.py        --predictions output/$ds/ollama_intent_only --metrics em,f1,rouge_l
done
```

**横向对比**：

```bash
uv run python eval.py \
  --predictions output/hotpotqa/ollama_intent_only \
                output/two_wiki/ollama_intent_only \
                output/musique/ollama_intent_only \
  --out output/qwen3_8b_compare --metrics em,f1,rouge_l
```

### 方式 B：LLM-only baseline（无检索，纯模型作答）

```bash
for ds in hotpotqa two_wiki musique; do
  uv run python scripts/run_llm_only.py \
    --dataset $ds --provider ollama --model qwen3:8b \
    --limit 500 --out output/llm_only/qwen3_8b/$ds
  uv run python eval.py --predictions output/llm_only/qwen3_8b/$ds --metrics em,f1,rouge_l
done
```

### 方式 C：DeepSeek 云端实验

```bash
# .env 改为：
# LLM_API_KEY=sk-xxxxxxxx
# LLM_BASE_URL=https://api.deepseek.com
# LLM_MODEL=deepseek-chat

for ds in hotpotqa two_wiki musique; do
  uv run python main.py ingest --config configs/deepseek_intent_only.yaml --dataset $ds --limit 500 --rebuild
  uv run python main.py infer  --config configs/deepseek_intent_only.yaml --dataset $ds --limit 500
  uv run python eval.py        --predictions output/$ds/deepseek_intent_only --metrics em,f1,rouge_l
done
```

---

## 参数速查

| 参数 | 所属命令 | 含义 |
|---|---|---|
| `--config` | ingest / infer | 指定配置文件（决定用哪个 LLM、哪些 plugin） |
| `--dataset` | ingest / infer | 数据集名（须在 yaml 的 `datasets:` 节注册） |
| `--limit N` | ingest / infer | ingest 读取前 N 条 QA 条目，这 N 条所附带的全部 paragraph 都会作为语料建图（500 条约产生 3,000~10,000 个 Document）；infer 限制问答样本数。**两边都加 500 才是对齐的 500 题评测** |
| `--rebuild` | ingest / infer | 忽略已有图谱，强制重建。仅首次或改了配置时需要 |
| `--predictions` | eval | 指向 predictions.jsonl 所在目录。多个目录触发横向对比模式 |
| `--metrics` | eval | 逗号分隔的指标名：em,f1,rouge_l |

## 产物路径约定

```
storage/<config_stem>/              ← ingest 产物（图谱 + 向量，可复用）
  graph.gpickle
  graph.gpickle.chunks.pkl
  graph.gpickle.meta.json
  vectors.faiss
  vectors.ids.pkl

output/<dataset>/<config_stem>/     ← infer 产物（预测 + 元信息）
  predictions.jsonl                 ← 每行 {qid, question, reference, prediction, latency_s, ...}
  meta.json                         ← 运行元信息（config、耗时、token 消耗）
  report.md                         ← eval 生成的报告
  report.csv
  report.json
```

---

## 参考实测数据

### Qwen3-8B LLM-only（2026-06-04，500 题，seed=42）

| 数据集 | EM | F1 | ROUGE-L | avg latency |
|---|---|---|---|---|
| HotpotQA | 0.222 | 0.306 | 0.306 | 20.4s |
| 2Wiki | 0.270 | 0.300 | 0.300 | 22.0s |
| MuSiQue-Ans | 0.050 | 0.134 | 0.133 | 25.9s |

> 完整方法对比表见 `paper/relink_qwen3_8b_estimate.md`。
