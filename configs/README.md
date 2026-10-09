# Chimera-RAG 配置文件索引

本目录保留 **6 份配置**，覆盖"构建侧基线 × 创新点二"的四种组合，加上一份完整字段注释模板与一份本地 Ollama 变体。所有 Mock-based 测试配置已内联到 `tests/conftest.py`。

运行任意子命令都需要用 `--config configs/<name>.yaml` 指定其一。

## 速查表

| 文件 | 构建侧基线 | CollaRAG | 检索器 | 主要用途 |
|---|:---:|:---:|---|---|
| `default.yaml` | — | — | — | 全字段注释模板（**注意：其中 `plugins.colla_rag.enabled` 为 `true`，是模板示例值，不直接运行**） |
| `vanilla.yaml` | ❌ | ❌ | `defaults.direct` | 裸 KG-RAG 基线（消融对照组） |
| `adagraph_only.yaml` | ✅ | ❌ | `defaults.direct` | 单独衡量**构建侧前置基线**的增益（**不是创新点一**） |
| `colla_rag_only.yaml` | ❌ | ✅ | `colla_rag.multi_agent` | 单独衡量**创新点二**的增益（云端 DeepSeek） |
| `full.yaml` | ✅ | ✅ | `colla_rag.multi_agent` | 构建侧基线 + 创新点二叠加 |
| `ollama_colla_rag_only.yaml` | ❌ | ✅ | `colla_rag.multi_agent` | 本地 Ollama Qwen3-8B（免 API 费用） |

**公共部分（三份云端配置相同）**：
- LLM：`deepseek-chat` @ `https://api.deepseek.com`
- Embedding：`text-embedding-3-small` @ `https://api.openai.com/v1`，**1536 维**（云端）
- 向量索引：`hnsw`（M=32 / efConstruction=200 / efSearch=64）
- 数据集：`mock_wiki`（100 条合成 wiki-style 多跳 QA）

**`ollama_colla_rag_only.yaml` 不同**：`qwen3:8b` + `nomic-embed-text`（768 维）@ `http://localhost:11434/v1`。

**`default.yaml` 不同**：走阿里云 DashScope（`qwen` + `text-embedding-v4` 1536 维 + `qwen3-rerank`），是字段齐全的注释模板。

## 命名规则

```
<plugin_combo>.yaml
      │
      ├── vanilla                      开关全关（defaults 裸基线）
      ├── adagraph_only                仅构建侧前置基线
      ├── colla_rag_only               仅创新点二（多Agent协作检索）
      └── full                         两者全开
```

## 典型运行

> 前置：`.env` 中配置 `LLM_API_KEY`。
>
> 说明：3 份云端配置用的是**云端 embedding**（`text-embedding-3-small`），不需要本地模型。
> 早期本地 embedding 方案的 `models/` 目录已作为遗留物清理；如需本地模型，
> `scripts/download_embedding_model.py` 可重新下载（当前配置均未引用它）。

### 单次查询
```bash
uv run python main.py query --config configs/colla_rag_only.yaml \
  "Which company acquired the employer of Ada Verdant?"
```

### 100 条基准评测（单配置，推理 + 打分两阶段解耦）
```bash
# 阶段 1：推理。首次会自动构图并落盘到 storage/，之后秒级复用；加 --rebuild 可强制重建
uv run python main.py infer \
  --config configs/colla_rag_only.yaml \
  --dataset mock_wiki --limit 100
# → output/mock_wiki/colla_rag_only/{predictions.jsonl, meta.json}

# 阶段 2：打分 + 报告（0 LLM 调用）
uv run python experiments/eval.py \
  --predictions output/mock_wiki/colla_rag_only/ \
  --metrics em,f1,rouge_l
```

### 多配置消融对比
```bash
# 对每份配置各跑一次 infer（各自复用各自的持久化图谱）
uv run python main.py infer --config configs/vanilla.yaml     --dataset mock_wiki --limit 100
uv run python main.py infer --config configs/colla_rag_only.yaml --dataset mock_wiki --limit 100

# 再用 experiments/eval.py 多目录对比（自动生成 compare 报告 + 柱图/雷达图）
uv run python experiments/eval.py \
  --predictions output/mock_wiki/vanilla/ \
                output/mock_wiki/colla_rag_only/ \
  --out output/mock_wiki/compare/ --metrics em,f1,rouge_l
```

## 改 LLM 连接信息

**模型名与 base_url 写在 yaml 里**（`llm.model` / `llm.base_url`），**`.env` 只放 secret**——key 的环境变量名由 `llm.api_key_env` 指定，默认 `LLM_API_KEY`。

```yaml
# configs/colla_rag_only.yaml
llm:
  model: deepseek-chat
  base_url: https://api.deepseek.com
  # api_key_env 默认 LLM_API_KEY
```

```bash
# .env（只需一行）
LLM_API_KEY=sk-xxx
```

优先级：**只读 `cfg.api_key_env` 指定的那一个变量**，找不到直接抛 `ProviderError`（`providers/llm.py` 里就是一句 `os.getenv(cfg.api_key_env)`，没有多级回退）。
`embedding` / `rerank` 可各自设 `api_key_env` 用不同的 key；`.env` 的加载是**非破坏性**的（`core/env_loader.py`，已存在的环境变量不会被覆盖）。

⚠️ **`model` 与 `base_url` 只从 yaml 读，不支持环境变量覆盖**（见 `tests/test_llm_env_override.py` 的文档串）。想换模型就改 yaml。

## 改向量算法 / device / 数据集只改 yaml

| 需求 | 改哪个字段 |
|---|---|
| 向量检索算法 | `storage.vector.index_type` (`flat_ip` / `flat_l2` / `ivf_flat` / `hnsw`) |
| HNSW 超参 | `storage.vector.hnsw.{M,ef_construction,ef_search}` |
| IVF 超参 | `storage.vector.ivf.{nlist,nprobe,train_size}` |
| Embedding 批大小 / 归一化 | `embedding.batch_size` / `embedding.normalize`（**无 `embedding.device` 字段**） |
| 数据集路径 | `datasets.<name>.{corpus_path, qa_path, output_dir}` |
| 抽取并发度 | `ingestion.extractor.params.extract_workers`（默认 1 = 串行；调大可显著缩短构图时间） |
| 检索 top_k | `storage.vector.top_k` 或 `query.retriever.params.top_k` |

## 可用数据集

**各配置注册的数据集并不一致**（`--dataset <name>` 只能选当前配置里声明过的）：

| 配置 | 注册的 datasets |
|---|---|
| `default.yaml` | `hotpotqa` · `two_wiki` · `musique` · `nq` · `popqa`（**不含 `mock_wiki`**） |
| `vanilla.yaml` / `adagraph_only.yaml` / `colla_rag_only.yaml` / `full.yaml` | 仅 `mock_wiki` |
| `ollama_colla_rag_only.yaml` | `mock_wiki` · `hotpotqa` · `two_wiki` · `musique` |

跑真实数据集（HotpotQA / 2Wiki / MuSiQue 等）时，建议以 `default.yaml` 为底改出实验配置，或把数据集条目补进目标配置。

`data/` 下已通过脚本下载好 5 个公开数据集，并提供两种格式：

| name | 格式 | loader | 来源 / 说明 |
|---|---|---|---|
| `hotpotqa` / `hotpotqa_mw` | 原始 / mock_wiki | `hotpotqa` / `mock_wiki` | HotpotQA distractor dev，7405 条 |
| `two_wiki` / `two_wiki_mw` | 原始 / mock_wiki | `two_wiki` / `mock_wiki` | 2WikiMultihopQA dev，12576 条 |
| `musique` / `musique_mw` | 原始 / mock_wiki | `musique` / `mock_wiki` | MuSiQue-Ans dev，2417 条 |
| `nq` / `nq_mw` | 原始 / mock_wiki | `generic_jsonl` / `mock_wiki` | Natural Questions (open)，3610 条；开放域无语料 |
| `popqa` / `popqa_mw` | 原始 / mock_wiki | `generic_jsonl` / `mock_wiki` | PopQA test，14267 条；开放域无语料 |
| `mock_wiki` | mock_wiki | `mock_wiki` | 100 条合成多跳 QA（默认/测试用） |

> ⚠️ `*_mw` 是 `data/` 里**文件目录**的命名（统一的 `corpus.txt` + `qa.jsonl` 布局），
> 目前 **`configs/*.yaml` 只注册了不带 `_mw` 的那一份**。要用 `_mw` 布局需自行补一条 dataset 条目并指向 `./data/<name>_mw/`。

- **`*_mw` 变体**：统一的 `corpus.txt`（空行分隔段落）+ `qa.jsonl`（`{question,answer,qid}`）布局，
  走 `mock_wiki` loader，便于统一摄取/评测流程。
- **`nq` / `popqa`** 为开放域，`*_mw` 的 `corpus.txt` 为空占位（数据集本身不带语料）。

下载与转换脚本（详见各脚本 docstring）：

```bash
uv run python scripts/download_datasets.py        # 下载 5 个数据集到 data/
uv run python scripts/convert_to_mock_wiki.py     # 转成 corpus.txt + qa.jsonl
```

## Mock 配置哪里去了？

为了避免 `configs/` 里留下一批既不跑真实 LLM 又不做重点测试的"占位" yaml，
我们把它们转为测试 fixture：

```python
# tests/conftest.py
@pytest.fixture
def mock_vanilla_yaml(tmp_path: Path) -> str: ...
@pytest.fixture
def mock_adagraph_only_yaml(tmp_path: Path) -> str: ...
@pytest.fixture
def mock_colla_rag_only_yaml(tmp_path: Path) -> str: ...
@pytest.fixture
def mock_full_yaml(tmp_path: Path) -> str: ...
```

测试里直接 `def test_xxx(mock_vanilla_yaml: str, ...)` 就能拿到一份
临时 YAML 路径，workspace_dir 自动挂在 `tmp_path` 下。需要新增 Mock
配置组合时，调用 `mock_config_yaml_builder(adagraph=..., colla_rag=...)`
即可。

## 实验专用配置

论文的 9 项实验（`experiments/` 目录下的脚本）会在运行时通过 `experiments/utils.py`
的 `make_config()` 动态生成临时配置文件，存放在 `output/experiments/<exp_name>/configs/`。
这些临时配置基于本目录的 yaml 派生（应用超参数覆盖、路由策略切换等），
不需要手动维护——实验脚本会自动处理。

示例：消融实验的 `w/o Memory` 变体会基于 `colla_rag_only.yaml`
生成一份禁用双层记忆的临时配置：

```python
cfg = make_config(
    base="colla_rag_only",
    overrides={"plugins": {"colla_rag": {"multi_agent": {"memory": {"shared": {"enabled": False}}}}}},
    workspace_suffix="exp2_wo_memory_hotpotqa",
)
```
