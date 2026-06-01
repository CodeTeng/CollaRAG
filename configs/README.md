# Chimera-RAG 配置文件索引

本目录保留 **5 份 DeepSeek 真实运行配置**，覆盖两个创新点的组合及创新点 2b 的
LangGraph 动态编排变体。所有 Mock-based 测试配置已内联到 `tests/conftest.py`
（作为 pytest fixtures），不再作为文件存在，以保持 `configs/` 目录聚焦于真实基准。

运行任意子命令都需要用 `--config configs/<name>.yaml` 指定其一。

## 速查表

| 文件 | AdaGraph | IntentAgent | 检索器 | 主要用途 |
|---|:---:|:---:|---|---|
| `deepseek_vanilla.yaml` | ❌ | ❌ | `defaults.direct` | 真实 API 基线（消融对照组） |
| `deepseek_adagraph_only.yaml` | ✅ | ❌ | `defaults.direct` | 真实 API 下单独衡量创新点一的增益 |
| `deepseek_intent_only.yaml` | ❌ | ✅ | `intent_agent.strategy_selector`（静态路由） | 真实 API 下单独衡量创新点二的增益 |
| `deepseek_intent_langgraph.yaml` | ❌ | ✅ | `intent_agent.langgraph`（动态编排） | 创新点 2b：对照 `intent_only` 衡量「动态工具编排 vs 静态路由」的增益（需 `uv sync --extra agent`） |
| `deepseek_full.yaml` | ✅ | ✅ | `intent_agent.strategy_selector` | 真实 API 下两创新点叠加的最终表现 |

**公共部分（四份相同）**：
- LLM：`deepseek`（model 从 `.env` 的 `LLM_MODEL` 读，默认 `deepseek-chat`）
- Embedding：本地 `sentence-transformers/all-MiniLM-L6-v2`（384 维，`./models/` 下）
- 向量索引：`hnsw`（M=32 / efConstruction=200 / efSearch=64）
- Device：`cpu`
- 数据集：`mock_wiki`（100 条合成 wiki-style 多跳 QA）

## 命名规则

```
deepseek_<plugin_combo>.yaml
             │
             ├── vanilla           两创新点全关（defaults 裸基线）
             ├── adagraph_only     仅创新点一
             ├── intent_only       仅创新点二（静态意图→策略路由）
             ├── intent_langgraph  创新点 2b（LangGraph 动态工具编排，需 --extra agent）
             └── full              两创新点全开
```

> **创新点 2b 专属前置**：`deepseek_intent_langgraph.yaml` 依赖可选 extra
> `agent`（`langgraph>=1.0` + `langchain-core>=1.0`）。先执行
> `uv sync --extra agent`（或 `uv sync --all-extras --dev`）再运行；未安装时
> 该检索器不会注册，配置切到它会报清晰错误。

## 典型运行

> 前置：`.env` 中配置 `LLM_API_KEY`（可选 `LLM_BASE_URL`、`LLM_MODEL`）；
> 本地模型已通过 `scripts/download_embedding_model.py` 下载。

### 单次查询
```bash
uv run python main.py query --config configs/deepseek_intent_only.yaml \
  "Which company acquired the employer of Ada Verdant?"
```

### 100 条基准评测（单配置）
```bash
uv run python main.py evaluate \
  --config configs/deepseek_intent_only.yaml \
  --dataset mock_wiki --limit 100
# 输出自动落到 output/mock_wiki/deepseek_intent_only.json
```

### 多配置消融对比
```bash
uv run python main.py benchmark \
  --configs configs/deepseek_vanilla.yaml,configs/deepseek_intent_only.yaml \
  --dataset mock_wiki --limit 100
```

## 改 LLM 连接信息

`deepseek_*.yaml` 里 `llm:` 段不写 `model / api_key_env / base_url_env`，
全部从 `.env` 读（优先级 `LLM_API_KEY > DEEPSEEK_API_KEY > provider 默认`）。

```bash
# .env
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

## 改向量算法 / device / 数据集只改 yaml

| 需求 | 改哪个字段 |
|---|---|
| 向量检索算法 | `storage.vector.index_type` (`flat_ip` / `flat_l2` / `ivf_flat` / `hnsw`) |
| HNSW 超参 | `storage.vector.hnsw.{M,ef_construction,ef_search}` |
| IVF 超参 | `storage.vector.ivf.{nlist,nprobe,train_size}` |
| Embedding device | `embedding.device` (`cpu` / `cuda` / `mps` / `auto`) |
| 数据集路径 | `datasets.<name>.{corpus_path, qa_path, output_dir}` |

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
def mock_intent_only_yaml(tmp_path: Path) -> str: ...
@pytest.fixture
def mock_full_yaml(tmp_path: Path) -> str: ...
```

测试里直接 `def test_xxx(mock_vanilla_yaml: str, ...)` 就能拿到一份
临时 YAML 路径，workspace_dir 自动挂在 `tmp_path` 下。需要新增 Mock
配置组合时，调用 `mock_config_yaml_builder(adagraph=..., intent_agent=...)`
即可。
