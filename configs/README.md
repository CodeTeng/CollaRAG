# Chimera-RAG 配置文件索引

本目录下共 **9 份 YAML**，按「LLM 后端 × 插件组合」两维组织。运行任意子命令都需要用
`--config configs/<name>.yaml` 指定其一。

## 速查表

| 文件 | LLM / Embedding | AdaGraph | IntentAgent | 主要用途 |
|---|---|:---:|:---:|---|
| `default.yaml` | mock / mock | — | — | **全字段参考模板，不用于运行** |
| `mock_vanilla.yaml` | mock / mock | ❌ | ❌ | 离线 CI 烟雾测试；纯 `defaults.*` 基线 |
| `mock_adagraph_only.yaml` | mock / mock | ✅ | ❌ | 离线单测创新点一的注入与运行 |
| `mock_intent_only.yaml` | mock / mock | ❌ | ✅ | 离线单测创新点二的意图 + 策略链路 |
| `mock_full.yaml` | mock / mock | ✅ | ✅ | 离线集成测试两创新点共存 |
| `deepseek_vanilla.yaml` | deepseek / ST-MiniLM | ❌ | ❌ | 真实 API 基线（消融对照组） |
| `deepseek_adagraph_only.yaml` | deepseek / ST-MiniLM | ✅ | ❌ | 真实 API 下单独衡量创新点一的增益 |
| `deepseek_intent_only.yaml` | deepseek / ST-MiniLM | ❌ | ✅ | 真实 API 下单独衡量创新点二的增益 |
| `deepseek_full.yaml` | deepseek / ST-MiniLM | ✅ | ✅ | 真实 API 下两创新点叠加的最终表现 |

> ST-MiniLM = `sentence-transformers/all-MiniLM-L6-v2`，384 维，本地
> `./models/sentence-transformers_all-MiniLM-L6-v2`（首次使用需执行
> `uv run python scripts/download_embedding_model.py`）。

## 命名约定

```
<backend>_<plugin_combo>.yaml
  ^             ^
  |             └── vanilla  两创新点全关（defaults baseline）
  |                 adagraph_only  仅创新点一
  |                 intent_only    仅创新点二
  |                 full           两创新点全开
  |
  └── mock      纯 Mock Provider，离线可跑
      deepseek  真实 DeepSeek Chat + 本地 sentence-transformers embedding
```

## 典型运行

### 离线（不需要 API key）
```bash
# Mock 全链路端到端
uv run python main.py ingest --config configs/mock_full.yaml --dataset sample
uv run python main.py query  --config configs/mock_full.yaml "What is Alice?"
```

### 真实 DeepSeek（需要 `.env` 中设置 `LLM_API_KEY`）
```bash
# 单次查询
uv run python main.py query --config configs/deepseek_intent_only.yaml \
  "Which company acquired the employer of Ada Verdant?"

# 100 条消融对比（输出自动落到 output/mock_wiki/<config_stem>.json）
uv run python main.py benchmark \
  --configs configs/deepseek_vanilla.yaml,configs/deepseek_intent_only.yaml,configs/deepseek_full.yaml \
  --dataset mock_wiki --limit 100
```

## 改 LLM 连接信息不用改 yaml

`deepseek_*.yaml` 里 `llm:` 段不写 `model / api_key_env / base_url_env`，
全部从 `.env` 读（优先级 `LLM_API_KEY > DEEPSEEK_API_KEY > provider 默认`）。
切换到 OpenAI / 自部署 Qwen 等 OpenAI 协议兼容端点，改 `.env` 即可：

```bash
# .env
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

## 改向量算法 / device / 数据集只改 yaml

| 需求 | 改哪个字段 |
|---|---|
| 向量检索算法 | `storage.vector.index_type` (`flat_ip` / `flat_l2` / `ivf_flat` / `hnsw`) |
| HNSW 超参 | `storage.vector.hnsw.{M,ef_construction,ef_search}` |
| IVF 超参 | `storage.vector.ivf.{nlist,nprobe,train_size}` |
| Embedding device | `embedding.device` (`cpu` / `cuda` / `mps` / `auto`) |
| 数据集路径 | `datasets.<name>.{corpus_path, qa_path, output_dir}` |

## 配置派生建议

如果要新增一份专用配置（例如 `deepseek_intent_only_with_reranker.yaml`），
直接复制最接近的一份，再按需修改。**永远从 `default.yaml` 或最接近的 Mock
配置开始派生**；避免新建"半完整"的配置漏字段。
