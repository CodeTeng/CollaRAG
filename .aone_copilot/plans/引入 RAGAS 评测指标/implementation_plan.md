

# 引入 RAGAS 评测指标

为 Chimera-RAG 接入 [RAGAS](https://docs.ragas.io) 框架，新增四个语义级 RAG 指标，弥补现有 EM/F1/ROUGE-L 纯词面匹配的不足；同时按需新增一个基于 LangChain 的长度分块器。

- **Faithfulness**：答案是否能由检索 context 支撑（衡量幻觉）
- **Answer Relevancy**：答案与问题的相关程度
- **Context Precision**：检索 context 中有用片段的排序质量
- **Context Recall**：标准答案所需信息是否被 context 覆盖

## User Review Required

> [!IMPORTANT]
> **数据缺口需重跑 infer**：RAGAS 的 Faithfulness / Context Precision / Context Recall 依赖检索片段（context）的**原文**，当前 `predictions.jsonl` 只有 `evidence_chunk_ids`。需修改推理阶段写入 `evidence_texts`，并**重新跑一次 infer** 生成新数据。

> [!NOTE]
> **judge LLM 与 embedding 复用 config**（已按你的要求调整）：`eval_ragas.py` 只需 `--config`，judge LLM 从 `config.llm`（DeepSeek/OpenAI 兼容）构建，embedding 从 `config.embedding` 构建。`deepseek_intent_only.yaml` 的 embedding 配的是本地 `sentence_transformer`，正好规避 DeepSeek 无 embedding endpoint 的问题。

> [!NOTE]
> **LangChain 分块器为独立增量**：现有 `defaults.fixed` 已是按字符长度、可配 `chunk_size`/`overlap` 的滑窗分块。本次新增一个用 LangChain `RecursiveCharacterTextSplitter` 的分块器并存，可在 config 切换，不改动 `defaults.fixed` 的行为。

## Proposed Changes

### LangChain 长度分块器（独立增量）

#### [NEW] [langchain_chunker.py](file:///Users/teng/Projects/my/Chimera-RAG/src/chimera_rag/defaults/langchain_chunker.py)
- 注册 `@register("chunker", "defaults.langchain")`，类 `LangChainChunker(BaseChunker)`。
- 构造参数 `chunk_size: int = 300`、`overlap: int = 50`，内部用 `langchain_text_splitters.RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=...)`。
- `chunk(document)` 调用 `splitter.split_text(document.content)`，产出 `Chunk(chunk_id=f"{doc_id}::{i}", ...)`，chunk_id 命名与 `FixedSizeChunker` 保持一致以兼容下游。
- `langchain_text_splitters` 延迟导入，缺失时给出清晰报错。

#### [MODIFY] [registry / defaults 导入处](file:///Users/teng/Projects/my/Chimera-RAG/src/chimera_rag/defaults/__init__.py)
- 确保新分块器模块在 defaults 包导入时被注册（与 `fixed_chunker` 同样方式）。

---

### 推理阶段：补齐 context 原文

#### [MODIFY] [inference.py](file:///Users/teng/Projects/my/Chimera-RAG/src/chimera_rag/evaluation/inference.py)
- 新增模块级 `_lookup_texts(pipeline, chunk_ids) -> list[str]`：从 `getattr(pipeline, "chunk_lookup", {})`（回退 `pipeline.state.chunk_lookup`）按 id 反查 `Chunk.text`，查不到则跳过。
- 在 `run` 写 `row` 时新增 `"evidence_texts": _lookup_texts(pipeline, answer.evidence_chunk_ids)`，不破坏现有字段。

---

### RAGAS 评测核心模块（复用 config 的 LLM/embedding）

#### [NEW] [ragas_eval.py](file:///Users/teng/Projects/my/Chimera-RAG/src/chimera_rag/evaluation/ragas_eval.py)
- 类 `RagasEvaluator(config)`，职责：
  - **judge LLM**：用 `config.llm` 的连接信息（model/base_url/api_key，与 `OpenAILLMProvider` 同源的环境变量解析）构造 `langchain_openai.ChatOpenAI`，包成 `ragas.llms.LangchainLLMWrapper`。
  - **embedding**：根据 `config.embedding.provider` 构造对应 LangChain embeddings —— `sentence_transformer` → `langchain_huggingface.HuggingFaceEmbeddings(model_name=config.embedding.model)`；`openai` → `OpenAIEmbeddings`；包成 `ragas.embeddings.LangchainEmbeddingsWrapper`。
  - **数据映射**：`read_predictions_dir` 的 rows → RAGAS `EvaluationDataset`（`user_input`=question、`response`=prediction、`retrieved_contexts`=evidence_texts、`reference`=reference）。
  - 调 `ragas.evaluate(dataset, metrics=[...], llm=..., embeddings=...)`，返回聚合分 + per-example 分。
  - 所有 `ragas` / `langchain*` import 延迟到方法内部。

#### [NEW] [ragas_report.py](file:///Users/teng/Projects/my/Chimera-RAG/src/chimera_rag/evaluation/ragas_report.py)
- `RagasReporter`：输出 `ragas_report.md` / `.json` / `.csv`，风格对齐 `EvalReporter`。

#### [MODIFY] [__init__.py](file:///Users/teng/Projects/my/Chimera-RAG/src/chimera_rag/evaluation/__init__.py)
- 导出 `RagasEvaluator`、`RagasReporter`。

---

### 独立命令行入口

#### [NEW] [eval_ragas.py](file:///Users/teng/Projects/my/Chimera-RAG/eval_ragas.py)
- 参数：`--predictions <dir>`（必填）、`--config <yaml>`（必填，提供 judge LLM + embedding）、`--metrics`（默认四个全开）、`--out`（默认 predictions 目录）、`--limit`、`--log-level`。
- 流程：`load_config(--config)` → `read_predictions_dir` → 校验 `evidence_texts` 是否存在（缺失时对 context 类指标告警/跳过）→ `RagasEvaluator(config).evaluate(rows, metrics)` → `RagasReporter.write` → 打印聚合分。

---

### 依赖与文档

#### [MODIFY] [pyproject.toml](file:///Users/teng/Projects/my/Chimera-RAG/pyproject.toml)
- `[project.optional-dependencies]` 新增：

```toml
ragas = [
    "ragas>=0.2",
    "langchain-openai>=0.2",
    "langchain-huggingface>=0.1",
    "datasets>=2.16",
]
chunk = [
    "langchain-text-splitters>=0.3",
]
```

#### [MODIFY] [README.md](file:///Users/teng/Projects/my/Chimera-RAG/README.md)
- 评测章节补充 RAGAS 用法（`uv sync --extra ragas` → 重跑 infer → `uv run python eval_ragas.py --predictions <dir> --config <yaml>`）；分块章节补充 `defaults.langchain` 选项。

---

### 测试

#### [NEW] [test_ragas_eval.py](file:///Users/teng/Projects/my/Chimera-RAG/tests/test_ragas_eval.py)
- monkeypatch `ragas.evaluate`，验证数据映射正确（question/prediction/contexts/reference 对齐）与缺 `evidence_texts` 的降级行为。
- `@pytest.mark.integration` 真实 DeepSeek 冒烟（默认 skip）。

#### [NEW] [test_langchain_chunker.py](file:///Users/teng/Projects/my/Chimera-RAG/tests/test_langchain_chunker.py)
- 验证 `LangChainChunker` 按 `chunk_size`/`overlap` 切分、chunk_id 命名、空文档边界。

#### [MODIFY] [test_cli_infer.py](file:///Users/teng/Projects/my/Chimera-RAG/tests/test_cli_infer.py)
- 断言 infer 产物每行含 `evidence_texts`。

## Verification Plan

### Automated Tests
- `uv run pytest tests/test_ragas_eval.py tests/test_langchain_chunker.py -v`（mock 路径，无需网络）
- `uv run pytest tests/test_cli_infer.py -v`
- `uv run pytest -m "not integration and not slow"`（确认无回归）
- `uv run ruff check src/chimera_rag/ eval_ragas.py`

### Manual Verification
- [manual] `uv sync --extra ragas --extra chunk` 安装依赖
- [manual] 重跑推理生成带 context 的新数据：`uv run python main.py infer --config configs/deepseek_intent_only.yaml --dataset mock_wiki`
- [manual] 运行 `uv run python eval_ragas.py --predictions output/mock_wiki/deepseek_intent_only/ --config configs/deepseek_intent_only.yaml`，确认四个指标 0~1 合理并生成报告
- [manual]（可选）在 config 中将 chunker 切到 `defaults.langchain` 验证分块器生效


---
生成时间: 2026/6/4 11:13:24
planId: e3e978cf-9f68-4ba2-8e07-f282bf2ff78b
plan_status: review