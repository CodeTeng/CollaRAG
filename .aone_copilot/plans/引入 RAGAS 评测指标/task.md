- [x] 新建 `src/chimera_rag/defaults/langchain_chunker.py`：基于 RecursiveCharacterTextSplitter 的可配置长度分块器
- [x] 修改 `src/chimera_rag/defaults/__init__.py`：注册导入新分块器
- [x] 新建 `tests/test_langchain_chunker.py`：分块器单测
- [x] 修改 `src/chimera_rag/evaluation/inference.py`：新增 `_lookup_texts`，row 写入 `evidence_texts`
- [x] 修改 `tests/test_cli_infer.py`：补充 `evidence_texts` 字段断言
- [x] 新建 `src/chimera_rag/evaluation/ragas_eval.py`：`RagasEvaluator`（从 config 构建 judge LLM + embedding + 数据映射 + ragas.evaluate）
- [x] 新建 `src/chimera_rag/evaluation/ragas_report.py`：`RagasReporter` 输出 md/json/csv
- [x] 修改 `src/chimera_rag/evaluation/__init__.py`：导出 RagasEvaluator 与 RagasReporter
- [x] 新建 `eval_ragas.py`：命令行入口（--config 提供 LLM/embedding）
- [x] 新建 `tests/test_ragas_eval.py`：mock 数据映射 + 降级 + integration 冒烟
- [x] 修改 `pyproject.toml`：新增 ragas 与 chunk 可选依赖组
- [x] 修改 `README.md`：补充 RAGAS 与 langchain 分块器用法
- [x] [manual] `uv sync --extra ragas --extra chunk` 安装依赖（解决 langchain-core 冲突：tool.uv conflicts + ragas>=0.3 + pillow）
- [x] [manual] 重跑 infer 生成带 evidence_texts 的新 predictions（DeepSeek 超时/Ollama 极慢，改用 mock provider 验证 evidence_texts 写入链路：5/5 chunk 反查原文成功）
- [x] [manual] 运行 eval_ragas.py 验证四个指标产出与报告文件（端到端跑通，本地 Ollama 评判+ST embedding，四指标全计算，md/json/csv 报告正常）
- [x] [manual] `uv run pytest -m "not integration and not slow"` 确认无回归（全量通过，仅 langgraph 测试因 agent/ragas extra 互斥按设计 SKIP；新增的 chunker/ragas/cli 测试全绿）

---
生成时间: 2026/6/4 11:13:24
planId: e3e978cf-9f68-4ba2-8e07-f282bf2ff78b