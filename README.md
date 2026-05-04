# Chimera-RAG

> 基于知识图谱的可插拔 RAG 框架，双创新点 **AdaGraph**（自适应知识图谱构建）+ **IntentAgent**（意图感知智能体）通过 `config.yaml` 一键开关。

## 特性

- **裸 KG-RAG 基线**：固定分块 + LLM 三元组提取 + NetworkX + FAISS，端到端可跑通
- **AdaGraph 插件**（创新点一）：动态粒度分块 + 分层三元组提取 + 双重冗余剪枝
- **IntentAgent 插件**（创新点二）：Memory → Reasoning → Decision → Execution 四阶段 + 3 个可插拔 Hook
- **统一入口**：`main.py` 支持 `ingest / query / evaluate / benchmark / export / serve`
- **Web 演示系统**：FastAPI + React/Vite/shadcn，覆盖摄取、查询、图谱、追踪、评测、插件、配置 7 大功能

## 快速开始

### 1. 安装（使用 uv）

```bash
# 安装 uv：https://github.com/astral-sh/uv
uv sync --all-extras --dev
```

### 2. 冒烟测试

```bash
uv run pytest -q
```

### 3. 运行（敬请期待）

```bash
uv run python main.py ingest  --config configs/vanilla.yaml --dataset sample
uv run python main.py query   --config configs/vanilla.yaml "who invented dynamite?"
uv run python main.py serve   --config configs/full.yaml
```

## 名字映射（研究材料 → 代码命名）

| 当前命名 | 原设计命名 |
|---------|-----------|
| Chimera-RAG | KG-RAG / Agent-Graph-RAG |
| AdaGraph | DEKER |
| IntentAgent | MRDE |

## 开发约定

- **TDD**：所有生产代码遵循 Red → Green → Refactor，先写失败测试
- **uv** 管理依赖；提交 `uv.lock`
- **分层解耦**：`core/` 不依赖 `defaults/` 与 `plugins/`
- **插件机制**：抽象基类 + `@register(slot, name)` + Registry 单例

## 项目状态

🚧 Work in Progress — 当前完成度请参考 `docs/plan.md`。
