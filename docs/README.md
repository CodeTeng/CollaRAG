# Chimera-RAG 文档目录

本目录存放项目设计文档与开发计划，作为仓库版本化的"事实来源"。

## 文档清单

| 文件 | 用途 |
|------|------|
| `plan.md` | 项目实施计划（从 Plan 模式同步，包含架构、目录结构、实施策略、页面规划等全部决策） |

## 名称映射

项目中出现的命名与早期设计材料（`my/` 目录）的对应关系：

| 当前命名 | 原设计命名 | 含义 |
|---------|-----------|------|
| **Chimera-RAG** | KG-RAG / Agent-Graph-RAG | 主框架名，希腊神话融合体象征双创新点 |
| **AdaGraph** | DEKER | 创新点一：Adaptive Graph Construction，知识构建双重增强 |
| **IntentAgent** | MRDE | 创新点二：意图感知智能体（Memory-Reasoning-Decision-Execution） |

## 开发方式

项目采用 **TDD（测试驱动开发）** 方式推进：

1. **RED**：先写失败的测试
2. **GREEN**：写最小代码让测试通过
3. **REFACTOR**：在保持绿灯的前提下重构
4. **COMMIT**：每个 TDD 循环结束做一次 git 提交

参见 `plan.md` 中的实施策略与 todolist。
