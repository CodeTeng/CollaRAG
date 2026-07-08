# Plan: MultiHop Agent 升级为 Plan-Execute 架构 + 全部 Agent 重命名

## Context

用户的 multi_hop agent 是最重要的 agent，目前使用纯 ReAct 循环（Thought→Action→Observation 平铺循环）。用户希望升级为 **Plan-Execute-Reflect** 三阶段架构：先规划步骤，中间每步用 ReAct 执行，最后反思综合。同时，所有 agent 类名从描述性命名（`ReActAgent`、`NativeRAGAgent` 等）改为注册名风格（`MultiHopAgent`、`SingleHopAgent` 等），并同步更新文档。

---

## 一、Agent 重命名

### 类名映射

| 旧类名 | 新类名 | 文件名变化 | 注册名（不变） |
|---|---|---|---|
| `ReActAgent` | `MultiHopAgent` | `react_agent.py` → `multi_hop.py` | `multi_hop` |
| `NativeRAGAgent` | `SingleHopAgent` | `native_rag.py` → `single_hop.py` | `single_hop` |
| `MapReduceAgent` | `SummarizationAgent` | `map_reduce.py` → `summarization.py` | `summarization` |
| `WebSearchAgent` | `OtherAgent` | `web_search.py` → `other.py` | `other` |
| `PreprocessorAgent` | 不变 | 不变 | `preprocessor` |

### 需修改的源码文件

1. **`agents/__init__.py`** — import 路径和 `__all__` 列表
2. **`agents/multi_hop.py`**（原 `react_agent.py`）— 类名 + docstring
3. **`agents/single_hop.py`**（原 `native_rag.py`）— 类名 + docstring
4. **`agents/summarization.py`**（原 `map_reduce.py`）— 类名 + docstring
5. **`agents/other.py`**（原 `web_search.py`）— 类名 + docstring

### 需修改的测试文件

1. **`tests/test_react_agent.py`** — import 路径、类名引用，文件名改为 `test_multi_hop_agent.py`
2. **`tests/test_native_rag_agent.py`** — 同上，改为 `test_single_hop_agent.py`
3. **`tests/test_map_reduce_agent.py`** — 同上，改为 `test_summarization_agent.py`
4. **`tests/test_web_search_agent.py`** — 同上，改为 `test_other_agent.py`

> 注：`agent_factory.py`、`orchestrator.py`、`TOOL_MATRIX` 中使用的都是注册名字符串（`"multi_hop"`、`"single_hop"` 等），**不需要修改**。

---

## 二、MultiHopAgent 升级为 Plan-Execute-Reflect 架构

### 设计思路

将原来的平铺 ReAct 循环改为三阶段：

```
Plan（规划）→ Execute（逐步 ReAct 执行）→ Reflect（反思综合）
```

### 阶段 1：Plan（规划）

- 使用新的 `PLAN_PROMPT`，让 LLM 分析 query 并生成结构化执行计划
- 输出：一个步骤列表（JSON），每步包含 `step`（描述）和 `tools_hint`（建议使用的工具）
- 如果 LLM 返回解析失败，降级为单步计划 `[{"step": original_query}]`

### 阶段 2：Execute（逐步执行）

- 对计划中的每个步骤，运行一个**迷你 ReAct 循环**（复用现有的 Action/Observation 解析逻辑）
- 每步最多 `max_step_iterations`（默认 2）轮 ReAct
- 每步的结果（evidence）累积到一个 `step_results` 列表

### 阶段 3：Reflect（反思综合）

- 将所有步骤的 evidence 汇总
- 调用 `assess_evidence` 评估整体质量
- 若质量不达标（`< quality_threshold`）且还有重试预算（`reflect_retries`，默认 1），生成反思文本，回到 Execute 阶段补充证据
- 最终用 `SYNTHESIZE_PROMPT` 综合所有证据生成最终答案

### 新增 Prompt 常量（在 `prompts.py` 中）

```python
PLAN_PROMPT = """You are a retrieval planning agent. Analyze the query and create a step-by-step plan.

Available tools:
{tool_descriptions}

Query: {query}

Output a JSON list of steps. Each step has "step" (description) and "tools_hint" (suggested tools).
Example: [{{"step": "Find information about X", "tools_hint": ["hybrid_search"]}}, ...]

Plan:"""

STEP_REACT_PROMPT = """You are executing step {step_num} of a retrieval plan.

Plan step: {step_description}
Original query: {query}
Previous steps evidence: {previous_evidence}

Available tools:
{tool_descriptions}

Use the following format:
Thought: reason about what to do for this step
Action: tool_name
Action Input: {{"param": "value"}}
Observation: (tool result)
... (repeat if needed)
Step Result: summarize what you found in this step

Begin!
Thought:"""

REFLECT_PROMPT = """Review the evidence gathered across all plan steps.

Original query: {query}
Plan: {plan}
Step results:
{step_results}

Quality assessment: {quality_feedback}

What is missing? What should be investigated further? Be specific.
Reflection:"""

SYNTHESIZE_PROMPT = """Based on all gathered evidence, provide a comprehensive answer.

Query: {query}
Evidence from plan execution:
{all_evidence}

Answer:"""
```

### 代码结构（`agents/multi_hop.py`）

```python
class MultiHopAgent(BaseAgent):
    def do_execute(self, query, context) -> Answer:
        # --- Phase 1: Plan ---
        plan = self._plan(query)
        
        # --- Phase 2: Execute ---
        step_results = self._execute_plan(query, plan)
        
        # --- Phase 3: Reflect ---
        answer = self._reflect_and_synthesize(query, plan, step_results)
        
        return answer
    
    def _plan(self, query) -> list[dict]:
        """LLM 生成执行计划，解析为步骤列表。"""
        
    def _execute_step(self, query, step, step_num, previous_evidence) -> str:
        """对单个步骤运行迷你 ReAct 循环。"""
        # 复用现有 _ACTION_RE 正则解析
        
    def _execute_plan(self, query, plan) -> list[str]:
        """依次执行计划中的每个步骤。"""
        
    def _reflect_and_synthesize(self, query, plan, step_results) -> Answer:
        """反思 + 综合最终答案。"""
```

### trace 记录扩展

```python
context.trace["plan"] = plan                    # 执行计划
context.trace["step_results"] = step_results    # 每步结果
context.trace["iterations"] = total_iterations  # 总迭代数
context.trace["tool_call_log"] = tool_call_log  # 工具调用序列
context.trace["reflect_count"] = reflect_count  # 反思次数
```

### 配置扩展

YAML `plugins.intent_agent.multi_agent.react` 新增两个可选参数：

```yaml
react:
  max_iterations: 3          # 保留（现在作为总步骤数上限的兜底）
  quality_threshold: 0.6     # 保留
  max_plan_steps: 5          # 新增：计划最大步骤数
  max_step_iterations: 2     # 新增：每步 ReAct 最大迭代数
  reflect_retries: 1         # 新增：反思重试次数
```

### `strategy_name` 变化

- 旧：`strategy_name="react"`
- 新：`strategy_name="plan_execute"`

---

## 三、文档更新

以下文档需同步更新 agent 名称和 multi_hop 架构描述：

### 必须更新的文档

1. **`README.md`**（~第 126-129 行）— 意图路由表中的 Agent 名称
2. **`CLAUDE.md`**（~第 126-129 行）— 目录结构中的文件名和类名
3. **`docs/architecture_overview.md`** — agent 名称表、数据流图
4. **`docs/innovation-2/intent-agent.md`** — 全文（agent 名称、§6 ReAct Agent 改为 Plan-Execute、§10 工具数修正为 22）
5. **`docs/innovation-2/design.md`** — §6.4 ReAct Agent 设计改为 Plan-Execute

### 更新要点

- 所有 `ReAct Agent` → `MultiHop Agent`（Plan-Execute 架构）
- 所有 `Native RAG Agent` → `SingleHop Agent`
- 所有 `Map-Reduce Agent` → `Summarization Agent`
- 所有 `WebSearch + LLM Agent` → `Other Agent`
- multi_hop 的架构描述从 ReAct 循环改为 Plan → Execute → Reflect 三阶段
- `docs/innovation-2/intent-agent.md` §10 标题 "19 个工具" → "22 个工具"

---

## 四、实施顺序

1. 重命名 4 个 agent 源码文件（写新文件，删旧文件，修改类名/docstring/__all__）
2. 更新 `agents/__init__.py`
3. 重写 `agents/multi_hop.py` 为 Plan-Execute-Reflect 架构
4. 在 `prompts.py` 添加 4 个新 prompt（PLAN/STEP_REACT/REFLECT/SYNTHESIZE），删除旧 REACT_SYSTEM_PROMPT
5. 重命名 4 个测试文件，更新 import 和类名引用
6. 为 MultiHopAgent 新增 plan-execute 测试用例
7. 更新 5 个文档文件
8. 运行全部测试验证

---

## 五、验证方式

```bash
# 1. 全量测试
cd /Users/teng/Projects/my/Chimera-RAG && uv run pytest tests/ -v

# 2. 重点检查 agent 相关测试
uv run pytest tests/test_multi_hop_agent.py tests/test_single_hop_agent.py tests/test_summarization_agent.py tests/test_other_agent.py tests/test_agent_factory.py tests/test_orchestrator.py -v

# 3. 确认旧文件名不再存在
ls src/chimera_rag/plugins/intent_agent/agents/
# 应只有: __init__.py multi_hop.py single_hop.py summarization.py other.py preprocessor.py

# 4. 确认旧类名不再出现
grep -rn "ReActAgent\|NativeRAGAgent\|MapReduceAgent\|WebSearchAgent" src/ tests/ --include="*.py"
# 应无输出
```
