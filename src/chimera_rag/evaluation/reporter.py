"""评测报告生成器 (中文版)

输入 :class:`EvalResult`，同时输出:

* ``<stem>.json`` —— 完整 EvalResult（model_dump），机器可读
* ``<stem>.md``   —— 可读的 Markdown 报告，章节如下:
    1. 总览（样本数 / 耗时 / Token 消耗 / 总平均指标）
    2. 按意图分组（仅当 IntentAgent 标签存在时显示）
    3. Top-10 最差样本（按 F1）
    4. Top-10 最好样本（按 F1）
    5. 全部样本（可折叠表格）
* ``<stem>.csv``  —— 每行一个样本的表格，适合拖进 Excel / 飞书 分析

报告生成器零额外依赖（不用 Jinja / CSV 库），只做字符串拼接。
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from chimera_rag.core.types import EvalResult

# 常见 metric 的中文标签
_METRIC_LABEL_ZH = {
    "em": "精确匹配 EM",
    "f1": "F1",
    "rouge_l": "ROUGE-L",
    "avg_latency_s": "平均延迟 (秒)",
}

# 6 类意图的中文显示
_INTENT_LABEL_ZH = {
    "factual": "事实型",
    "analytical": "分析型",
    "comparative": "比较型",
    "multi_hop": "多跳型",
    "exploratory": "探索型",
    "follow_up": "追问型",
}


def _fmt_int(n: int) -> str:
    """给数字加千位分隔符，便于阅读。"""
    return f"{n:,}"


def _fmt_duration(seconds: float) -> str:
    """把秒转成 `1m 23.4s` 的中文可读格式。"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)} 分 {s:.1f} 秒"
    h, m = divmod(int(m), 60)
    return f"{h} 时 {m} 分 {s:.0f} 秒"


class EvalReporter:
    """根据 EvalResult 渲染 json + markdown + csv 三件套。"""

    def __init__(self, result: EvalResult) -> None:
        self.result = result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write(self, out_dir: Path, stem: str) -> dict[str, Path]:
        """写出三种产物到 ``out_dir``；返回 `{kind: path}` 字典。"""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": out_dir / f"{stem}.json",
            "md":   out_dir / f"{stem}.md",
            "csv":  out_dir / f"{stem}.csv",
        }
        paths["json"].write_text(
            json.dumps(self.result.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths["md"].write_text(self.render_markdown(), encoding="utf-8")
        paths["csv"].write_text(self.render_csv(), encoding="utf-8")
        return paths

    # ------------------------------------------------------------------
    # Markdown (中文)
    # ------------------------------------------------------------------
    def render_markdown(self) -> str:
        r = self.result
        rows = r.per_example
        n = len(rows)

        lines: list[str] = []
        lines.append(f"# 评测报告 · `{r.config_name}`")
        lines.append("")

        # ---- 1. 总览 -------------------------------------------------
        lines.append("## 一、总览")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| 样本数 | {_fmt_int(n)} |")
        if r.elapsed_s:
            lines.append(f"| 总耗时 | {_fmt_duration(r.elapsed_s)} |")
            qps = n / r.elapsed_s if r.elapsed_s > 0 else 0
            lines.append(f"| 吞吐 | {qps:.2f} 题/秒 |")
        # LLM 调用与 Token 消耗
        if r.usage:
            calls = int(r.usage.get("calls", 0))
            prompt_t = int(r.usage.get("prompt_tokens", 0))
            completion_t = int(r.usage.get("completion_tokens", 0))
            total_t = int(r.usage.get("total_tokens", 0))
            if calls:
                lines.append(f"| LLM 调用次数 | {_fmt_int(calls)} |")
            if total_t:
                lines.append(f"| 输入 Token | {_fmt_int(prompt_t)} |")
                lines.append(f"| 输出 Token | {_fmt_int(completion_t)} |")
                lines.append(f"| 合计 Token | {_fmt_int(total_t)} |")
                if n > 0:
                    lines.append(f"| 平均每题 Token | {total_t / n:.1f} |")
        lines.append("")

        lines.append("### 综合指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---|")
        for k, v in r.metrics.items():
            label = _METRIC_LABEL_ZH.get(k, k.replace("_", " ").title())
            if k == "avg_latency_s":
                lines.append(f"| {label} | {v:.2f} |")
            else:
                lines.append(f"| {label} | {v:.4f} |")
        lines.append("")

        # ---- 2. 按意图分组 -------------------------------------------
        intents_present = [row.get("intent_label") for row in rows if row.get("intent_label")]
        if intents_present:
            lines.append("## 二、按意图分组")
            lines.append("")
            bucket: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                bucket[row.get("intent_label") or "(未识别)"].append(row)
            metric_keys = [k for k in r.metrics if k != "avg_latency_s"]
            header_labels = [_METRIC_LABEL_ZH.get(k, k.upper()) for k in metric_keys]
            header = ["意图", "样本数", *header_labels, "平均延迟(秒)", "平均 Token"]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for label in sorted(bucket):
                grp = bucket[label]
                avg_latency = statistics.mean(r_["latency_s"] for r_ in grp) if grp else 0.0
                avg_tokens = statistics.mean(r_.get("total_tokens", 0) for r_ in grp) if grp else 0.0
                metric_cells = []
                for mk in metric_keys:
                    vals = [r_.get(mk, 0.0) for r_ in grp]
                    metric_cells.append(f"{statistics.mean(vals):.3f}" if vals else "-")
                display_label = _INTENT_LABEL_ZH.get(label, label)
                cells = [
                    display_label, str(len(grp)),
                    *metric_cells,
                    f"{avg_latency:.2f}", f"{avg_tokens:.0f}",
                ]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

            # ---- 2b. 策略分布 ---------------------------------------
            strategies = Counter(row.get("strategy_name") for row in rows if row.get("strategy_name"))
            if strategies:
                lines.append("### 策略分布")
                lines.append("")
                lines.append("| 策略 | 命中次数 |")
                lines.append("|---|---|")
                for s, c in strategies.most_common():
                    lines.append(f"| `{s}` | {c} |")
                lines.append("")

        # ---- 3. Top-10 最差 / 最好 ----------------------------------
        if rows and "f1" in rows[0]:
            sorted_by_f1 = sorted(rows, key=lambda r_: r_["f1"])
            worst = sorted_by_f1[:10]
            best = list(reversed(sorted_by_f1[-10:]))

            lines.append("## 三、最差样本 Top-10（按 F1）")
            lines.append("")
            lines.extend(self._render_samples_table(worst))
            lines.append("")

            lines.append("## 四、最好样本 Top-10（按 F1）")
            lines.append("")
            lines.extend(self._render_samples_table(best))
            lines.append("")

        # ---- 5. 全部样本（可折叠） ----------------------------------
        lines.append(f"## 五、全部 {n} 条样本")
        lines.append("")
        lines.append("<details><summary>点击展开完整列表</summary>")
        lines.append("")
        lines.extend(self._render_samples_table(rows, trim_question=False))
        lines.append("")
        lines.append("</details>")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------------
    def render_csv(self) -> str:
        rows = self.result.per_example
        fieldnames = [
            "qid", "intent_label", "strategy_name", "question",
            "reference", "prediction",
            "em", "f1", "rouge_l",
            "confidence", "latency_s",
            "prompt_tokens", "completion_tokens", "total_tokens", "llm_calls",
        ]
        if not rows:
            return ",".join(fieldnames) + "\n"
        out_lines = [",".join(fieldnames)]
        for row in rows:
            cells = [self._csv_escape(row.get(k, "")) for k in fieldnames]
            out_lines.append(",".join(cells))
        return "\n".join(out_lines) + "\n"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _csv_escape(val: object) -> str:
        if val is None:
            return ""
        s = str(val)
        if "," in s or '"' in s or "\n" in s:
            s = '"' + s.replace('"', '""') + '"'
        return s

    def _render_samples_table(self, rows: list[dict], trim_question: bool = True) -> list[str]:
        if not rows:
            return ["_(无样本)_"]
        cols = [
            "qid", "intent_label", "strategy_name",
            "question", "reference", "prediction",
            "em", "f1", "latency_s", "total_tokens",
        ]
        cn_names = {
            "qid": "编号", "intent_label": "意图", "strategy_name": "策略",
            "question": "问题", "reference": "参考答案", "prediction": "模型回答",
            "em": "EM", "f1": "F1", "latency_s": "耗时(秒)", "total_tokens": "Token",
        }
        col_present = [c for c in cols if any(c in r_ for r_ in rows)]
        header = [cn_names.get(c, c) for c in col_present]
        out = [
            "| " + " | ".join(header) + " |",
            "|" + "|".join(["---"] * len(col_present)) + "|",
        ]
        for r_ in rows:
            cells: list[str] = []
            for c in col_present:
                v = r_.get(c)
                if v is None:
                    cells.append("")
                elif c == "intent_label" and isinstance(v, str):
                    cells.append(_INTENT_LABEL_ZH.get(v, v))
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}")
                elif c in ("question", "prediction", "reference") and trim_question:
                    s = str(v).replace("\n", " ").replace("|", "\\|")
                    cells.append(s[:60] + ("…" if len(s) > 60 else ""))
                else:
                    cells.append(str(v).replace("\n", " ").replace("|", "\\|"))
            out.append("| " + " | ".join(cells) + " |")
        return out


__all__ = ["EvalReporter"]
