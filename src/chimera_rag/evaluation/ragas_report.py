"""RAGAS 评测报告生成器（中文版）。

把 :class:`chimera_rag.evaluation.ragas_eval.RagasResult` 渲染为
``ragas_report.json`` / ``ragas_report.md`` / ``ragas_report.csv`` 三件套，
风格对齐 :class:`chimera_rag.evaluation.reporter.EvalReporter`。

零额外运行时依赖（csv 用 stdlib）。
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chimera_rag.evaluation.ragas_eval import RagasResult

logger = logging.getLogger(__name__)

# RAGAS 指标的中文标签
_RAGAS_LABEL_ZH = {
    "faithfulness": "忠实度 Faithfulness",
    "answer_relevancy": "答案相关性 Answer Relevancy",
    "context_precision": "上下文精确率 Context Precision",
    "context_recall": "上下文召回率 Context Recall",
}


class RagasReporter:
    """根据 RagasResult 渲染 json + markdown + csv 三件套。"""

    STEM = "ragas_report"

    def __init__(self, result: RagasResult) -> None:
        self.result = result

    # ------------------------------------------------------------------
    def write(self, out_dir: Path, stem: str | None = None) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or self.STEM
        paths = {
            "json": out_dir / f"{stem}.json",
            "md": out_dir / f"{stem}.md",
            "csv": out_dir / f"{stem}.csv",
        }
        paths["json"].write_text(self.render_json(), encoding="utf-8")
        paths["md"].write_text(self.render_markdown(), encoding="utf-8")
        paths["csv"].write_text(self.render_csv(), encoding="utf-8")
        return paths

    # ------------------------------------------------------------------
    def render_json(self) -> str:
        r = self.result
        payload = {
            "config_name": r.config_name,
            "n_examples": r.n_examples,
            "metrics": r.metrics,
            "skipped_metrics": r.skipped_metrics,
            "per_example": r.per_example,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    def render_markdown(self) -> str:
        r = self.result
        lines: list[str] = []
        lines.append(f"# RAGAS 评测报告 · `{r.config_name}`")
        lines.append("")
        lines.append("## 一、综合指标")
        lines.append("")
        lines.append(f"- 样本数：{r.n_examples}")
        if r.skipped_metrics:
            lines.append(
                f"- 已跳过（缺少 context 原文）：{', '.join(r.skipped_metrics)}"
            )
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---|")
        for key, value in r.metrics.items():
            label = _RAGAS_LABEL_ZH.get(key, key.replace("_", " ").title())
            lines.append(f"| {label} | {value:.4f} |")
        lines.append("")

        # ---- 逐条明细 -------------------------------------------------
        if r.per_example:
            lines.append("## 二、逐条明细")
            lines.append("")
            metric_keys = [k for k in r.metrics]
            header = ["#", "问题", *metric_keys]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "---|" * len(header))
            for i, row in enumerate(r.per_example):
                question = str(
                    row.get("user_input") or row.get("question") or ""
                )
                if len(question) > 40:
                    question = question[:37] + "..."
                cells = [str(i + 1), question]
                for k in metric_keys:
                    v = row.get(k)
                    cells.append(f"{float(v):.4f}" if _is_number(v) else "-")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def render_csv(self) -> str:
        r = self.result
        buf = io.StringIO()
        writer = csv.writer(buf)
        metric_keys = list(r.metrics.keys())
        writer.writerow(["index", "question", *metric_keys])
        for i, row in enumerate(r.per_example):
            question = str(row.get("user_input") or row.get("question") or "")
            cells = [i + 1, question]
            for k in metric_keys:
                v = row.get(k)
                cells.append(f"{float(v):.6f}" if _is_number(v) else "")
            writer.writerow(cells)
        # 末尾追加聚合行
        agg_cells = ["AVG", ""]
        for k in metric_keys:
            agg_cells.append(f"{r.metrics[k]:.6f}")
        writer.writerow(agg_cells)
        return buf.getvalue()


def _is_number(v: object) -> bool:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return f == f  # NaN != NaN


__all__ = ["RagasReporter"]
