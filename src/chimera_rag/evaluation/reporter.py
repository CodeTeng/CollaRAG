"""Evaluation report writer.

Takes an :class:`EvalResult` and renders:

* ``<stem>.json`` — the full EvalResult (model_dump); machine-readable
* ``<stem>.md``   — a human-readable Markdown report with sections:
    1. Summary (aggregated metrics + average latency)
    2. Intent breakdown (only shown when IntentAgent labels are present)
    3. Top 10 best / worst samples by F1
    4. All samples (collapsible table)
* ``<stem>.csv``  — per-example table for spreadsheet analysis

The reporter is deliberately standalone — no CSV library, no Jinja; just
string formatting — so it runs on a bare install without extra deps.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from chimera_rag.core.types import EvalResult


class EvalReporter:
    """Render a human-readable report next to the JSON output."""

    def __init__(self, result: EvalResult) -> None:
        self.result = result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write(self, out_dir: Path, stem: str) -> dict[str, Path]:
        """Write JSON + Markdown + CSV under ``out_dir``. Returns the paths."""
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
    # Markdown
    # ------------------------------------------------------------------
    def render_markdown(self) -> str:
        r = self.result
        rows = r.per_example
        n = len(rows)

        lines: list[str] = []
        lines.append(f"# Evaluation report — `{r.config_name}`")
        lines.append("")
        lines.append(f"**Samples:** {n}")
        lines.append("")

        # ---- 1. Summary metrics -------------------------------------
        lines.append("## 1. Aggregate metrics")
        lines.append("")
        lines.append("| Metric | Score |")
        lines.append("|---|---|")
        for k, v in r.metrics.items():
            label = k.upper() if len(k) <= 3 else k.replace("_", " ").title()
            lines.append(f"| {label} | {v:.4f} |")
        lines.append("")

        # ---- 2. Intent breakdown (if recorded) ----------------------
        intents_present = [row.get("intent_label") for row in rows if row.get("intent_label")]
        if intents_present:
            lines.append("## 2. Breakdown by intent")
            lines.append("")
            bucket: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                bucket[row.get("intent_label") or "(none)"].append(row)
            metric_keys = [k for k in r.metrics if k != "avg_latency_s"]
            header = ["Intent", "Count"] + [k.upper() if len(k) <= 3 else k for k in metric_keys] + ["Avg latency (s)"]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for label in sorted(bucket):
                grp = bucket[label]
                avg_latency = statistics.mean(r_["latency_s"] for r_ in grp) if grp else 0.0
                metric_cells = []
                for mk in metric_keys:
                    vals = [r_.get(mk, 0.0) for r_ in grp]
                    metric_cells.append(f"{statistics.mean(vals):.3f}" if vals else "-")
                lines.append(
                    "| " + " | ".join([label, str(len(grp)), *metric_cells, f"{avg_latency:.2f}"]) + " |"
                )
            lines.append("")

            # ---- 2b. Strategy distribution --------------------------
            strategies = Counter(row.get("strategy_name") for row in rows if row.get("strategy_name"))
            if strategies:
                lines.append("### Strategy distribution")
                lines.append("")
                lines.append("| Strategy | Count |")
                lines.append("|---|---|")
                for s, c in strategies.most_common():
                    lines.append(f"| {s} | {c} |")
                lines.append("")

        # ---- 3. Top 10 best / worst by F1 ---------------------------
        if rows and "f1" in rows[0]:
            sorted_by_f1 = sorted(rows, key=lambda r_: r_["f1"])
            worst = sorted_by_f1[:10]
            best = list(reversed(sorted_by_f1[-10:]))

            lines.append("## 3. Top-10 worst samples (by F1)")
            lines.append("")
            lines.extend(self._render_samples_table(worst))
            lines.append("")

            lines.append("## 4. Top-10 best samples (by F1)")
            lines.append("")
            lines.extend(self._render_samples_table(best))
            lines.append("")

        # ---- 5. All samples -----------------------------------------
        lines.append(f"## 5. All {n} samples")
        lines.append("")
        lines.append("<details><summary>Click to expand full table</summary>")
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
        if not rows:
            return "qid,question,reference,prediction,em,f1,latency_s\n"
        # Use a consistent column order across runs.
        fieldnames = [
            "qid", "intent_label", "strategy_name", "question",
            "reference", "prediction",
            "em", "f1", "rouge_l",
            "confidence", "latency_s",
        ]
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
            return ["_(no samples)_"]
        cols = ["qid", "intent_label", "strategy_name", "question", "reference", "prediction", "em", "f1", "latency_s"]
        # Detect which columns are actually present.
        col_present = [c for c in cols if any(c in r_ for r_ in rows)]
        header = [c.replace("_", " ").title() for c in col_present]
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join(["---"] * len(col_present)) + "|"]
        for r_ in rows:
            cells: list[str] = []
            for c in col_present:
                v = r_.get(c)
                if v is None:
                    cells.append("")
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}")
                elif c in ("question", "prediction", "reference") and trim_question:
                    s = str(v).replace("\n", " ").replace("|", "\\|")
                    cells.append(s[:80] + ("…" if len(s) > 80 else ""))
                else:
                    cells.append(str(v).replace("\n", " ").replace("|", "\\|"))
            out.append("| " + " | ".join(cells) + " |")
        return out


__all__ = ["EvalReporter"]
