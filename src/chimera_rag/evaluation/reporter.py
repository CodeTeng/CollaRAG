"""评测报告生成器（中文版）。

提供两个报告器：

* :class:`EvalReporter` ：单次评测报告
  —— 输出 ``<stem>.json`` / ``<stem>.md`` / ``<stem>.csv``，新增『耗时分布』和
     『Token 分布』两节（min/avg/max/P50/P95）。

* :class:`CompareReporter` ：多个 EvalResult 并排对比
  —— 输出 ``compare.md`` / ``compare.json`` / ``compare.csv``；如果
     ``matplotlib`` 可用，额外写出 ``compare_bar.png`` 和 ``compare_radar.png``
     并嵌到 md 里（装了 matplotlib 就有图，没装也不会报错）。

报告生成器零额外运行时强制依赖（matplotlib 为可选）。
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from chimera_rag.core.types import EvalResult

logger = logging.getLogger(__name__)

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
    return f"{n:,}"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)} 分 {s:.1f} 秒"
    h, m = divmod(int(m), 60)
    return f"{h} 时 {m} 分 {s:.0f} 秒"


def _percentile(values: list[float], p: float) -> float:
    """纯 stdlib 的百分位数实现（线性插值）。p ∈ [0, 100]。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def _distribution_stats(values: list[float]) -> dict[str, float]:
    """返回 count / sum / avg / min / max / p50 / p95。"""
    if not values:
        return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0,
                "p50": 0.0, "p95": 0.0}
    return {
        "count": len(values),
        "sum": float(sum(values)),
        "avg": float(statistics.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
    }


# ---------------------------------------------------------------------------
# 单次报告
# ---------------------------------------------------------------------------
class EvalReporter:
    """根据 EvalResult 渲染 json + markdown + csv 三件套。"""

    def __init__(self, result: EvalResult) -> None:
        self.result = result

    # ------------------------------------------------------------------
    def write(self, out_dir: Path, stem: str) -> dict[str, Path]:
        out_dir = Path(out_dir)
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
        lines.append("### 推理阶段")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| 样本数 | {_fmt_int(n)} |")
        if r.elapsed_s:
            lines.append(f"| 总耗时 | {_fmt_duration(r.elapsed_s)} |")
            qps = n / r.elapsed_s if r.elapsed_s > 0 else 0
            lines.append(f"| 吞吐 | {qps:.2f} 题/秒 |")
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

        # ---- 1b. 构图阶段（只在 meta.ingest 存在时显示） ----------
        if r.ingest:
            ing = r.ingest
            ing_usage = ing.get("usage") or {}
            lines.append("### 构图阶段（一次性）")
            lines.append("")
            lines.append("| 指标 | 数值 |")
            lines.append("|---|---|")
            if "documents" in ing:
                lines.append(f"| 文档数 | {_fmt_int(int(ing['documents']))} |")
            if "chunks" in ing:
                lines.append(f"| Chunk 数 | {_fmt_int(int(ing['chunks']))} |")
            if "triples_pruned" in ing:
                raw_t = int(ing.get("triples_raw", ing["triples_pruned"]))
                kept = int(ing["triples_pruned"])
                lines.append(f"| 三元组 | {_fmt_int(raw_t)} 抽取 → {_fmt_int(kept)} 保留 |")
            if ing.get("elapsed_s"):
                lines.append(f"| 构图耗时 | {_fmt_duration(float(ing['elapsed_s']))} |")
            if ing_usage:
                i_calls = int(ing_usage.get("calls", 0))
                i_total = int(ing_usage.get("total_tokens", 0))
                if i_calls:
                    lines.append(f"| LLM 调用次数 | {_fmt_int(i_calls)} |")
                if i_total:
                    lines.append(f"| 合计 Token | {_fmt_int(i_total)} |")
            lines.append("")

            # 构图 + 推理 合计
            total_elapsed = float(ing.get("elapsed_s", 0.0)) + float(r.elapsed_s or 0.0)
            total_calls = int(ing_usage.get("calls", 0)) + int(r.usage.get("calls", 0))
            total_tokens = int(ing_usage.get("total_tokens", 0)) + int(r.usage.get("total_tokens", 0))
            lines.append("### 总计（构图 + 推理）")
            lines.append("")
            lines.append("| 指标 | 构图 | 推理 | 合计 |")
            lines.append("|---|---|---|---|")
            lines.append(
                f"| 耗时 | {_fmt_duration(float(ing.get('elapsed_s', 0.0)))} | "
                f"{_fmt_duration(float(r.elapsed_s or 0.0))} | "
                f"{_fmt_duration(total_elapsed)} |"
            )
            lines.append(
                f"| LLM 调用 | {_fmt_int(int(ing_usage.get('calls', 0)))} | "
                f"{_fmt_int(int(r.usage.get('calls', 0)))} | "
                f"{_fmt_int(total_calls)} |"
            )
            lines.append(
                f"| Token | {_fmt_int(int(ing_usage.get('total_tokens', 0)))} | "
                f"{_fmt_int(int(r.usage.get('total_tokens', 0)))} | "
                f"{_fmt_int(total_tokens)} |"
            )
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

        # ---- 2. 耗时分布 ---------------------------------------------
        latencies = [float(row.get("latency_s", 0.0)) for row in rows]
        lat_stats = _distribution_stats(latencies)
        lines.append("## 二、耗时分布（秒）")
        lines.append("")
        lines.append("| 统计量 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| 平均 | {lat_stats['avg']:.2f} |")
        lines.append(f"| 最小 | {lat_stats['min']:.2f} |")
        lines.append(f"| 最大 | {lat_stats['max']:.2f} |")
        lines.append(f"| P50  | {lat_stats['p50']:.2f} |")
        lines.append(f"| P95  | {lat_stats['p95']:.2f} |")
        lines.append(f"| 合计 | {lat_stats['sum']:.1f} |")
        lines.append("")

        # ---- 3. Token 分布 -------------------------------------------
        lines.append("## 三、Token 分布")
        lines.append("")
        lines.append("| 字段 | 合计 | 平均 | 最小 | 最大 | P50 | P95 |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, label in (
            ("prompt_tokens", "输入 Token"),
            ("completion_tokens", "输出 Token"),
            ("total_tokens", "合计 Token"),
        ):
            vals = [float(row.get(key, 0)) for row in rows]
            s = _distribution_stats(vals)
            lines.append(
                f"| {label} | {_fmt_int(int(s['sum']))} | {s['avg']:.1f} | "
                f"{int(s['min'])} | {int(s['max'])} | "
                f"{s['p50']:.0f} | {s['p95']:.0f} |"
            )
        lines.append("")

        # ---- 4. 按意图分组 -------------------------------------------
        intents_present = [row.get("intent_label") for row in rows if row.get("intent_label")]
        if intents_present:
            lines.append("## 四、按意图分组")
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

            # ---- 4b. 策略分布 ---------------------------------------
            strategies = Counter(row.get("strategy_name") for row in rows if row.get("strategy_name"))
            if strategies:
                lines.append("### 策略分布")
                lines.append("")
                lines.append("| 策略 | 命中次数 |")
                lines.append("|---|---|")
                for s_name, c in strategies.most_common():
                    lines.append(f"| `{s_name}` | {c} |")
                lines.append("")

        # ---- 5. Top-10 最差 / 最好 ----------------------------------
        if rows and "f1" in rows[0]:
            sorted_by_f1 = sorted(rows, key=lambda r_: r_["f1"])
            worst = sorted_by_f1[:10]
            best = list(reversed(sorted_by_f1[-10:]))

            lines.append("## 五、最差样本 Top-10（按 F1）")
            lines.append("")
            lines.extend(self._render_samples_table(worst))
            lines.append("")

            lines.append("## 六、最好样本 Top-10（按 F1）")
            lines.append("")
            lines.extend(self._render_samples_table(best))
            lines.append("")

        # ---- 6. 全部样本（可折叠） ----------------------------------
        lines.append(f"## 七、全部 {n} 条样本")
        lines.append("")
        lines.append("<details><summary>点击展开完整列表</summary>")
        lines.append("")
        lines.extend(self._render_samples_table(rows, trim_question=False))
        lines.append("")
        lines.append("</details>")
        lines.append("")

        return "\n".join(lines)

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


# ---------------------------------------------------------------------------
# 多目录对比
# ---------------------------------------------------------------------------
class CompareReporter:
    """把多个 EvalResult 并排对比；支持表格 + 柱图 + 雷达图。"""

    def __init__(self, results: list[EvalResult]) -> None:
        if not results:
            raise ValueError("CompareReporter requires at least one EvalResult")
        self.results = results

    # ------------------------------------------------------------------
    def write(self, out_dir: Path) -> dict[str, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {
            "json": out_dir / "compare.json",
            "md":   out_dir / "compare.md",
            "csv":  out_dir / "compare.csv",
        }
        paths["json"].write_text(
            json.dumps(
                [r.model_dump() for r in self.results],
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths["csv"].write_text(self.render_csv(), encoding="utf-8")

        # 画图（可选）
        chart_paths: dict[str, Path] = {}
        try:
            chart_paths = self._render_charts(out_dir)
            paths.update(chart_paths)
        except Exception as e:  # pragma: no cover - matplotlib 问题不影响报告
            logger.warning("matplotlib 图表渲染失败：%s；仅输出表格", e)

        paths["md"].write_text(self.render_markdown(chart_paths), encoding="utf-8")
        return paths

    # ------------------------------------------------------------------
    def render_markdown(self, chart_paths: dict[str, Path] | None = None) -> str:
        chart_paths = chart_paths or {}
        results = self.results
        metric_keys = self._union_metric_keys()

        lines: list[str] = []
        lines.append("# 多配置对比报告")
        lines.append("")
        lines.append(f"参与对比的配置数：**{len(results)}**")
        lines.append("")

        # ---- 总览 ----
        lines.append("## 一、总览")
        lines.append("")
        header = [
            "配置", "样本数", "推理耗时", "推理 LLM 调用", "推理 Token",
            "构图耗时", "构图 Token", "吞吐(题/秒)",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in results:
            n = len(r.per_example)
            total_t = int(r.usage.get("total_tokens", 0))
            calls = int(r.usage.get("calls", 0))
            qps = n / r.elapsed_s if r.elapsed_s > 0 else 0.0
            if r.ingest:
                ing_elapsed = float(r.ingest.get("elapsed_s", 0.0))
                ing_tok = int((r.ingest.get("usage") or {}).get("total_tokens", 0))
                ing_elapsed_s = _fmt_duration(ing_elapsed) if ing_elapsed else "-"
                ing_tok_s = _fmt_int(ing_tok) if ing_tok else "-"
            else:
                ing_elapsed_s = "-"
                ing_tok_s = "-"
            lines.append(
                f"| `{r.config_name}` | {n} | {_fmt_duration(r.elapsed_s)} | "
                f"{_fmt_int(calls)} | {_fmt_int(total_t)} | "
                f"{ing_elapsed_s} | {ing_tok_s} | {qps:.2f} |"
            )
        lines.append("")

        # ---- 指标对比 ----
        lines.append("## 二、指标对比")
        lines.append("")
        header = ["配置", *[
            _METRIC_LABEL_ZH.get(k, k.replace("_", " ").title())
            for k in metric_keys
        ]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in results:
            cells = [f"`{r.config_name}`"]
            for k in metric_keys:
                v = r.metrics.get(k)
                if v is None:
                    cells.append("-")
                elif k == "avg_latency_s":
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(f"{v:.4f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        if "bar" in chart_paths:
            lines.append(f"![指标并排柱图]({chart_paths['bar'].name})")
            lines.append("")
        if "radar" in chart_paths:
            lines.append(f"![指标雷达图]({chart_paths['radar'].name})")
            lines.append("")

        # ---- 耗时分布对比 ----
        lines.append("## 三、耗时分布对比（秒）")
        lines.append("")
        lines.append("| 配置 | 平均 | 最小 | 最大 | P50 | P95 |")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            lats = [float(row.get("latency_s", 0.0)) for row in r.per_example]
            s = _distribution_stats(lats)
            lines.append(
                f"| `{r.config_name}` | {s['avg']:.2f} | {s['min']:.2f} | "
                f"{s['max']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} |"
            )
        lines.append("")

        # ---- Token 分布对比 ----
        lines.append("## 四、Token 分布对比（total_tokens）")
        lines.append("")
        lines.append("| 配置 | 合计 | 平均 | P50 | P95 |")
        lines.append("|---|---|---|---|---|")
        for r in results:
            toks = [float(row.get("total_tokens", 0)) for row in r.per_example]
            s = _distribution_stats(toks)
            lines.append(
                f"| `{r.config_name}` | {_fmt_int(int(s['sum']))} | "
                f"{s['avg']:.1f} | {s['p50']:.0f} | {s['p95']:.0f} |"
            )
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def render_csv(self) -> str:
        metric_keys = self._union_metric_keys()
        header = [
            "config", "n_examples", "elapsed_s", "total_tokens", "calls",
            *metric_keys,
            "latency_avg", "latency_p50", "latency_p95",
            "token_avg", "token_p50", "token_p95",
        ]
        rows = [",".join(header)]
        for r in self.results:
            lats = [float(row.get("latency_s", 0.0)) for row in r.per_example]
            toks = [float(row.get("total_tokens", 0)) for row in r.per_example]
            ls = _distribution_stats(lats)
            ts = _distribution_stats(toks)
            line = [
                EvalReporter._csv_escape(r.config_name),
                str(len(r.per_example)),
                f"{r.elapsed_s:.3f}",
                str(int(r.usage.get("total_tokens", 0))),
                str(int(r.usage.get("calls", 0))),
                *[
                    (f"{r.metrics[k]:.4f}" if k in r.metrics else "")
                    for k in metric_keys
                ],
                f"{ls['avg']:.4f}", f"{ls['p50']:.4f}", f"{ls['p95']:.4f}",
                f"{ts['avg']:.2f}", f"{ts['p50']:.2f}", f"{ts['p95']:.2f}",
            ]
            rows.append(",".join(line))
        return "\n".join(rows) + "\n"

    # ------------------------------------------------------------------
    def _union_metric_keys(self) -> list[str]:
        """保持稳定顺序（按首次出现）的全量 metric key 合集。"""
        seen: list[str] = []
        for r in self.results:
            for k in r.metrics:
                if k not in seen:
                    seen.append(k)
        return seen

    # ------------------------------------------------------------------
    def _render_charts(self, out_dir: Path) -> dict[str, Path]:
        """绘制并排柱图和雷达图；matplotlib 不可用时返回空 dict。"""
        try:
            import matplotlib  # type: ignore

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            logger.info("matplotlib 未安装，跳过图表渲染（只输出表格）")
            return {}

        # 中文字体：尽力配置，不可用则回退（标签带中文会显示为方块，但不报错）
        try:
            matplotlib.rcParams["font.sans-serif"] = [
                "PingFang SC", "Heiti SC", "Hiragino Sans GB",
                "Microsoft YaHei", "SimHei", "Arial Unicode MS",
                "DejaVu Sans",
            ]
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception:
            pass

        # 只挑"越大越好"的指标做图；avg_latency_s 另行处理
        metric_keys = [k for k in self._union_metric_keys() if k != "avg_latency_s"]
        if not metric_keys:
            return {}

        config_names = [r.config_name for r in self.results]
        matrix: list[list[float]] = []
        for k in metric_keys:
            matrix.append([float(r.metrics.get(k, 0.0)) for r in self.results])

        chart_paths: dict[str, Path] = {}

        # ---- 并排柱图 ----
        bar_path = out_dir / "compare_bar.png"
        fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(config_names) + 2), 4.2))
        n_metrics = len(metric_keys)
        n_cfg = len(config_names)
        group_w = 0.8
        bar_w = group_w / max(n_metrics, 1)
        for i, k in enumerate(metric_keys):
            xs = [
                j - group_w / 2 + bar_w * (i + 0.5)
                for j in range(n_cfg)
            ]
            ax.bar(
                xs, matrix[i], width=bar_w,
                label=_METRIC_LABEL_ZH.get(k, k),
            )
        ax.set_xticks(range(n_cfg))
        ax.set_xticklabels(config_names, rotation=15, ha="right")
        ax.set_ylabel("score")
        ax.set_ylim(0, max(1.0, max((max(row) for row in matrix), default=1.0) * 1.1))
        ax.set_title("Metric comparison across configs")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        chart_paths["bar"] = bar_path

        # ---- 雷达图（至少 3 个指标才画） ----
        if n_metrics >= 3:
            try:
                import math

                radar_path = out_dir / "compare_radar.png"
                angles = [2 * math.pi * i / n_metrics for i in range(n_metrics)]
                angles_closed = [*angles, angles[0]]

                fig = plt.figure(figsize=(5.5, 5.5))
                ax = fig.add_subplot(111, polar=True)
                for c_idx, cfg_name in enumerate(config_names):
                    values = [matrix[m_idx][c_idx] for m_idx in range(n_metrics)]
                    values_closed = [*values, values[0]]
                    ax.plot(angles_closed, values_closed, linewidth=1.5, label=cfg_name)
                    ax.fill(angles_closed, values_closed, alpha=0.12)
                ax.set_xticks(angles)
                ax.set_xticklabels([_METRIC_LABEL_ZH.get(k, k) for k in metric_keys])
                ax.set_ylim(0, max(1.0, max((max(row) for row in matrix), default=1.0) * 1.05))
                ax.set_title("Radar: metric profile")
                ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
                fig.tight_layout()
                fig.savefig(radar_path, dpi=150)
                plt.close(fig)
                chart_paths["radar"] = radar_path
            except Exception as e:  # pragma: no cover
                logger.warning("雷达图渲染失败：%s", e)

        return chart_paths


__all__ = ["CompareReporter", "EvalReporter"]
