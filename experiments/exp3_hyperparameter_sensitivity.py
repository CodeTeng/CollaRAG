"""实验 3：超参数敏感性分析（Hyperparameter Sensitivity）

在 HotpotQA 上扫描 5 个关键超参数，验证默认值落在最佳范围内：
  1. max_plan_steps: [3, 5, 7, 10]
  2. quality_threshold τ: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  3. reflect_retries: [0, 1, 2, 3]
  4. max_step_iterations: [1, 2, 3, 4]
  5. hybrid_search_top_k: [5, 10, 15, 20]

对应文档：docs/experiments/hyperparameter-sensitivity.md

Usage:
    uv run python experiments/exp3_hyperparameter_sensitivity.py
    uv run python experiments/exp3_hyperparameter_sensitivity.py --param max_plan_steps
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    add_common_args,
    collect_predictions,
    compute_metrics,
    make_config,
    render_markdown_table,
    run_inference,
    save_results,
    setup_experiment,
    write_temp_config,
)

logger = logging.getLogger("chimera.exp3")

EXP_NAME = "exp3_hyperparameter"
EVAL_DATASET = "hotpotqa"

PARAM_SWEEPS = {
    "max_plan_steps": {
        "values": [3, 5, 7, 10],
        "default": 5,
        "config_path": ["plugins", "colla_rag", "multi_agent", "react", "max_plan_steps"],
        "description": "MultiHop Agent Plan 阶段最大步骤数",
    },
    "quality_threshold": {
        "values": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "default": 0.6,
        "config_path": ["plugins", "colla_rag", "multi_agent", "react", "quality_threshold"],
        "description": "Reflect 阶段触发阈值 τ",
    },
    "reflect_retries": {
        "values": [0, 1, 2, 3],
        "default": 2,
        "config_path": ["plugins", "colla_rag", "multi_agent", "react", "reflect_retries"],
        "description": "Reflect 阶段最大重试次数",
    },
    "max_step_iterations": {
        "values": [1, 2, 3, 4],
        "default": 3,
        "config_path": ["plugins", "colla_rag", "multi_agent", "react", "max_step_iterations"],
        "description": "每步 ReAct 循环最大迭代次数",
    },
    "hybrid_search_top_k": {
        "values": [5, 10, 15, 20],
        "default": 10,
        "config_path": ["plugins", "colla_rag", "multi_agent", "native_rag", "top_k"],
        "description": "hybrid_search 检索 chunk 数量",
    },
}


def _set_nested(d: dict, path: list[str], value) -> dict:
    current = d
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value
    return d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Hyperparameter Sensitivity"
    )
    add_common_args(parser)
    parser.add_argument(
        "--param", nargs="+", default=list(PARAM_SWEEPS.keys()),
        choices=list(PARAM_SWEEPS.keys()),
        help="Parameters to sweep",
    )
    parser.add_argument(
        "--dataset", default=EVAL_DATASET,
        help=f"Dataset for evaluation (default: {EVAL_DATASET})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)
    metrics_list = [m.strip() for m in args.metrics.split(",")]

    all_results: dict[str, list[dict]] = {}

    for param_name in args.param:
        sweep = PARAM_SWEEPS[param_name]
        logger.info("=== Sweeping: %s (%s) ===", param_name, sweep["description"])

        param_results = []

        for value in sweep["values"]:
            logger.info("  %s = %s%s", param_name, value,
                        " (default)" if value == sweep["default"] else "")

            overrides = {}
            _set_nested(overrides, sweep["config_path"], value)

            cfg = make_config(
                base="colla_rag_only",
                overrides=overrides,
                workspace_suffix=f"exp3_{param_name}_{value}",
            )

            config_path = write_temp_config(
                cfg, f"{param_name}_{value}", out_dir,
            )

            pred_dir = out_dir / param_name / str(value)

            try:
                run_inference(
                    config_path, args.dataset, pred_dir,
                    limit=args.limit, rebuild=args.rebuild,
                )
                rows, meta = collect_predictions(pred_dir)
                scores = compute_metrics(rows, metrics_list)

                # Compute additional diagnostics from predictions
                avg_latency = (
                    sum(r.get("latency_s", 0) for r in rows) / len(rows)
                    if rows else 0
                )

                result_entry = {
                    "value": value,
                    "is_default": value == sweep["default"],
                    **scores,
                    "avg_latency_s": round(avg_latency, 3),
                    "n_samples": len(rows),
                }
                param_results.append(result_entry)
                logger.info("    Scores: %s", scores)

            except Exception as e:
                logger.error("    FAILED: %s", e)
                param_results.append({
                    "value": value,
                    "is_default": value == sweep["default"],
                    **{m: -1 for m in metrics_list},
                })

        all_results[param_name] = param_results

        # Generate per-parameter table
        headers = [param_name] + [m.upper() for m in metrics_list] + ["Avg Latency(s)"]
        table_rows = []
        for entry in param_results:
            val_str = f"**{entry['value']} (default)**" if entry["is_default"] else str(entry["value"])
            row = [val_str]
            for m in metrics_list:
                score = entry.get(m, -1)
                s = f"{score:.3f}" if score >= 0 else "N/A"
                if entry["is_default"]:
                    s = f"**{s}**"
                row.append(s)
            row.append(f"{entry.get('avg_latency_s', 0):.3f}")
            table_rows.append(row)

        table = render_markdown_table(headers, table_rows)
        param_dir = out_dir / param_name
        param_dir.mkdir(parents=True, exist_ok=True)
        with open(param_dir / "table.md", "w") as f:
            f.write(f"## {param_name}: {sweep['description']}\n\n")
            f.write(table)
            f.write("\n")

    # Summary table
    _generate_summary(all_results, out_dir)
    save_results(all_results, out_dir, "hyperparameter_results")
    logger.info("All hyperparameter results saved to %s", out_dir)
    return 0


def _generate_summary(results: dict, out_dir: Path) -> None:
    md_path = out_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("# Hyperparameter Sensitivity Summary\n\n")
        f.write(f"> Dataset: {EVAL_DATASET}, Backbone: Qwen3-8B\n\n")

        headers = ["Parameter", "Default", "Best Range", "Sensitivity"]
        rows = []
        for param_name, sweep in PARAM_SWEEPS.items():
            param_results = results.get(param_name, [])
            if param_results:
                ems = [(r["value"], r.get("em", 0)) for r in param_results if r.get("em", -1) >= 0]
                if ems:
                    best_val, best_em = max(ems, key=lambda x: x[1])
                    worst_val, worst_em = min(ems, key=lambda x: x[1])
                    spread = best_em - worst_em
                    sensitivity = "Low" if spread < 0.02 else "Medium" if spread < 0.04 else "High"
                    rows.append([
                        param_name,
                        str(sweep["default"]),
                        f"{best_val}",
                        sensitivity,
                    ])

        f.write(render_markdown_table(headers, rows))
        f.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
