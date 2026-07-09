"""实验 7：路由分布统计（Routing Distribution per Benchmark）

统计 CollaRAG 在各数据集上的意图标签分布。
验证 TreeIntentClassifier 的分类结果与数据集天然意图类型的一致性。

对应文档：docs/experiments/routing_distribution.md

Usage:
    uv run python experiments/exp7_routing_distribution.py
    uv run python experiments/exp7_routing_distribution.py --datasets nq hotpotqa
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    add_common_args,
    collect_predictions,
    make_config,
    render_markdown_table,
    run_inference,
    save_results,
    setup_experiment,
    write_temp_config,
)

logger = logging.getLogger("chimera.exp7")

EXP_NAME = "exp7_routing_distribution"

ALL_DATASETS = ["nq", "popqa", "hotpotqa", "two_wiki", "musique", "asqa"]
INTENT_LABELS = ["greeting", "single_hop", "multi_hop", "summarization", "other"]

DATASET_EXPECTED_TYPE = {
    "nq": "single-hop",
    "popqa": "single-hop",
    "hotpotqa": "multi-hop",
    "two_wiki": "multi-hop",
    "musique": "multi-hop",
    "asqa": "summarization",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 7: Routing Distribution per Benchmark"
    )
    add_common_args(parser)
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)

    cfg = make_config(
        base="colla_rag_only",
        workspace_suffix="exp7_routing",
    )
    config_path = write_temp_config(cfg, "collarag", out_dir)

    distribution: dict[str, dict[str, int]] = {}

    for dataset in args.datasets:
        logger.info("=== Routing Distribution: dataset=%s ===", dataset)

        pred_dir = out_dir / dataset
        try:
            run_inference(
                config_path, dataset, pred_dir,
                limit=args.limit, rebuild=args.rebuild,
            )
            rows, _meta = collect_predictions(pred_dir)

            intent_counts = Counter(
                r.get("intent_label", "unknown") for r in rows
            )
            distribution[dataset] = {
                label: intent_counts.get(label, 0)
                for label in INTENT_LABELS
            }
            distribution[dataset]["total"] = len(rows)

            logger.info("  Distribution: %s", dict(intent_counts))

        except Exception as e:
            logger.error("  FAILED: %s", e)
            distribution[dataset] = {label: 0 for label in INTENT_LABELS}
            distribution[dataset]["total"] = 0

    # Generate count table
    md_path = out_dir / "routing_distribution_report.md"
    with open(md_path, "w") as f:
        f.write("# Routing Distribution per Benchmark\n\n")
        f.write(f"> n={args.limit} per dataset, seed=42, temperature=0, Qwen3-8B backbone\n\n")

        # Count table
        f.write("## Counts\n\n")
        headers = ["Dataset", "Intent Type"] + INTENT_LABELS + ["Total"]
        rows_table = []
        for ds in args.datasets:
            d = distribution.get(ds, {})
            row = [ds, DATASET_EXPECTED_TYPE.get(ds, "?")]
            for label in INTENT_LABELS:
                row.append(str(d.get(label, 0)))
            row.append(str(d.get("total", 0)))
            rows_table.append(row)

        # Overall row
        overall = {label: sum(distribution.get(ds, {}).get(label, 0) for ds in args.datasets) for label in INTENT_LABELS}
        overall_total = sum(distribution.get(ds, {}).get("total", 0) for ds in args.datasets)
        overall_row = ["**Overall**", "—"]
        for label in INTENT_LABELS:
            overall_row.append(str(overall[label]))
        overall_row.append(str(overall_total))
        rows_table.append(overall_row)

        f.write(render_markdown_table(headers, rows_table))
        f.write("\n\n")

        # Percentage table
        f.write("## Percentages\n\n")
        headers = ["Dataset"] + INTENT_LABELS
        rows_pct = []
        for ds in args.datasets:
            d = distribution.get(ds, {})
            total = d.get("total", 1)
            row = [ds]
            for label in INTENT_LABELS:
                pct = d.get(label, 0) / total * 100 if total > 0 else 0
                row.append(f"{pct:.1f}%")
            rows_pct.append(row)

        overall_row = ["**Overall**"]
        for label in INTENT_LABELS:
            pct = overall[label] / overall_total * 100 if overall_total > 0 else 0
            overall_row.append(f"{pct:.1f}%")
        rows_pct.append(overall_row)

        f.write(render_markdown_table(headers, rows_pct))
        f.write("\n")

    save_results(distribution, out_dir, "routing_distribution")
    logger.info("Routing distribution saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
