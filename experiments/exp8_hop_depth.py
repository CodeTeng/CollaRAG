"""实验 8：多跳深度分析（Hop Depth Analysis）

分析 CollaRAG 在 MuSiQue 数据集上按 2/3/4 跳深度拆分的表现，
揭示 PER 架构在极端多跳场景下的能力边界和衰减率。

对应文档：docs/experiments/hop-depth-analysis.md

Usage:
    uv run python experiments/exp8_hop_depth.py
    uv run python experiments/exp8_hop_depth.py --methods collarag hipporag2 --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    PROJECT_ROOT,
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

logger = logging.getLogger("chimera.exp8")

EXP_NAME = "exp8_hop_depth"
DATASET = "musique"

METHODS = {
    "vanilla_rag": {
        "base_config": "vanilla",
        "overrides": {},
    },
    "raptor": {
        "base_config": "vanilla",
        "overrides": {
            "query": {"retriever": {"params": {"top_k": 10, "neighbor_hops": 2}}},
        },
    },
    "graphrag": {
        "base_config": "vanilla",
        "overrides": {
            "query": {"retriever": {"params": {"top_k": 10, "neighbor_hops": 2, "use_community": True}}},
        },
    },
    "hipporag": {
        "base_config": "vanilla",
        "overrides": {
            "query": {"retriever": {"params": {"top_k": 10, "neighbor_hops": 2, "use_ppr": True}}},
        },
    },
    "hipporag2": {
        "base_config": "vanilla",
        "overrides": {
            "query": {"retriever": {"params": {"top_k": 10, "neighbor_hops": 3, "use_ppr": True}}},
        },
    },
    "collarag": {
        "base_config": "colla_rag_only",
        "overrides": {},
    },
}


def load_hop_labels(dataset: str = DATASET) -> dict[str, int]:
    """Load hop depth labels from the dataset's qa.jsonl (MuSiQue includes hop metadata)."""
    qa_path = PROJECT_ROOT / "data" / dataset / "qa.jsonl"
    hop_labels = {}

    if not qa_path.exists():
        logger.warning("QA file not found: %s", qa_path)
        return hop_labels

    with open(qa_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("qid") or row.get("id", "")
            num_hops = row.get("num_hops") or row.get("hops") or row.get("n_hops")
            if num_hops is not None:
                hop_labels[str(qid)] = int(num_hops)
            else:
                decomposition = row.get("question_decomposition") or row.get("decomposition", [])
                if decomposition:
                    hop_labels[str(qid)] = len(decomposition)

    logger.info("Loaded hop labels for %d questions", len(hop_labels))
    return hop_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 8: Hop Depth Analysis"
    )
    add_common_args(parser)
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)

    hop_labels = load_hop_labels()

    results_by_hop: dict[str, dict[int, dict[str, float]]] = {}
    results_overall: dict[str, dict[str, float]] = {}

    for method_name in args.methods:
        method = METHODS[method_name]
        logger.info("=== Hop Depth: method=%s ===", method_name)

        cfg = make_config(
            base=method["base_config"],
            overrides=method.get("overrides", {}),
            workspace_suffix=f"exp8_{method_name}",
        )
        config_path = write_temp_config(cfg, method_name, out_dir)
        pred_dir = out_dir / method_name

        try:
            run_inference(
                config_path, DATASET, pred_dir,
                limit=args.limit, rebuild=args.rebuild,
            )
            rows, _meta = collect_predictions(pred_dir)

            # Overall score
            results_overall[method_name] = compute_metrics(rows, ["em"])

            # Split by hop depth
            by_hop: dict[int, list[dict]] = defaultdict(list)
            for row in rows:
                qid = str(row.get("qid", ""))
                hops = hop_labels.get(qid, 0)
                if hops >= 2:
                    by_hop[hops].append(row)

            results_by_hop[method_name] = {}
            for hop_n in sorted(by_hop.keys()):
                scores = compute_metrics(by_hop[hop_n], ["em"])
                results_by_hop[method_name][hop_n] = {
                    **scores,
                    "n_samples": len(by_hop[hop_n]),
                }
                logger.info("  %d-hop (n=%d): EM=%.3f", hop_n, len(by_hop[hop_n]), scores["em"])

        except Exception as e:
            logger.error("  FAILED: %s", e)
            results_overall[method_name] = {"em": -1}
            results_by_hop[method_name] = {}

    # Generate report
    _generate_report(results_by_hop, results_overall, args.methods, out_dir)

    save_results(
        {"by_hop": {m: {str(k): v for k, v in hops.items()} for m, hops in results_by_hop.items()},
         "overall": results_overall},
        out_dir, "hop_depth_results",
    )
    logger.info("Hop depth analysis saved to %s", out_dir)
    return 0


def _generate_report(
    by_hop: dict[str, dict[int, dict]],
    overall: dict[str, dict],
    methods: list[str],
    out_dir: Path,
) -> None:
    md_path = out_dir / "hop_depth_report.md"
    all_hops = sorted({h for m in by_hop.values() for h in m})
    if not all_hops:
        all_hops = [2, 3, 4]

    with open(md_path, "w") as f:
        f.write("# Hop Depth Analysis Report\n\n")
        f.write(f"> Dataset: MuSiQue-Answerable, Backbone: Qwen3-8B\n\n")

        # Performance table
        f.write("## 1. EM by Hop Depth\n\n")
        headers = ["Method"] + [f"{h}-hop" for h in all_hops] + ["Overall"]
        rows = []
        for method in methods:
            row = [method]
            for h in all_hops:
                em = by_hop.get(method, {}).get(h, {}).get("em", -1)
                row.append(f"{em:.3f}" if em >= 0 else "N/A")
            ov = overall.get(method, {}).get("em", -1)
            row.append(f"{ov:.3f}" if ov >= 0 else "N/A")
            rows.append(row)
        f.write(render_markdown_table(headers, rows))
        f.write("\n\n")

        # Decay rate table
        f.write("## 2. Decay Rates\n\n")
        if len(all_hops) >= 2:
            headers = ["Method"]
            for i in range(len(all_hops) - 1):
                headers.append(f"{all_hops[i]}-hop -> {all_hops[i+1]}-hop")
            if len(all_hops) >= 3:
                headers.append(f"{all_hops[0]}-hop -> {all_hops[-1]}-hop Total")

            rows = []
            for method in methods:
                row = [method]
                hop_ems = []
                for h in all_hops:
                    hop_ems.append(by_hop.get(method, {}).get(h, {}).get("em", 0))

                for i in range(len(all_hops) - 1):
                    if hop_ems[i] > 0:
                        decay = (hop_ems[i+1] - hop_ems[i]) / hop_ems[i] * 100
                        row.append(f"{decay:+.1f}%")
                    else:
                        row.append("N/A")

                if len(all_hops) >= 3 and hop_ems[0] > 0:
                    total_decay = (hop_ems[-1] - hop_ems[0]) / hop_ems[0] * 100
                    row.append(f"{total_decay:+.1f}%")
                elif len(all_hops) >= 3:
                    row.append("N/A")

                rows.append(row)
            f.write(render_markdown_table(headers, rows))
            f.write("\n")

    logger.info("Report written to %s", md_path)


if __name__ == "__main__":
    raise SystemExit(main())
