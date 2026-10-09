"""实验 6：路由反事实实验（Routing Counterfactual）

验证性能提升来自路由决策正确，而非某个 Agent 本身更强。
四种路由策略：Normal / All->SingleHop / All->MultiHop / Random

对应文档：docs/experiments/routing-counterfactual.md

Usage:
    uv run python experiments/exp6_routing_counterfactual.py
    uv run python experiments/exp6_routing_counterfactual.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
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

logger = logging.getLogger("chimera.exp6")

EXP_NAME = "exp6_routing_counterfactual"

ROUTING_STRATEGIES = {
    "normal": {
        "description": "Normal routing (TreeIntentClassifier)",
        "overrides": {},
    },
    "all_single_hop": {
        "description": "All -> SingleHop",
        "overrides": {
            "plugins": {
                "colla_rag": {
                    "multi_agent": {
                        "classifier": {"force_intent": "single_hop"},
                    },
                },
            },
        },
    },
    "all_multi_hop": {
        "description": "All -> MultiHop",
        "overrides": {
            "plugins": {
                "colla_rag": {
                    "multi_agent": {
                        "classifier": {"force_intent": "multi_hop"},
                    },
                },
            },
        },
    },
    "random": {
        "description": "Random routing",
        "overrides": {
            "plugins": {
                "colla_rag": {
                    "multi_agent": {
                        "classifier": {"force_intent": "random"},
                    },
                },
            },
        },
    },
}

MIXED_DATASETS = {
    "nq": {"type": "single_hop", "sample_n": 100},
    "popqa": {"type": "single_hop", "sample_n": 100},
    "two_wiki": {"type": "multi_hop", "sample_n": 100},
    "hotpotqa": {"type": "multi_hop", "sample_n": 100},
    "musique": {"type": "multi_hop", "sample_n": 200},
    "asqa": {"type": "summarization", "sample_n": 100},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 6: Routing Counterfactual"
    )
    add_common_args(parser)
    parser.add_argument(
        "--strategies", nargs="+", default=list(ROUTING_STRATEGIES.keys()),
        choices=list(ROUTING_STRATEGIES.keys()),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)

    overall_results: dict[str, dict[str, Any]] = {}
    per_type_results: dict[str, dict[str, dict[str, float]]] = {}

    for strategy_name in args.strategies:
        strategy = ROUTING_STRATEGIES[strategy_name]
        logger.info("=== Strategy: %s ===", strategy["description"])

        cfg = make_config(
            base="colla_rag_only",
            overrides=strategy.get("overrides", {}),
            workspace_suffix=f"exp6_{strategy_name}",
        )
        config_path = write_temp_config(cfg, strategy_name, out_dir)

        all_predictions = []
        for ds_name, ds_info in MIXED_DATASETS.items():
            pred_dir = out_dir / strategy_name / ds_name
            try:
                run_inference(
                    config_path, ds_name, pred_dir,
                    limit=ds_info["sample_n"], rebuild=args.rebuild,
                )
                rows, _meta = collect_predictions(pred_dir)
                for r in rows:
                    r["dataset"] = ds_name
                    r["query_type"] = ds_info["type"]
                all_predictions.extend(rows)
            except Exception as e:
                logger.error("  FAILED for %s: %s", ds_name, e)

        overall = compute_metrics(all_predictions, ["em", "f1"])
        overall_results[strategy_name] = {
            "description": strategy["description"],
            **overall,
            "n_samples": len(all_predictions),
        }

        per_type_results[strategy_name] = {}
        for qtype in ["single_hop", "multi_hop", "summarization"]:
            typed_preds = [r for r in all_predictions if r.get("query_type") == qtype]
            if typed_preds:
                per_type_results[strategy_name][qtype] = compute_metrics(typed_preds, ["em", "f1"])
            else:
                per_type_results[strategy_name][qtype] = {"em": 0, "f1": 0}

    # Generate report
    md_path = out_dir / "counterfactual_report.md"
    with open(md_path, "w") as f:
        f.write("# Routing Counterfactual Report\n\n")

        f.write("## Overall Results\n\n")
        headers = ["Strategy", "EM", "F1"]
        rows = []
        for s in args.strategies:
            r = overall_results.get(s, {})
            rows.append([r.get("description", s), f"{r.get('em', 0):.3f}", f"{r.get('f1', 0):.3f}"])
        f.write(render_markdown_table(headers, rows))
        f.write("\n\n")

        f.write("## Per-Type Breakdown (EM)\n\n")
        headers = ["Strategy", "Single-hop (n=200)", "Multi-hop (n=400)", "Summarization (n=100)"]
        rows = []
        for s in args.strategies:
            r = per_type_results.get(s, {})
            rows.append([
                overall_results.get(s, {}).get("description", s),
                f"{r.get('single_hop', {}).get('em', 0):.3f}",
                f"{r.get('multi_hop', {}).get('em', 0):.3f}",
                f"{r.get('summarization', {}).get('em', 0):.3f}",
            ])
        f.write(render_markdown_table(headers, rows))
        f.write("\n")

    save_results(
        {"overall": overall_results, "per_type": per_type_results},
        out_dir, "counterfactual_results",
    )
    logger.info("Routing counterfactual results saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
