"""实验 2：消融实验（Ablation Study）

三个变体分别对应论文的三个创新点：
  - w/o Query Triage：移除 TreeIntentClassifier，所有查询统一路由到 SingleHop
  - w/o PER：MultiHop Agent 退化为标准 ReAct（移除 Plan 和 Reflect）
  - w/o Memory：禁用 SharedMemory 和 AgentPrivateMemory

对应文档：docs/experiments/ablation-full.md

Usage:
    uv run python experiments/exp2_ablation.py
    uv run python experiments/exp2_ablation.py --datasets hotpotqa --limit 100
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

logger = logging.getLogger("chimera.exp2")

EXP_NAME = "exp2_ablation"

ABLATION_DATASETS = ["two_wiki", "hotpotqa", "musique", "asqa"]

VARIANTS = {
    "collarag_full": {
        "description": "CollaRAG (Full)",
        "innovation_removed": "—",
        "base_config": "colla_rag_only",
        "overrides": {},
    },
    "wo_query_triage": {
        "description": "w/o Query Triage",
        "innovation_removed": "创新点 1：Intent-Aware Query Triage",
        "base_config": "colla_rag_only",
        "overrides": {
            "query": {
                "intent_classifier": {
                    "active": "defaults.rule",
                    "params": {"default_intent": "single_hop"},
                },
            },
            "plugins": {
                "colla_rag": {
                    "multi_agent": {
                        "classifier": {
                            "force_intent": "single_hop",
                        },
                    },
                },
            },
        },
    },
    "wo_per": {
        "description": "w/o PER",
        "innovation_removed": "创新点 2：Plan-Execute-Reflect",
        "base_config": "colla_rag_only",
        "overrides": {
            "plugins": {
                "colla_rag": {
                    "multi_agent": {
                        "react": {
                            "disable_plan": True,
                            "disable_reflect": True,
                            "max_iterations": 3,
                        },
                    },
                },
            },
        },
    },
    "wo_memory": {
        "description": "w/o Memory",
        "innovation_removed": "创新点 3：Dual-Layer Experiential Memory",
        "base_config": "colla_rag_only",
        "overrides": {
            "plugins": {
                "colla_rag": {
                    "multi_agent": {
                        "memory": {
                            "shared": {"enabled": False},
                            "agent_private": {"enabled": False},
                        },
                    },
                },
            },
        },
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Ablation Study"
    )
    add_common_args(parser)
    parser.add_argument(
        "--datasets", nargs="+", default=ABLATION_DATASETS,
        choices=ABLATION_DATASETS,
        help="Datasets to evaluate on (multi-hop + summarization)",
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
        help="Ablation variants to run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)
    metrics_list = [m.strip() for m in args.metrics.split(",")]

    results: dict[str, dict[str, dict[str, float]]] = {}

    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        results[variant_name] = {}

        for dataset in args.datasets:
            logger.info(
                "=== Ablation: variant=%s dataset=%s ===",
                variant_name, dataset,
            )

            cfg = make_config(
                base=variant["base_config"],
                overrides=variant.get("overrides", {}),
                workspace_suffix=f"exp2_{variant_name}_{dataset}",
            )

            config_path = write_temp_config(cfg, f"{variant_name}_{dataset}", out_dir)
            pred_dir = out_dir / dataset / variant_name

            try:
                run_inference(config_path, dataset, pred_dir, limit=args.limit, rebuild=args.rebuild)
                rows, _meta = collect_predictions(pred_dir)
                scores = compute_metrics(rows, metrics_list)
                results[variant_name][dataset] = scores
                logger.info("  Scores: %s", scores)
            except Exception as e:
                logger.error("  FAILED: %s", e)
                results[variant_name][dataset] = {m: -1 for m in metrics_list}

    # Generate ablation table
    headers = ["Variant", "Innovation Removed"] + [
        f"{ds} EM" for ds in args.datasets
    ]
    rows_table = []
    for vname in args.variants:
        variant = VARIANTS[vname]
        row = [variant["description"], variant["innovation_removed"]]
        for ds in args.datasets:
            em = results.get(vname, {}).get(ds, {}).get("em", -1)
            row.append(f"**{em:.3f}**" if vname == "collarag_full" else f"{em:.3f}")
        rows_table.append(row)

    table = render_markdown_table(headers, rows_table)
    md_path = out_dir / "ablation_table.md"
    with open(md_path, "w") as f:
        f.write("# Ablation Study Results\n\n")
        f.write(f"> Backbone: Qwen3-8B, {args.limit} samples/dataset, seed=42\n\n")
        f.write(table)
        f.write("\n")

    save_results(results, out_dir, "ablation_results")
    logger.info("Ablation results saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
