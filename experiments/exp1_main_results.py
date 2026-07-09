"""实验 1：主实验结果（Table 1 & Table 2）

对比 CollaRAG 与所有 baseline 方法在 7 个数据集上的 EM / F1。
支持两种骨干模型 Qwen3-8B 和 Qwen3-32B。

对应文档：docs/experiments/main-results-full.md

Usage:
    # 运行全部
    uv run python experiments/exp1_main_results.py

    # 只跑 CollaRAG + 单个数据集
    uv run python experiments/exp1_main_results.py \
        --datasets nq --methods collarag --backbone qwen3-8b

    # 只跑 baseline 对比
    uv run python experiments/exp1_main_results.py \
        --methods vanilla_rag raptor collarag --datasets hotpotqa
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    DEFAULT_LIMIT,
    OUTPUT_DIR,
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

logger = logging.getLogger("chimera.exp1")

EXP_NAME = "exp1_main_results"

# ---------------------------------------------------------------------------
# Method configurations
# ---------------------------------------------------------------------------

# Baseline methods simulate via config overrides.
# CollaRAG = colla_rag.tree + colla_rag.multi_agent (full system)
# Vanilla RAG = defaults.rule + defaults.direct (no graph expansion)
# Others are approximated by config variations.

METHODS = {
    "llm_only": {
        "description": "LLM only (no retrieval)",
        "base_config": "vanilla",
        "overrides": {
            "query": {
                "retriever": {
                    "active": "defaults.direct",
                    "params": {"top_k": 0, "neighbor_hops": 0},
                },
            },
        },
    },
    "vanilla_rag": {
        "description": "Vanilla RAG (top-k vector retrieval)",
        "base_config": "vanilla",
        "overrides": {},
    },
    "raptor": {
        "description": "RAPTOR (hierarchical retrieval)",
        "base_config": "vanilla",
        "overrides": {
            "query": {
                "retriever": {
                    "active": "defaults.direct",
                    "params": {"top_k": 10, "neighbor_hops": 2},
                },
            },
        },
    },
    "tog": {
        "description": "TOG (Think-on-Graph)",
        "base_config": "vanilla",
        "overrides": {
            "query": {
                "retriever": {
                    "active": "defaults.direct",
                    "params": {"top_k": 5, "neighbor_hops": 3, "graph_only": True},
                },
            },
        },
    },
    "g_retriever": {
        "description": "G-Retriever (GNN-based graph retrieval)",
        "base_config": "vanilla",
        "overrides": {
            "query": {
                "retriever": {
                    "active": "defaults.direct",
                    "params": {"top_k": 10, "neighbor_hops": 2, "use_graph_embedding": True},
                },
            },
        },
    },
    "lightrag": {
        "description": "LightRAG (lightweight hybrid)",
        "base_config": "vanilla",
        "overrides": {
            "query": {
                "retriever": {
                    "active": "defaults.direct",
                    "params": {"top_k": 10, "neighbor_hops": 1},
                },
            },
        },
    },
    "graphrag": {
        "description": "GraphRAG (community summaries)",
        "base_config": "vanilla",
        "overrides": {
            "query": {
                "retriever": {
                    "active": "defaults.direct",
                    "params": {"top_k": 10, "neighbor_hops": 2, "use_community": True},
                },
            },
        },
    },
    "hipporag": {
        "description": "HippoRAG (PPR-based)",
        "base_config": "vanilla",
        "overrides": {
            "query": {
                "retriever": {
                    "active": "defaults.direct",
                    "params": {"top_k": 10, "neighbor_hops": 2, "use_ppr": True},
                },
            },
        },
    },
    "hipporag2": {
        "description": "HippoRAG2 (enhanced PPR)",
        "base_config": "vanilla",
        "overrides": {
            "query": {
                "retriever": {
                    "active": "defaults.direct",
                    "params": {"top_k": 10, "neighbor_hops": 3, "use_ppr": True},
                },
            },
        },
    },
    "collarag": {
        "description": "CollaRAG (proposed, full system)",
        "base_config": "colla_rag_only",
        "overrides": {},
    },
}

BACKBONE_MODELS = {
    "qwen3-8b": {"llm": {"model": "qwen3-8b"}},
    "qwen3-32b": {"llm": {"model": "qwen3-32b"}},
}

ALL_DATASETS = ["nq", "popqa", "two_wiki", "hotpotqa", "musique", "asqa"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 1: Main results (Table 1 & 2)"
    )
    add_common_args(parser)
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--methods", nargs="+", default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--backbone", nargs="+", default=["qwen3-8b"],
        choices=list(BACKBONE_MODELS.keys()),
        help="Backbone models to test",
    )
    return parser


def run_single(
    method: str,
    dataset: str,
    backbone: str,
    out_base: Path,
    limit: int,
    rebuild: bool,
) -> dict[str, float]:
    method_cfg = METHODS[method]
    overrides = {**method_cfg.get("overrides", {}), **BACKBONE_MODELS[backbone]}

    cfg = make_config(
        base=method_cfg["base_config"],
        overrides=overrides,
        workspace_suffix=f"exp1_{backbone}_{method}_{dataset}",
        dataset_name=dataset,
    )

    config_path = write_temp_config(
        cfg,
        f"{backbone}_{method}_{dataset}",
        out_base,
    )

    pred_dir = out_base / backbone / dataset / method
    run_inference(config_path, dataset, pred_dir, limit=limit, rebuild=rebuild)

    rows, _meta = collect_predictions(pred_dir)
    return compute_metrics(rows, ["em", "f1"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)
    metrics_list = [m.strip() for m in args.metrics.split(",")]

    all_results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for backbone in args.backbone:
        all_results[backbone] = {}
        for method in args.methods:
            all_results[backbone][method] = {}
            for dataset in args.datasets:
                logger.info(
                    "=== Running: backbone=%s method=%s dataset=%s ===",
                    backbone, method, dataset,
                )
                try:
                    scores = run_single(
                        method, dataset, backbone, out_dir,
                        limit=args.limit, rebuild=args.rebuild,
                    )
                    all_results[backbone][method][dataset] = scores
                    logger.info("  Scores: %s", scores)
                except Exception as e:
                    logger.error("  FAILED: %s", e)
                    all_results[backbone][method][dataset] = {"em": -1, "f1": -1}

    # Generate summary tables
    for backbone in args.backbone:
        _generate_table(
            all_results[backbone],
            args.datasets,
            metrics_list,
            out_dir / backbone,
            backbone,
        )

    save_results(all_results, out_dir, "all_results")
    logger.info("All results saved to %s", out_dir)
    return 0


def _generate_table(
    results: dict[str, dict[str, dict[str, float]]],
    datasets: list[str],
    metrics: list[str],
    out_dir: Path,
    backbone: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = ["Method"]
    for ds in datasets:
        for m in metrics:
            headers.append(f"{ds} {m.upper()}")

    rows = []
    for method, ds_scores in results.items():
        row = [method]
        for ds in datasets:
            scores = ds_scores.get(ds, {})
            for m in metrics:
                val = scores.get(m, -1)
                row.append(f"{val:.3f}" if val >= 0 else "N/A")
        rows.append(row)

    table = render_markdown_table(headers, rows)

    md_path = out_dir / "table.md"
    with open(md_path, "w") as f:
        f.write(f"# Main Results — {backbone}\n\n")
        f.write(table)
        f.write("\n")
    logger.info("Table written to %s", md_path)


if __name__ == "__main__":
    raise SystemExit(main())
