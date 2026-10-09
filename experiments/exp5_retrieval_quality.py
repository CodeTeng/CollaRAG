"""实验 5：检索质量与忠实性分析（Retrieval Quality and Faithfulness）

评估 CollaRAG 在多跳数据集上的 Recall@k、忠实性、幻觉率、证据支撑率。

对应文档：docs/experiments/retrieval-quality.md

Usage:
    uv run python experiments/exp5_retrieval_quality.py
    uv run python experiments/exp5_retrieval_quality.py --datasets hotpotqa --methods collarag
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

logger = logging.getLogger("chimera.exp5")

EXP_NAME = "exp5_retrieval_quality"
EVAL_DATASETS = ["two_wiki", "hotpotqa", "musique"]

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


def compute_recall_at_k(
    predictions: list[dict],
    gold_evidence_key: str = "gold_evidence_ids",
    evidence_key: str = "evidence_chunk_ids",
    k_values: list[int] | None = None,
) -> dict[str, float]:
    k_values = k_values or [5, 10]
    results = {}
    for k in k_values:
        hits = 0
        total = 0
        for row in predictions:
            gold = set(row.get(gold_evidence_key, []))
            retrieved = row.get(evidence_key, [])[:k]
            if gold:
                total += 1
                if gold & set(retrieved):
                    hits += 1
        results[f"recall@{k}"] = round(hits / total, 3) if total > 0 else 0.0
    return results


def compute_faithfulness(predictions: list[dict]) -> dict[str, float]:
    from chimera_rag.evaluation.metrics import token_f1

    total_sentences = 0
    supported_sentences = 0

    for row in predictions:
        pred_text = str(row.get("prediction", ""))
        evidence_texts = row.get("evidence_texts", [])
        if not pred_text.strip() or not evidence_texts:
            continue

        sentences = [s.strip() for s in pred_text.replace("。", ".").split(".") if s.strip()]
        for sentence in sentences:
            total_sentences += 1
            max_overlap = max(
                (token_f1(sentence, str(ev)) for ev in evidence_texts),
                default=0.0,
            )
            if max_overlap > 0.3:
                supported_sentences += 1

    if total_sentences == 0:
        return {"faithfulness_score": 0.0, "hallucination_rate": 0.0, "evidence_support_rate": 0.0}

    support_rate = supported_sentences / total_sentences
    return {
        "faithfulness_score": round(support_rate, 3),
        "hallucination_rate": round(1.0 - support_rate, 3),
        "evidence_support_rate": round(support_rate, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Retrieval Quality & Faithfulness"
    )
    add_common_args(parser)
    parser.add_argument("--datasets", nargs="+", default=EVAL_DATASETS)
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for method_name in args.methods:
        method = METHODS[method_name]
        all_results[method_name] = {}

        for dataset in args.datasets:
            logger.info("=== Retrieval Quality: method=%s dataset=%s ===", method_name, dataset)

            cfg = make_config(
                base=method["base_config"],
                overrides=method.get("overrides", {}),
                workspace_suffix=f"exp5_{method_name}_{dataset}",
            )
            config_path = write_temp_config(cfg, f"{method_name}_{dataset}", out_dir)
            pred_dir = out_dir / dataset / method_name

            try:
                run_inference(config_path, dataset, pred_dir, limit=args.limit, rebuild=args.rebuild)
                rows, _meta = collect_predictions(pred_dir)

                recall = compute_recall_at_k(rows)
                faithfulness = compute_faithfulness(rows)

                result = {**recall, **faithfulness, "n_samples": len(rows)}
                all_results[method_name][dataset] = result
                logger.info("  Results: %s", result)
            except Exception as e:
                logger.error("  FAILED: %s", e)
                all_results[method_name][dataset] = {"error": str(e)}

    _generate_report(all_results, args.datasets, args.methods, out_dir)
    save_results(all_results, out_dir, "retrieval_quality_results")
    logger.info("Retrieval quality results saved to %s", out_dir)
    return 0


def _generate_report(results: dict, datasets: list[str], methods: list[str], out_dir: Path) -> None:
    md_path = out_dir / "retrieval_quality_report.md"
    with open(md_path, "w") as f:
        f.write("# Retrieval Quality & Faithfulness Report\n\n")

        f.write("## 1. Recall@k\n\n")
        headers = ["Method"] + [f"{ds} R@5" for ds in datasets] + [f"{ds} R@10" for ds in datasets]
        rows = []
        for method in methods:
            row = [method]
            for ds in datasets:
                r = results.get(method, {}).get(ds, {})
                row.append(f"{r.get('recall@5', 0):.3f}")
            for ds in datasets:
                r = results.get(method, {}).get(ds, {})
                row.append(f"{r.get('recall@10', 0):.3f}")
            rows.append(row)
        f.write(render_markdown_table(headers, rows))
        f.write("\n\n")

        f.write("## 2. Faithfulness Score\n\n")
        headers = ["Method"] + datasets + ["Average"]
        rows = []
        for method in methods:
            row = [method]
            scores = []
            for ds in datasets:
                r = results.get(method, {}).get(ds, {})
                s = r.get("faithfulness_score", 0)
                row.append(f"{s:.3f}")
                scores.append(s)
            row.append(f"{sum(scores) / len(scores):.3f}" if scores else "N/A")
            rows.append(row)
        f.write(render_markdown_table(headers, rows))
        f.write("\n\n")

        f.write("## 3. Hallucination Rate\n\n")
        headers = ["Method"] + datasets + ["Average"]
        rows = []
        for method in methods:
            row = [method]
            scores = []
            for ds in datasets:
                r = results.get(method, {}).get(ds, {})
                s = r.get("hallucination_rate", 0)
                row.append(f"{s:.3f}")
                scores.append(s)
            row.append(f"{sum(scores) / len(scores):.3f}" if scores else "N/A")
            rows.append(row)
        f.write(render_markdown_table(headers, rows))
        f.write("\n")

    logger.info("Report written to %s", md_path)


if __name__ == "__main__":
    raise SystemExit(main())
