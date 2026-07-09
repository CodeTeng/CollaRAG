"""实验 9：记忆累积效应曲线（Memory Accumulation Curve）

验证 Dual-Layer Memory 的飞轮效应——随着查询数量增加，
QA cache 命中率和答案质量逐步提升。

对应文档：docs/experiments/memory-accumulation.md

Usage:
    uv run python experiments/exp9_memory_accumulation.py
    uv run python experiments/exp9_memory_accumulation.py --window-size 50 --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    PROJECT_ROOT,
    add_common_args,
    render_markdown_table,
    save_results,
    setup_experiment,
)

logger = logging.getLogger("chimera.exp9")

EXP_NAME = "exp9_memory_accumulation"
DEFAULT_DATASET = "hotpotqa"
DEFAULT_WINDOW_SIZE = 50


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 9: Memory Accumulation Curve"
    )
    add_common_args(parser)
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help=f"Dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
        help=f"Window size for accumulation tracking (default: {DEFAULT_WINDOW_SIZE})",
    )
    parser.add_argument(
        "--config", default="configs/colla_rag_only.yaml",
        help="Config file with memory enabled",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)

    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import QAExample, Query
    from chimera_rag.datasets import get_dataset_loader
    from chimera_rag.evaluation.metrics import exact_match, token_f1

    config_path = str(PROJECT_ROOT / args.config)

    # Clear any existing memory to start fresh
    memory_dir = PROJECT_ROOT / ".chimera_memory"
    if memory_dir.exists():
        logger.info("Clearing existing memory at %s for clean cold-start", memory_dir)
        shutil.rmtree(memory_dir, ignore_errors=True)

    chimera = ChimeraRAG.from_config(config_path)

    # Load dataset
    from chimera_rag.core.config import load_config
    cfg = load_config(config_path)
    ds_cfg = cfg.datasets[args.dataset]

    loader_cls = get_dataset_loader(ds_cfg.loader)
    kwargs = {}
    if ds_cfg.corpus_path:
        kwargs["corpus_path"] = ds_cfg.corpus_path
    if ds_cfg.qa_path:
        kwargs["qa_path"] = ds_cfg.qa_path
    loader = loader_cls(**kwargs)

    examples = loader.load_examples(limit=args.limit)
    if not examples:
        logger.error("No examples loaded for dataset %s", args.dataset)
        return 1

    # Ensure graph is loaded
    if chimera.stats()["chunks"] == 0:
        chimera.try_load_graph(args.dataset)
    if chimera.stats()["chunks"] == 0:
        docs = loader.load_documents(limit=args.limit)
        if docs:
            chimera.ingest(docs)
            chimera.persist_graph(args.dataset)

    # Run queries sequentially, tracking metrics per window
    window_size = args.window_size
    windows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    current_window_preds: list[dict] = []

    for i, ex in enumerate(examples):
        t0 = time.perf_counter()
        answer = chimera.query(Query(text=ex.question))
        latency = time.perf_counter() - t0

        pred = {
            "qid": ex.qid,
            "question": ex.question,
            "reference": ex.answer,
            "prediction": answer.text,
            "latency_s": round(latency, 4),
            "intent_label": answer.intent_label,
            "confidence": answer.confidence,
            "em": exact_match(answer.text, ex.answer),
            "f1": token_f1(answer.text, ex.answer),
        }
        all_predictions.append(pred)
        current_window_preds.append(pred)

        if (i + 1) % 50 == 0:
            logger.info(
                "[%d/%d] EM=%.3f F1=%.3f lat=%.2fs",
                i + 1, len(examples),
                pred["em"], pred["f1"], latency,
            )

        # Record window stats
        if len(current_window_preds) >= window_size or i == len(examples) - 1:
            window_start = i + 1 - len(current_window_preds) + 1
            window_end = i + 1

            window_em = sum(p["em"] for p in current_window_preds) / len(current_window_preds)
            window_f1 = sum(p["f1"] for p in current_window_preds) / len(current_window_preds)
            window_latency = sum(p["latency_s"] for p in current_window_preds) / len(current_window_preds)

            # Try to read memory stats
            memory_stats = _get_memory_stats(chimera)

            window_data = {
                "window": f"{window_start}-{window_end}",
                "n_queries": len(current_window_preds),
                "em": round(window_em, 3),
                "f1": round(window_f1, 3),
                "avg_latency_s": round(window_latency, 3),
                **memory_stats,
            }
            windows.append(window_data)
            logger.info(
                "Window %s: EM=%.3f F1=%.3f cache_hit=%.1f%%",
                window_data["window"], window_em, window_f1,
                memory_stats.get("qa_cache_hit_rate", 0) * 100,
            )
            current_window_preds = []

    # Compute cold-start vs warm-up comparison
    if len(windows) >= 2:
        cold = windows[0]
        warm = windows[-1]
        improvement = {
            "em_improvement": f"+{((warm['em'] - cold['em']) / cold['em'] * 100):.1f}%" if cold["em"] > 0 else "N/A",
            "f1_improvement": f"+{((warm['f1'] - cold['f1']) / cold['f1'] * 100):.1f}%" if cold["f1"] > 0 else "N/A",
        }
    else:
        improvement = {}

    # Generate report
    _generate_report(windows, improvement, out_dir, args)

    # Save detailed predictions
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in all_predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    save_results(
        {"windows": windows, "improvement": improvement, "total_queries": len(all_predictions)},
        out_dir, "memory_accumulation_results",
    )
    logger.info("Memory accumulation results saved to %s", out_dir)
    return 0


def _get_memory_stats(chimera) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "qa_cache_hit_rate": 0.0,
        "qa_cache_size": 0,
        "entity_aliases": 0,
        "routing_stats": 0,
    }

    try:
        retriever = chimera.query_pipeline.retriever
        memory_manager = getattr(retriever, "memory_manager", None)
        if memory_manager is None:
            return stats

        shared = getattr(memory_manager, "shared", None)
        if shared:
            qa_cache = getattr(shared, "qa_cache", {})
            stats["qa_cache_size"] = len(qa_cache) if isinstance(qa_cache, dict) else 0

            cache_stats = getattr(shared, "cache_stats", None)
            if cache_stats and isinstance(cache_stats, dict):
                hits = cache_stats.get("hits", 0)
                total = cache_stats.get("total", 0)
                stats["qa_cache_hit_rate"] = round(hits / total, 3) if total > 0 else 0.0

            aliases = getattr(shared, "entity_aliases", {})
            stats["entity_aliases"] = len(aliases) if isinstance(aliases, dict) else 0

            routing = getattr(shared, "routing_stats", [])
            stats["routing_stats"] = len(routing) if isinstance(routing, list) else 0

        agent_private = getattr(memory_manager, "agent_private", None)
        if agent_private:
            buckets = getattr(agent_private, "buckets", {})
            if isinstance(buckets, dict):
                stats["private_memory_entries"] = sum(
                    len(v) for v in buckets.values() if isinstance(v, list)
                )
    except Exception as e:
        logger.debug("Failed to read memory stats: %s", e)

    return stats


def _generate_report(
    windows: list[dict],
    improvement: dict,
    out_dir: Path,
    args,
) -> None:
    md_path = out_dir / "memory_accumulation_report.md"
    with open(md_path, "w") as f:
        f.write("# Memory Accumulation Curve Report\n\n")
        f.write(f"> Dataset: {args.dataset}, Backbone: Qwen3-8B, {args.limit} queries\n\n")

        # Accumulation table
        f.write("## Accumulation Curve\n\n")
        headers = ["Window", "EM", "F1", "QA Cache Hit Rate", "Cache Size"]
        rows = []
        for w in windows:
            is_cold = w == windows[0]
            is_warm = w == windows[-1]
            label = w["window"]
            if is_cold:
                label += " (cold)"
            elif is_warm:
                label += " (warm)"
            rows.append([
                label,
                f"{w['em']:.3f}",
                f"{w['f1']:.3f}",
                f"{w.get('qa_cache_hit_rate', 0) * 100:.1f}%",
                str(w.get("qa_cache_size", 0)),
            ])
        f.write(render_markdown_table(headers, rows))
        f.write("\n\n")

        # Cold vs Warm comparison
        if len(windows) >= 2 and improvement:
            f.write("## Cold Start vs Fully Warmed\n\n")
            cold = windows[0]
            warm = windows[-1]
            headers = ["Metric", "Cold Start", "Fully Warmed", "Improvement"]
            rows = [
                ["EM", f"{cold['em']:.3f}", f"{warm['em']:.3f}", improvement.get("em_improvement", "N/A")],
                ["F1", f"{cold['f1']:.3f}", f"{warm['f1']:.3f}", improvement.get("f1_improvement", "N/A")],
            ]
            f.write(render_markdown_table(headers, rows))
            f.write("\n")

    logger.info("Report written to %s", md_path)


if __name__ == "__main__":
    raise SystemExit(main())
