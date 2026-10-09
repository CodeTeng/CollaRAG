"""Offline scorer / reporter for Chimera-RAG.

This script is the second half of the split evaluation pipeline:

    main.py infer  →  output/<dataset>/<config_stem>/predictions.jsonl
                       + meta.json
    experiments/eval.py
                   →  report.md / report.json / report.csv  (single-dir)
                      compare.md / compare.json / compare.csv
                      (+ compare_bar.png / compare_radar.png) (multi-dir)

Usage:

    # single-directory scoring
    uv run python experiments/eval.py \
        --predictions output/mock_wiki/colla_rag_only/ \
        --metrics em,f1,rouge_l

    # multi-directory comparison
    uv run python experiments/eval.py \
        --predictions output/mock_wiki/vanilla/ \
                      output/mock_wiki/colla_rag_only/ \
        --out output/mock_wiki/compare/ \
        --metrics em,f1,rouge_l
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("chimera.eval")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval", description="Offline scoring & reporting for Chimera-RAG."
    )
    p.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help=(
            "one or more directories (or predictions.jsonl files) produced by "
            "`main.py infer`. Two or more triggers compare mode."
        ),
    )
    p.add_argument(
        "--metrics",
        default=None,
        help=(
            "comma-separated metric names (em, f1, rouge_l). "
            "If omitted, falls back to --config.evaluation.metrics, "
            "then to em,f1,rouge_l."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        help="optional config.yaml; only used to derive default metrics.",
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "output directory for report(s). In single mode defaults to the "
            "predictions dir itself. In compare mode is required (or defaults "
            "to the first predictions dir's parent / 'compare')."
        ),
    )
    p.add_argument(
        "--stem",
        default="report",
        help="filename stem for single-dir report (default: report)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _resolve_metrics(args: argparse.Namespace) -> list[str]:
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        if metrics:
            return metrics
    if args.config:
        try:
            from chimera_rag.core.config import load_config

            cfg = load_config(args.config)
            metrics = list(cfg.evaluation.metrics)
            if metrics:
                return metrics
        except Exception as e:
            logger.warning("failed to load metrics from --config %s: %s", args.config, e)
    return ["em", "f1", "rouge_l"]


def _score_one(pred_path: Path, metrics: list[str]):
    # Returns EvalResult; imported lazily to keep --help cheap.
    from chimera_rag.evaluation import Scorer, read_predictions_dir

    rows, meta = read_predictions_dir(pred_path)
    if not rows:
        raise SystemExit(f"no predictions found in {pred_path}")

    config_name = meta.get("config_name") or _guess_config_name(pred_path)
    result = Scorer(metrics=metrics).score(
        per_example=rows,
        config_name=config_name,
        usage=meta.get("usage_total", {}),
        elapsed_s=float(meta.get("elapsed_s", 0.0)),
        ingest=meta.get("ingest"),
    )
    return result


def _guess_config_name(pred_path: Path) -> str:
    p = pred_path
    if p.is_file():
        p = p.parent
    return p.name or "predictions"


def _run_single(args: argparse.Namespace, metrics: list[str]) -> int:
    from chimera_rag.evaluation import EvalReporter

    pred_path = Path(args.predictions[0])
    result = _score_one(pred_path, metrics)

    out_dir = Path(args.out) if args.out else (
        pred_path if pred_path.is_dir() else pred_path.parent
    )
    paths = EvalReporter(result).write(out_dir, args.stem)
    print("wrote report:")
    for kind, path in paths.items():
        print(f"  {kind:5s} -> {path}")

    print()
    print("metrics:")
    for k, v in result.metrics.items():
        print(f"  {k:15s} = {v:.4f}" if k != "avg_latency_s" else f"  {k:15s} = {v:.2f}")
    return 0


def _run_compare(args: argparse.Namespace, metrics: list[str]) -> int:
    from chimera_rag.evaluation import CompareReporter

    pred_paths = [Path(p) for p in args.predictions]
    results = []
    for pp in pred_paths:
        logger.info("scoring %s ...", pp)
        results.append(_score_one(pp, metrics))

    if args.out:
        out_dir = Path(args.out)
    else:
        # 猜一个合理位置：第一个路径的父目录下的 compare/
        first = pred_paths[0]
        parent = first.parent if first.is_dir() else first.parent.parent
        out_dir = parent / "compare"

    paths = CompareReporter(results).write(out_dir)
    print("wrote comparison report:")
    for kind, path in paths.items():
        print(f"  {kind:5s} -> {path}")

    print()
    print("summary:")
    for r in results:
        cells = [f"{k}={v:.4f}" if k != "avg_latency_s" else f"{k}={v:.2f}"
                 for k, v in r.metrics.items()]
        print(f"  {r.config_name:30s} {' | '.join(cells)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from chimera_rag.core.logging import setup_logging

        setup_logging(level=args.log_level, format="rich")
    except Exception:
        logging.basicConfig(level=args.log_level)

    metrics = _resolve_metrics(args)
    logger.info("scoring with metrics=%s", metrics)

    if len(args.predictions) == 1:
        return _run_single(args, metrics)
    return _run_compare(args, metrics)


if __name__ == "__main__":
    raise SystemExit(main())
