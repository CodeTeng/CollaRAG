"""RAGAS semantic scorer for Chimera-RAG.

This script is a third, optional evaluation path that complements the
word-overlap metrics produced by ``experiments/eval.py``. It computes RAGAS's four
semantic metrics on the predictions written by ``main.py infer``:

    main.py infer  →  output/<dataset>/<config_stem>/predictions.jsonl
                       (must contain the `evidence_texts` field)
    experiments/eval_ragas.py
                   →  ragas_report.md / ragas_report.json / ragas_report.csv

Both the judge LLM and the embedding model are built from the project
``config.yaml`` (--config), so DeepSeek / OpenAI / local sentence-transformers
all work without extra wiring.

Usage:

    uv run python experiments/eval_ragas.py \
        --predictions output/mock_wiki/colla_rag_only/ \
        --config configs/colla_rag_only.yaml

Install the extra first:

    uv sync --extra ragas
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger("chimera.eval_ragas")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_ragas",
        description="RAGAS semantic scoring (faithfulness / answer relevancy / "
        "context precision / context recall) for Chimera-RAG.",
    )
    p.add_argument(
        "--predictions",
        required=True,
        help="directory (or predictions.jsonl) produced by `main.py infer`.",
    )
    p.add_argument(
        "--config",
        required=True,
        help="config.yaml; provides the judge LLM (config.llm) and embedding "
        "(config.embedding) used by RAGAS.",
    )
    p.add_argument(
        "--metrics",
        default=None,
        help="comma-separated RAGAS metric names. Defaults to all four: "
        "faithfulness,answer_relevancy,context_precision,context_recall.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output directory for the RAGAS report; defaults to the "
        "predictions dir itself.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only score the first N predictions (smoke runs).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _resolve_metrics(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    metrics = [m.strip() for m in raw.split(",") if m.strip()]
    return metrics or None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from chimera_rag.core.logging import setup_logging

        setup_logging(level=args.log_level, format="rich")
    except Exception:
        logging.basicConfig(level=args.log_level)

    # Load a local .env so the RAGAS judge LLM / embeddings can resolve
    # LLM_API_KEY / LLM_BASE_URL / LLM_MODEL (same as ChimeraRAG.from_config).
    from chimera_rag.core.env_loader import load_env_file

    load_env_file()

    from chimera_rag.core.config import load_config
    from chimera_rag.evaluation import (
        RagasEvaluator,
        RagasReporter,
        read_predictions_dir,
    )

    pred_path = Path(args.predictions)
    rows, meta = read_predictions_dir(pred_path)
    if not rows:
        raise SystemExit(f"no predictions found in {pred_path}")
    if args.limit is not None:
        rows = rows[: args.limit]

    if not any(r.get("evidence_texts") for r in rows):
        logger.warning(
            "predictions 不含 evidence_texts；需要 context 的指标将被跳过。"
            "请用更新后的 `main.py infer` 重跑以获得完整 RAGAS 指标。"
        )

    config = load_config(args.config)
    metrics = _resolve_metrics(args.metrics)
    config_name = meta.get("config_name") or Path(args.config).stem

    result = RagasEvaluator(config).evaluate(
        rows, metrics=metrics, config_name=config_name
    )

    out_dir = Path(args.out) if args.out else (
        pred_path if pred_path.is_dir() else pred_path.parent
    )
    paths = RagasReporter(result).write(out_dir)

    print("wrote RAGAS report:")
    for kind, path in paths.items():
        print(f"  {kind:5s} -> {path}")

    if result.skipped_metrics:
        print()
        print(f"skipped (no context): {', '.join(result.skipped_metrics)}")

    print()
    print("RAGAS metrics:")
    for k, v in result.metrics.items():
        print(f"  {k:20s} = {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
