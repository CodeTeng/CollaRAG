"""Shared utilities for all experiment scripts.

Provides:
- Dynamic YAML config generation with parameter overrides
- Result collection and aggregation
- Markdown table rendering
- Common CLI argument helpers
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("chimera.experiments")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "output" / "experiments"

DATASETS_ALL = ["nq", "popqa", "hotpotqa", "two_wiki", "musique", "asqa"]
DATASETS_SINGLE_HOP = ["nq", "popqa"]
DATASETS_MULTI_HOP = ["hotpotqa", "two_wiki", "musique"]
DATASETS_SUMMARIZATION = ["asqa"]

DEFAULT_SEED = 42
DEFAULT_LIMIT = 500
DEFAULT_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def load_base_config(config_name: str = "colla_rag_only") -> dict[str, Any]:
    path = CONFIGS_DIR / f"{config_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def make_config(
    *,
    base: str = "colla_rag_only",
    overrides: dict[str, Any] | None = None,
    workspace_suffix: str = "",
    dataset_name: str | None = None,
) -> dict[str, Any]:
    cfg = load_base_config(base)

    if workspace_suffix:
        cfg["app"]["workspace_dir"] = f"./storage/exp_{workspace_suffix}"
        if "storage" in cfg:
            stem = f"exp_{workspace_suffix}"
            cfg["storage"]["graph"]["persist_path"] = f"./storage/{stem}/graph.gpickle"
            cfg["storage"]["vector"]["persist_path"] = f"./storage/{stem}/vectors"

    cfg["llm"]["temperature"] = DEFAULT_TEMPERATURE

    if overrides:
        _deep_merge(cfg, overrides)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def write_temp_config(cfg: dict[str, Any], name: str, exp_dir: Path) -> Path:
    config_dir = exp_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return path


# ---------------------------------------------------------------------------
# Running inference via CLI
# ---------------------------------------------------------------------------

def run_inference(
    config_path: Path | str,
    dataset: str,
    out_dir: Path | str,
    limit: int = DEFAULT_LIMIT,
    rebuild: bool = False,
) -> Path:
    import subprocess

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python", "main.py", "infer",
        "--config", str(config_path),
        "--dataset", dataset,
        "--out", str(out_dir),
        "--limit", str(limit),
    ]
    if rebuild:
        cmd.append("--rebuild")

    logger.info("Running: %s", " ".join(cmd))
    t0 = time.perf_counter()

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    elapsed = time.perf_counter() - t0
    logger.info("Finished in %.1fs (returncode=%d)", elapsed, result.returncode)

    if result.returncode != 0:
        logger.error("STDERR: %s", result.stderr[-2000:] if result.stderr else "(empty)")
        raise RuntimeError(
            f"Inference failed for config={config_path}, dataset={dataset}:\n"
            f"{result.stderr[-1000:]}"
        )

    return out_dir


def run_scoring(
    predictions_dirs: list[Path | str],
    metrics: list[str] | None = None,
    out_dir: Path | str | None = None,
) -> dict[str, Any]:
    import subprocess

    cmd = [
        "uv", "run", "python", "experiments/eval.py",
        "--predictions", *[str(p) for p in predictions_dirs],
    ]
    if metrics:
        cmd.extend(["--metrics", ",".join(metrics)])
    if out_dir:
        cmd.extend(["--out", str(out_dir)])

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Scoring failed: %s", result.stderr[-1000:])
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def collect_predictions(pred_dir: Path) -> tuple[list[dict], dict]:
    pred_file = pred_dir / "predictions.jsonl"
    meta_file = pred_dir / "meta.json"

    rows = []
    if pred_file.exists():
        with open(pred_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    meta = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

    return rows, meta


def compute_metrics(rows: list[dict], metrics: list[str] | None = None) -> dict[str, float]:
    from chimera_rag.evaluation.metrics import exact_match, token_f1, rouge_l

    metric_fns = {
        "em": exact_match,
        "f1": token_f1,
        "rouge_l": rouge_l,
    }
    metrics = metrics or ["em", "f1"]

    results = {}
    for m in metrics:
        fn = metric_fns[m]
        scores = [fn(str(r.get("prediction", "")), str(r.get("reference", ""))) for r in rows]
        results[m] = sum(scores) / len(scores) if scores else 0.0
    return results


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_markdown_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> str:
    if alignments is None:
        alignments = ["left"] + ["right"] * (len(headers) - 1)

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _pad(cell: str, width: int, align: str) -> str:
        if align == "right":
            return cell.rjust(width)
        return cell.ljust(width)

    header_line = "| " + " | ".join(
        _pad(h, col_widths[i], alignments[i]) for i, h in enumerate(headers)
    ) + " |"

    sep_line = "|"
    for i, a in enumerate(alignments):
        w = col_widths[i]
        if a == "right":
            sep_line += "-" * w + ":|"
        elif a == "center":
            sep_line += ":" + "-" * (w - 1) + ":|"
        else:
            sep_line += "-" * (w + 1) + "|"

    data_lines = []
    for row in rows:
        line = "| " + " | ".join(
            _pad(cell, col_widths[i], alignments[i]) for i, cell in enumerate(row)
        ) + " |"
        data_lines.append(line)

    return "\n".join([header_line, sep_line] + data_lines)


def save_results(results: dict[str, Any], out_dir: Path, name: str = "results") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_path = out_dir / f"{name}.md"
    with open(md_path, "w") as f:
        f.write(f"# {name}\n\n")
        f.write(f"```json\n{json.dumps(results, indent=2, ensure_ascii=False)}\n```\n")

    return {"json": json_path, "md": md_path}


# ---------------------------------------------------------------------------
# Common CLI helpers
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"Number of QA samples per dataset (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--metrics", default="em,f1",
        help="Comma-separated metrics (default: em,f1)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: output/experiments/<exp_name>/)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild graph (ignore cached)",
    )


def setup_experiment(args: argparse.Namespace, exp_name: str) -> Path:
    try:
        from chimera_rag.core.logging import setup_logging
        setup_logging(level=args.log_level, format="rich")
    except Exception:
        logging.basicConfig(level=args.log_level)

    out_dir = args.out_dir or (OUTPUT_DIR / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Experiment '%s' output → %s", exp_name, out_dir)
    return out_dir
