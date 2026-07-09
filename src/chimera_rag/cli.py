"""Command-line entry point for Chimera-RAG.

``main.py`` at the repo root imports :func:`main` from here.
Subcommands are dispatched via :mod:`argparse`:

* ``ingest``  — build the knowledge graph from a configured dataset
* ``query``   — single-shot question answering
* ``infer``   — batch-run a QA dataset and write predictions to disk
                (no metric calculation; use ``experiments/eval.py`` for that)
* ``export``  — dump graph/vectors/stats to ``output/``
* ``serve``   — start the FastAPI + React web UI

Evaluation is now split into two stages:

* ``main.py infer`` produces predictions.jsonl + meta.json
* ``experiments/eval.py``       reads those, computes metrics, and renders the report

This decoupling means expensive inference only runs once, while metric
iteration and report regeneration are free.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chimera_rag.core.exceptions import ChimeraError
from chimera_rag.core.logging import setup_logging


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--config",
        required=True,
        help="Path to a Chimera-RAG config.yaml",
    )
    sp.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chimera", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ingest ------------------------------------------------------------
    p_ing = sub.add_parser("ingest", help="build the knowledge graph from a dataset")
    _add_common(p_ing)
    p_ing.add_argument("--dataset", required=True, help="named dataset from config.datasets")
    p_ing.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "cap the number of SOURCE ROWS scanned from the dataset before "
            "ingestion (e.g. --limit 500 keeps the first 500 QA rows and only "
            "their attached paragraphs become Documents). Use this to align "
            "the ingestion corpus with a 500-sample evaluation protocol."
        ),
    )
    p_ing.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore any persisted graph and rebuild from scratch",
    )
    p_ing.set_defaults(func=cmd_ingest)

    # query -------------------------------------------------------------
    p_q = sub.add_parser("query", help="ask a single question")
    _add_common(p_q)
    p_q.add_argument("question", help="natural-language question")
    p_q.set_defaults(func=cmd_query)

    # infer -------------------------------------------------------------
    p_i = sub.add_parser(
        "infer",
        help="batch-run QA dataset and write predictions.jsonl + meta.json",
    )
    _add_common(p_i)
    p_i.add_argument("--dataset", required=True)
    p_i.add_argument("--limit", type=int, default=None)
    p_i.add_argument(
        "--out",
        default=None,
        help=(
            "output directory for predictions.jsonl + meta.json; "
            "defaults to <dataset.output_dir>/<config_stem>/"
        ),
    )
    p_i.add_argument(
        "--rebuild",
        action="store_true",
        help="ignore any persisted graph and rebuild from scratch before inference",
    )
    p_i.set_defaults(func=cmd_infer)

    # export ------------------------------------------------------------
    p_x = sub.add_parser("export", help="dump graph / vectors / stats")
    _add_common(p_x)
    p_x.add_argument("--out", required=True)
    p_x.set_defaults(func=cmd_export)

    # serve -------------------------------------------------------------
    p_s = sub.add_parser("serve", help="start the FastAPI web UI")
    _add_common(p_s)
    p_s.add_argument("--host", default=None)
    p_s.add_argument("--port", type=int, default=None)
    p_s.add_argument(
        "--dataset",
        default=None,
        help=(
            "named dataset whose persisted graph should be loaded on startup "
            "so the GraphPage can render it without re-ingesting; must match "
            "the dataset used when the graph was built"
        ),
    )
    p_s.set_defaults(func=cmd_serve)

    return parser


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------
def _build_chimera(config_path: str):
    from chimera_rag import ChimeraRAG

    return ChimeraRAG.from_config(config_path)


def _load_dataset_from_config(config, dataset_name: str):
    from chimera_rag.core.exceptions import DatasetError
    from chimera_rag.datasets import get_dataset_loader

    try:
        ds_cfg = config.datasets[dataset_name]
    except KeyError as e:
        raise DatasetError(
            f"dataset {dataset_name!r} not declared in config.datasets; "
            f"available={list(config.datasets)}"
        ) from e

    loader_cls = get_dataset_loader(ds_cfg.loader)
    kwargs: dict = {}
    if ds_cfg.corpus_path:
        kwargs["corpus_path"] = ds_cfg.corpus_path
    if ds_cfg.qa_path:
        kwargs["qa_path"] = ds_cfg.qa_path
    if ds_cfg.path and not kwargs:
        kwargs["path"] = ds_cfg.path
    elif ds_cfg.path:
        kwargs.setdefault("path", ds_cfg.path)

    if ds_cfg.eval_field_mapping:
        kwargs["eval_field_mapping"] = ds_cfg.eval_field_mapping
    try:
        return loader_cls(**kwargs)
    except TypeError as e:
        if ds_cfg.path:
            return loader_cls(path=ds_cfg.path)
        raise DatasetError(
            f"loader {ds_cfg.loader!r} does not accept the given arguments: {e}"
        ) from e


# ------------------- ingest
def cmd_ingest(args: argparse.Namespace) -> int:
    crag = _build_chimera(args.config)
    loader = _load_dataset_from_config(crag.config, args.dataset)

    # Reuse a persisted graph unless --rebuild was passed.
    if not args.rebuild and crag.try_load_graph(args.dataset):
        print(
            f">>> 复用已持久化图谱（{crag.stats()['chunks']} chunk、"
            f"{crag.stats()['triples']} 三元组），跳过构图。"
            "如需重建请加 --rebuild",
            file=sys.stderr,
        )
        print(json.dumps({"ingest": "reused-persisted", "state": crag.stats()}, indent=2))
        return 0

    docs = loader.load_documents(limit=args.limit)
    if not docs:
        print(f"dataset {args.dataset!r} yielded no documents", file=sys.stderr)
        return 1
    if args.limit is not None:
        print(
            f">>> ingest limited to first {args.limit} rows → {len(docs)} documents",
            file=sys.stderr,
        )
    stats = crag.ingest(docs)
    paths = crag.persist_graph(args.dataset)
    print(json.dumps({"ingest": stats, "state": crag.stats(), "persisted": paths}, indent=2))
    return 0


# ------------------- query
def cmd_query(args: argparse.Namespace) -> int:
    from chimera_rag.core.types import Query

    crag = _build_chimera(args.config)
    ans = crag.query(Query(text=args.question))
    print(json.dumps(
        {
            "answer": ans.text,
            "intent": ans.intent_label,
            "evidence_chunk_ids": ans.evidence_chunk_ids,
            "confidence": ans.confidence,
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


# ------------------- infer
def _snapshot_usage(crag) -> dict[str, int]:
    """读取 crag.llm.usage（OpenAI/DeepSeek provider）的累计值。"""
    llm = getattr(crag, "llm", None)
    usage = getattr(llm, "usage", None)
    if isinstance(usage, dict):
        return dict(usage)
    return {}


def _diff_usage(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    keys = set(after) | set(before)
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in keys}


def cmd_infer(args: argparse.Namespace) -> int:
    import time as _time

    from chimera_rag.evaluation import InferenceRunner, write_predictions_dir

    crag = _build_chimera(args.config)
    loader = _load_dataset_from_config(crag.config, args.dataset)
    examples = loader.load_examples(limit=args.limit)
    if not examples:
        print(f"dataset {args.dataset!r} yielded no examples", file=sys.stderr)
        return 1

    # 优先复用已持久化的图谱（除非 --rebuild）；命中则直接跳过构图。
    if not args.rebuild and crag.stats()["chunks"] == 0 and crag.try_load_graph(args.dataset):
        print(
            f">>> 复用已持久化图谱（{crag.stats()['chunks']} chunk、"
            f"{crag.stats()['triples']} 三元组），跳过构图。"
            "如需重建请加 --rebuild",
            file=sys.stderr,
        )

    # 如果图谱尚未注入（无持久化或 --rebuild），触发一次 ingest 并落盘；
    # 把 token/耗时/图谱规模单独记录，便于在报告里区分"构图消耗"与"推理消耗"。
    ingest_info: dict[str, Any] | None = None
    if crag.stats()["chunks"] == 0:
        try:
            docs = loader.load_documents()
        except NotImplementedError:
            docs = []
        if docs:
            print(
                f">>> 首次使用 config `{Path(args.config).stem}` + dataset "
                f"`{args.dataset}`，开始构图（{len(docs)} 篇文档），"
                "请稍候……",
                file=sys.stderr,
            )
            usage_before = _snapshot_usage(crag)
            t0 = _time.perf_counter()
            ingest_stats = crag.ingest(docs)
            elapsed = _time.perf_counter() - t0
            usage_delta = _diff_usage(_snapshot_usage(crag), usage_before)
            ingest_info = {
                "elapsed_s": round(elapsed, 3),
                "usage": usage_delta,
                "documents": int(ingest_stats.get("documents", len(docs))),
                "chunks": int(ingest_stats.get("chunks", 0)),
                "triples_raw": int(ingest_stats.get("triples_raw", 0)),
                "triples_pruned": int(ingest_stats.get("triples_pruned", 0)),
            }
            print(
                f">>> 构图完成：{ingest_info['chunks']} 个 chunk、"
                f"{ingest_info['triples_pruned']} 个三元组，耗时 "
                f"{elapsed:.1f}s，LLM 调用 {int(usage_delta.get('calls', 0))} 次、"
                f"token {int(usage_delta.get('total_tokens', 0))}",
                file=sys.stderr,
            )
            # 落盘，供后续运行秒级复用（除非下次显式 --rebuild）。
            persisted = crag.persist_graph(args.dataset)
            print(
                f">>> 图谱已持久化到 {persisted['graph']}（下次自动复用，"
                "如需重建加 --rebuild）",
                file=sys.stderr,
            )

    config_stem = Path(args.config).stem
    per_example, meta = InferenceRunner().run(
        pipeline=crag,
        dataset_examples=examples,
        config_name=config_stem,
    )

    # 元信息补充：配置、数据集、limit，便于离线打分展示
    meta["config_path"] = str(Path(args.config).resolve())
    meta["dataset"] = args.dataset
    if args.limit is not None:
        meta["limit"] = args.limit
    if ingest_info is not None:
        meta["ingest"] = ingest_info

    # 输出目录解析：--out 优先；否则 <dataset.output_dir>/<config_stem>/
    if args.out:
        out_dir = Path(args.out)
    else:
        ds_cfg = crag.config.datasets.get(args.dataset)
        if ds_cfg and ds_cfg.output_dir:
            out_dir = Path(ds_cfg.output_dir) / config_stem
        else:
            out_dir = Path("output") / args.dataset / config_stem

    paths = write_predictions_dir(out_dir, per_example=per_example, meta=meta)
    print("wrote predictions:")
    for kind, p in paths.items():
        print(f"  {kind:12s} -> {p}")
    return 0


# ------------------- export
def cmd_export(args: argparse.Namespace) -> int:
    crag = _build_chimera(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    triples = [
        {
            "subject": t.subject,
            "predicate": t.predicate,
            "object": t.object,
            "layer": t.layer,
            "confidence": t.confidence,
            "source_chunk_id": t.source_chunk_id,
        }
        for t in crag.graph_store.all_triples()
    ]
    (out / "triples.json").write_text(json.dumps(triples, indent=2, ensure_ascii=False))

    chunks = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "index": c.index, "text": c.text}
        for c in crag.state.chunk_lookup.values()
    ]
    (out / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))

    (out / "stats.json").write_text(json.dumps(crag.stats(), indent=2))
    print(f"exported to {out}")
    return 0


# ------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # type: ignore

        from chimera_rag.web.app import create_app
    except ImportError as e:  # pragma: no cover - optional path
        print(
            f"web extras not installed; run `uv sync --extra web` first ({e})",
            file=sys.stderr,
        )
        return 2

    crag = _build_chimera(args.config)

    # 优先复用已持久化的图谱，让前端 GraphPage 起步即可渲染，无需重新构图。
    dataset = getattr(args, "dataset", None)
    if crag.try_load_graph(dataset):
        print(
            f">>> 已加载持久化图谱（{crag.stats()['chunks']} chunk、"
            f"{crag.stats()['triples']} 三元组），可视化数据就绪。",
            file=sys.stderr,
        )
    else:
        print(
            ">>> 未找到匹配的持久化图谱，图谱页面将为空。"
            "可先用 `infer`/`ingest` 构图落盘（必要时带 --dataset 与本次一致），"
            "或在 Web UI 的 Ingest 页面构图。",
            file=sys.stderr,
        )

    app = create_app(crag)
    host = args.host or crag.config.web.host
    port = args.port or crag.config.web.port
    uvicorn.run(app, host=host, port=port)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(level=getattr(args, "log_level", "INFO"), format="rich")

    func: Callable[[argparse.Namespace], int] = args.func
    try:
        return func(args)
    except ChimeraError as e:
        logging.getLogger(__name__).error("%s: %s", type(e).__name__, e)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
