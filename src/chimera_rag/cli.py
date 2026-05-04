"""Command-line entry point for Chimera-RAG.

``main.py`` at the repo root imports :func:`main` from here.
Subcommands are dispatched via :mod:`argparse`:

* ``ingest``  — build the knowledge graph from a configured dataset
* ``query``   — single-shot question answering
* ``infer``   — batch-run a QA dataset and write predictions to disk
                (no metric calculation; use ``eval.py`` for that)
* ``export``  — dump graph/vectors/stats to ``output/``
* ``serve``   — start the FastAPI + React web UI

Evaluation is now split into two stages:

* ``main.py infer`` produces predictions.jsonl + meta.json
* ``eval.py``       reads those, computes metrics, and renders the report

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
    docs = loader.load_documents()
    if not docs:
        print(f"dataset {args.dataset!r} yielded no documents", file=sys.stderr)
        return 1
    stats = crag.ingest(docs)
    print(json.dumps({"ingest": stats, "state": crag.stats()}, indent=2))
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
def cmd_infer(args: argparse.Namespace) -> int:
    from chimera_rag.evaluation import InferenceRunner, write_predictions_dir

    crag = _build_chimera(args.config)
    loader = _load_dataset_from_config(crag.config, args.dataset)
    examples = loader.load_examples(limit=args.limit)
    if not examples:
        print(f"dataset {args.dataset!r} yielded no examples", file=sys.stderr)
        return 1

    # 如果图谱尚未注入，尝试用同一份 dataset 的 documents 注入一次
    if crag.stats()["chunks"] == 0:
        try:
            docs = loader.load_documents()
            if docs:
                crag.ingest(docs)
        except NotImplementedError:
            pass  # 部分 loader 只提供 QA，跳过

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
