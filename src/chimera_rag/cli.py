"""Command-line entry point for Chimera-RAG.

``main.py`` at the repo root imports :func:`main` from here. Six
subcommands are dispatched via :mod:`argparse`:

* ``ingest``    — build the knowledge graph from a configured dataset
* ``query``     — single-shot question answering
* ``evaluate``  — batch run a QA dataset with EM/F1 metrics
* ``benchmark`` — compare multiple configs on the same dataset
* ``export``    — dump graph/vectors/stats to ``output/``
* ``serve``     — start the FastAPI + React web UI

Each subcommand is intentionally small and delegates to the ChimeraRAG
facade / dataset / evaluator modules.
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

    # evaluate ----------------------------------------------------------
    p_e = sub.add_parser("evaluate", help="batch-evaluate on a QA dataset")
    _add_common(p_e)
    p_e.add_argument("--dataset", required=True)
    p_e.add_argument("--limit", type=int, default=None)
    p_e.add_argument("--out", default=None, help="path to write JSON results")
    p_e.set_defaults(func=cmd_evaluate)

    # benchmark ---------------------------------------------------------
    p_b = sub.add_parser("benchmark", help="compare multiple configs on one dataset")
    p_b.add_argument("--configs", required=True, help="comma-separated yaml paths")
    p_b.add_argument("--dataset", required=True)
    p_b.add_argument("--limit", type=int, default=None)
    p_b.add_argument("--out", default=None)
    p_b.add_argument("--log-level", default="INFO")
    p_b.set_defaults(func=cmd_benchmark)

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
    # Import lazily to avoid pulling heavy deps for --help invocations.
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
    kwargs: dict = {"path": ds_cfg.path}
    if ds_cfg.eval_field_mapping:
        kwargs["eval_field_mapping"] = ds_cfg.eval_field_mapping
    return loader_cls(**kwargs)


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


# ------------------- evaluate
def cmd_evaluate(args: argparse.Namespace) -> int:
    from chimera_rag.evaluation import Evaluator

    crag = _build_chimera(args.config)
    loader = _load_dataset_from_config(crag.config, args.dataset)
    examples = loader.load_examples(limit=args.limit)
    if not examples:
        print(f"dataset {args.dataset!r} yielded no examples", file=sys.stderr)
        return 1

    # If the pipeline has an empty graph, attempt to ingest documents first.
    if crag.stats()["chunks"] == 0:
        try:
            docs = loader.load_documents()
            if docs:
                crag.ingest(docs)
        except NotImplementedError:
            pass  # dataset does not provide documents; proceed regardless

    result = Evaluator(metrics=crag.config.evaluation.metrics).run(
        pipeline=crag,
        dataset_examples=examples,
        config_name=Path(args.config).stem,
    )
    payload = result.model_dump()
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
    else:
        print(json.dumps({"config": result.config_name, "metrics": result.metrics}, indent=2))
    return 0


# ------------------- benchmark
def cmd_benchmark(args: argparse.Namespace) -> int:
    from chimera_rag.evaluation import Evaluator

    config_paths = [c.strip() for c in args.configs.split(",") if c.strip()]
    results = []
    for cfg in config_paths:
        crag = _build_chimera(cfg)
        loader = _load_dataset_from_config(crag.config, args.dataset)
        examples = loader.load_examples(limit=args.limit)
        try:
            docs = loader.load_documents()
            if docs:
                crag.ingest(docs)
        except NotImplementedError:
            pass
        r = Evaluator(metrics=crag.config.evaluation.metrics).run(
            pipeline=crag, dataset_examples=examples, config_name=Path(cfg).stem
        )
        results.append({"config": r.config_name, "metrics": r.metrics})

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


# ------------------- export
def cmd_export(args: argparse.Namespace) -> int:
    crag = _build_chimera(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # graph triples
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

    # chunks
    chunks = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "index": c.index, "text": c.text}
        for c in crag.state.chunk_lookup.values()
    ]
    (out / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))

    # stats
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
