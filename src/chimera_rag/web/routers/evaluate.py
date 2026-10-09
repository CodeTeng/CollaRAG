"""POST /api/evaluate — batch evaluation on a named dataset."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from chimera_rag import ChimeraRAG
from chimera_rag.core.exceptions import DatasetError
from chimera_rag.datasets import get_dataset_loader
from chimera_rag.evaluation import InferenceRunner, Scorer
from chimera_rag.web.deps import get_chimera
from chimera_rag.web.schemas import EvaluateRequest, EvaluateResponse

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(
    body: EvaluateRequest,
    chimera: ChimeraRAG = Depends(get_chimera),
) -> EvaluateResponse:
    ds_cfg = chimera.config.datasets.get(body.dataset)
    if ds_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"dataset {body.dataset!r} not declared in config.datasets",
        )
    try:
        loader_cls = get_dataset_loader(ds_cfg.loader)
    except DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    kwargs: dict = {"path": ds_cfg.path}
    if ds_cfg.eval_field_mapping:
        kwargs["eval_field_mapping"] = ds_cfg.eval_field_mapping

    try:
        loader = loader_cls(**kwargs)
        examples = loader.load_examples(limit=body.limit)
    except DatasetError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except NotImplementedError:
        raise HTTPException(
            status_code=400,
            detail=f"loader {ds_cfg.loader!r} does not support load_examples()",
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e)) from e

    # The old monolithic ``Evaluator`` was split (commit c82aeaf) into an
    # inference stage (InferenceRunner) and an offline scoring stage (Scorer).
    # Recompose them here: run the pipeline over the examples, then score.
    runner = InferenceRunner()
    per_example, meta = runner.run(
        pipeline=chimera,
        dataset_examples=examples,
        config_name=body.dataset,
        show_progress=False,
    )
    result = Scorer(metrics=chimera.config.evaluation.metrics).score(
        per_example=per_example,
        config_name=body.dataset,
        usage=meta.get("usage_total"),
        elapsed_s=meta.get("elapsed_s", 0.0),
    )
    return EvaluateResponse(
        config_name=result.config_name,
        metrics=result.metrics,
        per_example=result.per_example,
    )


__all__ = ["router"]
