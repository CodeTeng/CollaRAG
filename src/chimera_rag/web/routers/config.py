"""GET /api/config + PUT /api/config (deep-merge partial update + hot reload)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from chimera_rag import ChimeraRAG
from chimera_rag.core.config import AppConfig
from chimera_rag.web.deps import get_chimera
from chimera_rag.web.schemas import ConfigResponse, ConfigUpdateRequest

router = APIRouter(tags=["config"])


# Fields considered sensitive: always redacted on GET; preserved on PUT
# when the caller sends the redacted value back untouched.
_REDACTED = "****"
_SENSITIVE_KEY_NAMES = {"api_key", "password"}


def _redact(obj: Any) -> Any:
    """Recursively redact any dict key whose lowercased form contains a sensitive name."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if any(s in k.lower() for s in _SENSITIVE_KEY_NAMES):
                out[k] = _REDACTED if v else v
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for k, v in overlay.items():
        if (
            k in merged
            and isinstance(merged[k], dict)
            and isinstance(v, dict)
        ):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


@router.get("/config", response_model=ConfigResponse)
def get_config(chimera: ChimeraRAG = Depends(get_chimera)) -> ConfigResponse:
    raw = chimera.config.model_dump(mode="python")
    return ConfigResponse(config=_redact(raw))


@router.put("/config")
def put_config(
    body: ConfigUpdateRequest,
    chimera: ChimeraRAG = Depends(get_chimera),
) -> dict:
    current = chimera.config.model_dump(mode="python")
    new_raw = _deep_merge(current, body.config)

    try:
        new_cfg = AppConfig.model_validate(new_raw)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors()) from e

    # Swap the config and rebuild pipelines while preserving state.
    chimera.config = new_cfg
    chimera._rebuild_pipelines_preserving_state()
    return {"ok": True, "active_implementations": chimera.active_implementations()}


__all__ = ["router"]
