"""推理产物 (predictions.jsonl + meta.json) 的读写工具。

目录约定::

    <out_dir>/
        predictions.jsonl   # 每行一条样本，UTF-8
        meta.json           # 运行级元信息

该模块**只做 I/O**，不做指标计算，也不调用 pipeline。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREDICTIONS_FILE = "predictions.jsonl"
META_FILE = "meta.json"


def write_predictions_dir(
    out_dir: str | Path,
    *,
    per_example: list[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Path]:
    """把推理结果写入 ``out_dir`` 下的 predictions.jsonl + meta.json。

    返回 ``{"predictions": Path, "meta": Path, "dir": Path}``。
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)

    pred_path = d / PREDICTIONS_FILE
    with pred_path.open("w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    meta_path = d / META_FILE
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"dir": d, "predictions": pred_path, "meta": meta_path}


def read_predictions_dir(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取推理产物。

    参数 ``path`` 可以是目录（按约定读 predictions.jsonl + meta.json），
    也可以直接是 predictions.jsonl 文件路径（此时 meta 尝试同目录读取）。
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".jsonl":
        pred_path = p
        meta_path = p.parent / META_FILE
    else:
        pred_path = p / PREDICTIONS_FILE
        meta_path = p / META_FILE

    if not pred_path.is_file():
        raise FileNotFoundError(
            f"predictions file not found: {pred_path} "
            f"(expected a dir containing {PREDICTIONS_FILE} or a .jsonl file)"
        )

    rows: list[dict[str, Any]] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for ln_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"malformed JSONL at {pred_path}:{ln_no}: {e}"
                ) from e

    meta: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"malformed meta.json at {meta_path}: {e}") from e
    return rows, meta


__all__ = [
    "META_FILE",
    "PREDICTIONS_FILE",
    "read_predictions_dir",
    "write_predictions_dir",
]
