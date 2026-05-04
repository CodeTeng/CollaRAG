"""Pre-download sentence-transformers weights into the project's ``./models`` dir.

Usage::

    uv run python scripts/download_embedding_model.py

After a successful run, the model path
``./models/sentence-transformers_all-MiniLM-L6-v2`` is populated with a
fully self-contained model (config.json, tokenizer files, pytorch_model.bin
or model.safetensors, etc). Configs can then point at it with::

    embedding:
      provider: sentence_transformer
      model: ./models/sentence-transformers_all-MiniLM-L6-v2

The script is idempotent — if the directory already contains a usable
model, it exits immediately without re-downloading.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "models" / MODEL_REPO.replace("/", "_")


def already_downloaded(target: Path) -> bool:
    """Return True if the target directory looks like a complete ST model."""
    if not target.is_dir():
        return False
    required = ("config.json", "tokenizer.json")
    return all((target / f).is_file() for f in required)


def main() -> int:
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if already_downloaded(TARGET_DIR):
        print(f"✓ model already present at {TARGET_DIR}")
        return 0

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as e:
        print(
            f"ERROR: sentence-transformers not installed: {e}\n"
            "Run `uv sync --extra st` first.",
            file=sys.stderr,
        )
        return 2

    print(f"downloading {MODEL_REPO} -> {TARGET_DIR} (~90MB, one-time)")
    model = SentenceTransformer(MODEL_REPO)
    model.save(str(TARGET_DIR))

    if not already_downloaded(TARGET_DIR):
        print(
            f"ERROR: download completed but {TARGET_DIR} looks incomplete",
            file=sys.stderr,
        )
        return 3

    print(f"✓ saved model to {TARGET_DIR}")
    print("\nTo use it, set in your config.yaml:")
    print(f"  embedding.model: {TARGET_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
