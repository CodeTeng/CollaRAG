"""Load ``.env`` from the current working directory into ``os.environ``.

Used at :class:`ChimeraRAG` construction time so configs that reference
``api_key_env: DEEPSEEK_API_KEY`` pick up values from a local ``.env``
without the user having to ``export`` manually.

Existing values in ``os.environ`` take precedence — we never overwrite
a variable the user has already exported explicitly.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env_file(path: str | Path | None = None) -> bool:
    """Load a ``.env`` file into ``os.environ`` (non-destructive).

    Returns ``True`` if a file was loaded, ``False`` otherwise (no file,
    or dotenv unavailable). Never raises on missing file / bad values.
    """
    p = Path(path) if path else Path.cwd() / ".env"
    if not p.is_file():
        return False

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:  # pragma: no cover - python-dotenv is a main dep
        logger.warning("python-dotenv not installed; .env at %s ignored", p)
        return False

    # override=False ensures os.environ already-set values are preserved.
    load_dotenv(dotenv_path=p, override=False)
    logger.debug("loaded environment variables from %s", p)
    return True


__all__ = ["load_env_file"]
