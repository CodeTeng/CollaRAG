"""Chimera-RAG unified entry point.

All five subcommands (ingest / query / infer / export / serve)
are dispatched from here to :mod:`chimera_rag.cli`.

Run ``python main.py <cmd> --help`` for details. Each subcommand is
implemented via TDD in the :mod:`chimera_rag.cli` module.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Real CLI wiring lands in chimera_rag.cli (tested separately).
    # For now, keep main.py a thin shim so the file exists and is importable.
    try:
        from chimera_rag.cli import main as cli_main
    except ImportError:  # pragma: no cover - handled by TDD tests later
        print("chimera_rag.cli is not yet implemented.", file=sys.stderr)
        return 1
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
