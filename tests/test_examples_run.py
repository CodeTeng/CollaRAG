"""Tests that each bundled example script runs end-to-end under MockProviders."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parent.parent


def _run(script: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {
        **_env(),
        "CHIMERA_WORKSPACE": str(tmp_path),  # examples honour this to isolate storage
    }
    return subprocess.run(
        [sys.executable, str(REPO / "examples" / script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _env() -> dict[str, str]:
    import os

    return {**os.environ}


@pytest.mark.parametrize(
    "script",
    [
        "demo_vanilla.py",
        "demo_adagraph.py",
        "demo_intent_agent.py",
        "demo_full.py",
    ],
)
def test_example_script_runs_successfully(script: str, tmp_path: Path):
    r = _run(script, tmp_path)
    assert r.returncode == 0, (
        f"{script} failed (rc={r.returncode})\n"
        f"STDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    # Each demo prints an answer; verify some output was emitted.
    combined = (r.stdout + r.stderr).lower()
    assert "answer" in combined or "chunks" in combined
