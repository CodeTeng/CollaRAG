"""Tests for the CLI via subprocess, plus unit-level cli.main()."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str], cwd: Path = REPO_ROOT, env=None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "main.py", *args]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)


def test_main_no_args_shows_help_and_exits_nonzero():
    r = _run([])
    # argparse exits with code 2 and writes usage to stderr when required
    # subcommand is missing.
    assert r.returncode != 0
    assert "usage" in r.stderr.lower() or "usage" in r.stdout.lower()


def test_main_help_lists_all_subcommands():
    r = _run(["--help"])
    assert r.returncode == 0
    out = r.stdout + r.stderr
    for sub in ["ingest", "query", "infer", "export", "serve"]:
        assert sub in out
    # evaluate/benchmark are intentionally removed; scoring lives in experiments/eval.py now.
    assert "evaluate" not in out
    assert "benchmark" not in out


@pytest.mark.integration
def test_main_ingest_runs_end_to_end_with_vanilla_config(
    mock_vanilla_yaml: str, tmp_path: Path
):
    r = _run(["ingest", "--config", mock_vanilla_yaml, "--dataset", "sample"])
    assert r.returncode == 0, f"stderr={r.stderr}\nstdout={r.stdout}"
    # Summary line should mention doc / chunk / triple counts.
    out = r.stdout + r.stderr
    assert any(k in out.lower() for k in ["chunks", "triples", "ingest"])
