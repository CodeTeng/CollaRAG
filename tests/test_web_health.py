"""Tests for the FastAPI app factory + basic health endpoint.

We exercise every web route through FastAPI's TestClient so no real
server is needed. ChimeraRAG is built with a tiny vanilla config that
uses Mock providers + a tmp_path workspace so tests are hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def test_chimera(mock_vanilla_yaml: str, tmp_path: Path):
    """Build a ChimeraRAG with mock_vanilla.yaml rerouted under tmp_path."""
    from chimera_rag import ChimeraRAG

    raw = Path(mock_vanilla_yaml).read_text()
    raw = raw.replace(
        "./storage/vanilla",
        str(tmp_path / "storage").replace("\\", "/"),
    )
    p = tmp_path / "config.yaml"
    p.write_text(raw)
    return ChimeraRAG.from_config(p)


@pytest.fixture()
def client(test_chimera):
    from fastapi.testclient import TestClient

    from chimera_rag.web.app import create_app

    app = create_app(test_chimera)
    return TestClient(app)


def test_create_app_returns_fastapi_instance(test_chimera):
    from fastapi import FastAPI

    from chimera_rag.web.app import create_app

    app = create_app(test_chimera)
    assert isinstance(app, FastAPI)


def test_health_returns_200_with_status_payload(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data
