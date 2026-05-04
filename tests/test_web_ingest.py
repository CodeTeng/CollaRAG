"""Tests for :mod:`chimera_rag.web.routers.ingest`."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client(mock_vanilla_yaml: str, tmp_path: Path):
    from fastapi.testclient import TestClient

    from chimera_rag import ChimeraRAG
    from chimera_rag.web.app import create_app

    raw = Path(mock_vanilla_yaml).read_text()
    raw = raw.replace(
        "./storage/vanilla",
        str(tmp_path / "storage").replace("\\", "/"),
    )
    p = tmp_path / "config.yaml"
    p.write_text(raw)
    crag = ChimeraRAG.from_config(p)
    return TestClient(create_app(crag))


def test_ingest_by_named_dataset_from_config(client):
    r = client.post("/api/ingest", json={"dataset": "sample"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["stats"]["documents"] >= 1
    assert data["state"]["chunks"] >= 1


def test_ingest_inline_documents(client):
    r = client.post(
        "/api/ingest",
        json={
            "documents": [
                {"doc_id": "d1", "content": "Alfred Nobel invented dynamite."},
                {"doc_id": "d2", "content": "Einstein formulated relativity."},
            ]
        },
    )
    assert r.status_code == 200
    assert r.json()["stats"]["documents"] == 2


def test_ingest_rejects_missing_source(client):
    r = client.post("/api/ingest", json={})
    assert r.status_code == 400
    assert "dataset" in r.text.lower() or "documents" in r.text.lower()


def test_ingest_unknown_dataset_returns_404(client):
    r = client.post("/api/ingest", json={"dataset": "not_in_config"})
    assert r.status_code == 404
