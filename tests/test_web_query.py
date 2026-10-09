"""Tests for :mod:`chimera_rag.web.routers.query`."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client(mock_vanilla_yaml: str, tmp_path: Path):
    from fastapi.testclient import TestClient

    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Document
    from chimera_rag.web.app import create_app

    raw = Path(mock_vanilla_yaml).read_text()
    raw = raw.replace(
        "./storage/vanilla",
        str(tmp_path / "storage").replace("\\", "/"),
    )
    p = tmp_path / "config.yaml"
    p.write_text(raw)
    crag = ChimeraRAG.from_config(p)
    # Pre-ingest so we have something to retrieve.
    crag.ingest([Document(doc_id="d0", content="Alfred Nobel invented dynamite in 1867.")])
    return TestClient(create_app(crag))


def test_query_returns_200_with_answer_payload(client):
    r = client.post(
        "/api/query",
        json={"text": "Who invented dynamite?"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data["answer"], str)
    assert data["intent"] in {
        None,
        "factual", "analytical", "comparative",
        "multi_hop", "exploratory", "follow_up",
    }
    assert "evidence_chunk_ids" in data
    assert "chunks" in data
    assert isinstance(data["chunks"], list)


def test_query_rejects_missing_text(client):
    r = client.post("/api/query", json={})
    assert r.status_code == 422  # Pydantic validation error


def test_query_returns_retrieved_chunks_with_text(client):
    r = client.post("/api/query", json={"text": "dynamite"})
    data = r.json()
    # Under vanilla config we retrieved the one doc we ingested.
    assert len(data["chunks"]) >= 1
    assert all("text" in c for c in data["chunks"])


def test_query_passes_top_k_parameter(client):
    r = client.post("/api/query", json={"text": "dynamite", "top_k": 1})
    assert r.status_code == 200
    data = r.json()
    assert len(data["chunks"]) <= 1
