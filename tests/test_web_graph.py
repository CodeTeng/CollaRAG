"""Tests for :mod:`chimera_rag.web.routers.graph`."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client_with_triples(mock_vanilla_yaml: str, tmp_path: Path):
    from fastapi.testclient import TestClient

    from chimera_rag import ChimeraRAG
    from chimera_rag.core.types import Triple
    from chimera_rag.web.app import create_app

    raw = Path(mock_vanilla_yaml).read_text()
    raw = raw.replace(
        "./storage/vanilla",
        str(tmp_path / "storage").replace("\\", "/"),
    )
    p = tmp_path / "config.yaml"
    p.write_text(raw)
    crag = ChimeraRAG.from_config(p)
    # Seed triples directly.
    for t in [
        Triple(subject="Nobel", predicate="invented", object="Dynamite"),
        Triple(subject="Nobel", predicate="founded", object="NobelPrize"),
        Triple(subject="Einstein", predicate="formulated", object="Relativity"),
    ]:
        crag.graph_store.add_triple(t)
    return TestClient(create_app(crag))


def test_graph_returns_nodes_and_edges(client_with_triples):
    r = client_with_triples.get("/api/graph")
    assert r.status_code == 200
    data = r.json()
    # 4 unique entities: Nobel, Dynamite, NobelPrize, Einstein, Relativity.
    assert len(data["nodes"]) == 5
    assert len(data["edges"]) == 3


def test_graph_stats_returns_counts(client_with_triples):
    r = client_with_triples.get("/api/graph/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"] == 5
    assert data["edges"] == 3
    assert "chunks" in data
    assert "triples" in data


def test_graph_filter_by_entity_returns_subgraph(client_with_triples):
    r = client_with_triples.get("/api/graph?entity=Nobel")
    assert r.status_code == 200
    data = r.json()
    # Only edges touching Nobel.
    assert len(data["edges"]) == 2
    # Node list should contain Nobel + its neighbours only.
    node_ids = {n["id"] for n in data["nodes"]}
    assert "Nobel" in node_ids
    assert "Einstein" not in node_ids
