"""Tests for :mod:`chimera_rag.web.routers.plugins` (list + toggle)."""

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


def test_plugins_list_returns_status_for_both_innovations(client):
    r = client.get("/api/plugins")
    assert r.status_code == 200
    data = r.json()
    names = {p["name"] for p in data["plugins"]}
    assert names == {"adagraph", "colla_rag"}
    # Initially both disabled under mock_vanilla.yaml.
    for p in data["plugins"]:
        assert p["enabled"] is False
    assert "active_implementations" in data


def test_plugins_toggle_flips_enabled_flag(client):
    r = client.post("/api/plugins/toggle", json={"name": "adagraph", "enabled": True})
    assert r.status_code == 200
    # After toggle, querying the plugin list shows it enabled.
    r2 = client.get("/api/plugins")
    ada = next(p for p in r2.json()["plugins"] if p["name"] == "adagraph")
    assert ada["enabled"] is True


def test_plugins_toggle_unknown_name_returns_404(client):
    r = client.post("/api/plugins/toggle", json={"name": "nope", "enabled": True})
    assert r.status_code == 404


def test_plugins_toggle_rebuilds_active_implementations_after_flip(client):
    # Initially chunker active is defaults.fixed (vanilla config)
    r1 = client.get("/api/plugins")
    assert r1.json()["active_implementations"]["chunker"] == "defaults.fixed"

    # Flip adagraph on and ask again; the facade rebuilds the pipeline
    # using the adagraph.* slots.
    client.post("/api/plugins/toggle", json={"name": "adagraph", "enabled": True})
    r2 = client.get("/api/plugins")
    assert r2.json()["active_implementations"]["chunker"].startswith("adagraph.")
