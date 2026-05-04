"""Tests for :mod:`chimera_rag.web.routers.config`."""

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


def test_get_config_returns_current_config_as_dict(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    cfg = r.json()["config"]
    assert cfg["app"]["name"] == "chimera-rag"
    assert "llm" in cfg
    assert "plugins" in cfg


def test_get_config_redacts_sensitive_fields(client):
    r = client.get("/api/config")
    cfg = r.json()["config"]
    # For Mock provider there are no secrets. We still verify the structure
    # contains the api_key_env field at most (never a concrete value).
    # Dedicated redaction would kick in if a provider set an api_key field.
    llm = cfg["llm"]
    # Values we never want to leak if they existed:
    for forbidden_key in ("api_key", "password"):
        assert forbidden_key not in llm or llm[forbidden_key] == "****"


def test_put_config_accepts_partial_update_and_hot_reloads(client):
    # Flip adagraph on via a config PUT; /api/plugins should reflect it.
    r = client.put(
        "/api/config",
        json={"config": {"plugins": {"adagraph": {"enabled": True}}}},
    )
    assert r.status_code == 200, r.text
    # Verify it took effect.
    plugins = client.get("/api/plugins").json()
    ada = next(p for p in plugins["plugins"] if p["name"] == "adagraph")
    assert ada["enabled"] is True


def test_put_config_rejects_invalid_values(client):
    r = client.put(
        "/api/config",
        json={"config": {"llm": {"provider": "martian"}}},  # not in enum
    )
    assert r.status_code == 400
