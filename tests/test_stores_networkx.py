"""Tests for :mod:`chimera_rag.stores.graph`."""

from __future__ import annotations

from pathlib import Path


def test_networkx_store_add_triple_and_all_triples():
    from chimera_rag.core.types import Triple
    from chimera_rag.stores.graph import NetworkXGraphStore

    s = NetworkXGraphStore()
    s.add_triple(Triple(subject="A", predicate="invented", object="B"))
    s.add_triple(Triple(subject="B", predicate="is_a", object="C"))
    triples = s.all_triples()
    assert len(triples) == 2
    assert {t.subject for t in triples} == {"A", "B"}


def test_networkx_store_get_neighbors_one_hop_finds_inbound_and_outbound():
    from chimera_rag.core.types import Triple
    from chimera_rag.stores.graph import NetworkXGraphStore

    s = NetworkXGraphStore()
    s.add_triple(Triple(subject="Nobel", predicate="invented", object="dynamite"))
    s.add_triple(Triple(subject="Sweden", predicate="birthplace_of", object="Nobel"))
    s.add_triple(Triple(subject="Nobel", predicate="founded", object="NobelPrize"))

    out = s.get_neighbors("Nobel", hops=1)
    subjects = {t.subject for t in out}
    objects = {t.object for t in out}
    # all three edges touch Nobel (2 outbound + 1 inbound)
    assert len(out) == 3
    assert "Nobel" in subjects or "Nobel" in objects


def test_networkx_store_get_neighbors_two_hops_expands_frontier():
    from chimera_rag.core.types import Triple
    from chimera_rag.stores.graph import NetworkXGraphStore

    s = NetworkXGraphStore()
    s.add_triple(Triple(subject="A", predicate="p", object="B"))
    s.add_triple(Triple(subject="B", predicate="p", object="C"))
    s.add_triple(Triple(subject="C", predicate="p", object="D"))

    one = s.get_neighbors("A", hops=1)
    two = s.get_neighbors("A", hops=2)
    # hops=2 must strictly include everything hops=1 returns
    assert len(two) >= len(one)
    assert any(t.subject == "B" and t.object == "C" for t in two)


def test_networkx_store_persist_and_load_round_trip(tmp_path: Path):
    from chimera_rag.core.types import Triple
    from chimera_rag.stores.graph import NetworkXGraphStore

    s1 = NetworkXGraphStore()
    s1.add_triple(Triple(subject="A", predicate="p", object="B", confidence=0.9, layer="L2"))
    s1.add_triple(Triple(subject="B", predicate="p", object="C"))
    p = tmp_path / "g.gpickle"
    s1.persist(str(p))
    assert p.is_file()

    s2 = NetworkXGraphStore()
    s2.load(str(p))
    assert len(s2.all_triples()) == 2
    # Metadata survives the round-trip
    abc = next(t for t in s2.all_triples() if t.subject == "A")
    assert abc.confidence == 0.9
    assert abc.layer == "L2"
