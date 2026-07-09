"""Structural evidence-completeness checker for G-PER's Reflect phase.

The central novelty of G-PER over scalar-gate reflection (Reflexion-style
verbal self-reflection) is that *sufficiency of evidence* is reframed as a
**structural constraint-satisfaction check on the evidence subgraph**, not a
scalar quality score. Given a query and the triples gathered so far, this
module:

1. infers the *required evidence shape* implied by the query's compositionality
   (comparison / bridge / attribute), and
2. diffs that shape against the gathered evidence subgraph, emitting **typed
   gaps** (``missing_attribute`` / ``missing_path`` / ``missing_relation``)
   each carrying a targeted repair query — so the supplementary retrieval
   round fixes a *specific* missing edge rather than issuing a vague "search
   more".

When the graph store is absent (graph-poor setting), the checker degrades to
``ShapeSpec(compositionality="unknown")`` and emits no structural gaps, so
G-PER falls back to text-only PER behavior.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from chimera_rag.core.types import Triple

logger = logging.getLogger(__name__)

Compositionality = Literal["comparison", "bridge", "attribute", "unknown"]
GapKind = Literal["missing_attribute", "missing_path", "missing_relation", "missing_entity"]

_COMPARATOR_RE = (
    r"\b(compare|comparison|differ(?:ence)?|versus|vs\.?|similar|alike|"
    r"both|whereas|while|than|between)\b"
)
_BRIDGE_CUES = (
    "who", "whose", "which", "that", "led to", "caused", "resulted",
    "because", "therefore", "followed by", "then",
    "related", "relation", "connection", "connected", "link", "linked",
    "associate", "associated", "between",
)


@dataclass
class ShapeSpec:
    """The evidence shape a query requires for a structurally complete answer."""

    compositionality: Compositionality = "unknown"
    entities: list[str] = field(default_factory=list)
    # (subject, predicate, object) triples the answer needs; predicate "*" = any.
    required_relations: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def needs_graph(self) -> bool:
        return self.compositionality in ("comparison", "bridge")


@dataclass
class EvidenceGap:
    """A typed, repairable structural gap in the gathered evidence."""

    kind: GapKind
    detail: str
    repair_query: str
    repair_entities: list[str] = field(default_factory=list)
    severity: float = 0.5  # 0..1, used to prioritize repair


class StructuralCompletenessChecker:
    """Infers the required evidence shape and diffs it against gathered triples."""

    def __init__(self, graph_store=None) -> None:
        self._gs = graph_store

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------
    def infer_shape(self, query: str, entities: list[str]) -> ShapeSpec:
        """Heuristic compositionality inference from query phrasing + entities.

        We deliberately keep this rule-based and cheap: the checker runs every
        reflection round, and an LLM call here would defeat the cost budget
        that Reflect is supposed to protect. The shape only needs to be coarse
        enough to drive typed-gap detection.
        """
        import re

        q = query.lower()
        ents = [e for e in entities if e]
        is_comparison = bool(re.search(_COMPARATOR_RE, q)) and len(ents) >= 2
        is_bridge = (not is_comparison) and len(ents) >= 2 and any(
            cue in q for cue in _BRIDGE_CUES
        )

        if is_comparison:
            return ShapeSpec(compositionality="comparison", entities=ents)
        if is_bridge:
            return ShapeSpec(
                compositionality="bridge", entities=ents,
                required_relations=[(ents[0], "*", ents[1])],
            )
        if len(ents) >= 1:
            return ShapeSpec(compositionality="attribute", entities=ents)
        return ShapeSpec(compositionality="unknown", entities=ents)

    # ------------------------------------------------------------------
    # Structural diff
    # ------------------------------------------------------------------
    def check(
        self,
        query: str,
        entities: list[str],
        evidence_triples: list[Triple],
    ) -> list[EvidenceGap]:
        """Diff the required shape against the gathered evidence subgraph.

        Returns a list of typed gaps; an empty list means the evidence is
        structurally complete. Each gap carries a targeted ``repair_query``
        and the entities the repair should aim at.
        """
        shape = self.infer_shape(query, entities)
        if shape.compositionality == "unknown":
            return []

        ev_by_subj: dict[str, list[Triple]] = {}
        ev_nodes: set[str] = set()
        for t in evidence_triples:
            ev_by_subj.setdefault(t.subject, []).append(t)
            ev_nodes.add(t.subject)
            ev_nodes.add(t.object)

        gaps: list[EvidenceGap] = []

        if shape.compositionality == "comparison":
            # Each compared entity must have at least one attribute in evidence.
            for e in shape.entities:
                if e not in ev_by_subj and e not in ev_nodes:
                    gaps.append(EvidenceGap(
                        kind="missing_attribute", detail=f"no attribute for '{e}'",
                        repair_query=f"What are the attributes of {e}?",
                        repair_entities=[e], severity=0.8,
                    ))
                elif e not in ev_by_subj:
                    gaps.append(EvidenceGap(
                        kind="missing_attribute",
                        detail=f"'{e}' appears only as object; no outgoing attribute",
                        repair_query=f"What are the properties of {e}?",
                        repair_entities=[e], severity=0.5,
                    ))
            return gaps

        if shape.compositionality == "bridge":
            a, b = shape.entities[0], shape.entities[-1]
            connected = self._evidence_connects(a, b, evidence_triples)
            if not connected:
                # Can the *full* KG supply a path that retrieval missed?
                kg_path: list[Triple] = []
                if self._gs is not None:
                    kg_path = self._gs.shortest_path(a, b, max_hops=4)
                if kg_path:
                    missing_edge = kg_path[0]
                    gaps.append(EvidenceGap(
                        kind="missing_relation",
                        detail=(
                            f"evidence lacks the bridge {a} -> {b}; KG has a path "
                            f"via ({missing_edge.subject}, {missing_edge.predicate}, "
                            f"{missing_edge.object}) that was not retrieved"
                        ),
                        repair_query=(
                            f"What is the relationship between {a} and {b}?"
                        ),
                        repair_entities=[a, b], severity=0.9,
                    ))
                else:
                    gaps.append(EvidenceGap(
                        kind="missing_path",
                        detail=f"no evidence path connecting '{a}' to '{b}'",
                        repair_query=(
                            f"How is {a} related to {b}? Trace the connection."
                        ),
                        repair_entities=[a, b], severity=0.7,
                    ))
            return gaps

        # attribute: entity must have an outgoing triple in evidence.
        e = shape.entities[0]
        if e and e not in ev_by_subj:
            gaps.append(EvidenceGap(
                kind="missing_attribute", detail=f"no attribute for '{e}'",
                repair_query=f"What are the attributes of {e}?",
                repair_entities=[e], severity=0.6,
            ))
        return gaps

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _evidence_connects(a: str, b: str, triples: list[Triple]) -> bool:
        """True iff the gathered triples connect ``a`` to ``b`` (undirected)."""
        import networkx as nx

        g = nx.Graph()
        for t in triples:
            g.add_edge(t.subject, t.object)
        if a not in g or b not in g:
            return False
        try:
            return nx.has_path(g, a, b)
        except nx.NetworkXError:
            return False


__all__ = ["EvidenceGap", "ShapeSpec", "StructuralCompletenessChecker"]
