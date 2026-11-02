"""W4 — the defect-driven repair loop (Phase 3 + 4), PrismRAG contribution 3.

Two things are pinned here, and the first one is the reason the contribution
exists at all.

**Risk R1** (``docs/innovation-1/design.md`` §8): if "repair" meant extracting
the same span again and unioning the results, PrismRAG would be indistinguishable
from the ``extract-twice-union`` counter-baseline that E2 mandates. Every action
in :mod:`chimera_rag.plugins.prism.repair` is therefore a *boundary change*, and
the headline invariant below states that directly: a repair round never changes
**what** text the extractor sees or **what** text the vector store sees — only
how that text is cut.

**Risk R3**: the loop amplifies detector noise, so a defect the detector could
not falsify carries ``confidence="low"`` and is recorded but never acted on.
``ChunkDefect.repair_driving`` is where that rule lives; the pipeline tests
below are what make it enforceable rather than advisory.
"""

from __future__ import annotations

import numpy as np
import pytest

from chimera_rag.core.pipeline import IngestionPipeline, PipelineState
from chimera_rag.core.types import (
    EXTRACTED_GRANULARITIES,
    INDEXED_GRANULARITIES,
    Chunk,
    ChunkDefect,
    Document,
    RepairOutcome,
    Triple,
)
from chimera_rag.defaults.pruner import NoOpPruner
from chimera_rag.interfaces.chunker import BaseChunkAuditor, BaseChunker
from chimera_rag.interfaces.extractor import BaseTripleExtractor
from chimera_rag.interfaces.graph_store import BaseGraphStore
from chimera_rag.interfaces.vector_store import BaseVectorStore
from chimera_rag.plugins.prism.features import split_sentences
from chimera_rag.plugins.prism.repair import REPAIR_PRIORITY, apply_repairs

S = ["A one.", "B two.", "C three.", "D four.", "E five.", "F six."]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _chunk_id(doc: str, index: int, granularity: str) -> str:
    """The id convention ``repair._emit`` re-derives after every round."""
    return f"{doc}::{index}" if granularity == "unified" else f"{doc}::{granularity}::{index}"


def _chunk(
    index: int,
    sentences: list[str],
    *,
    granularity: str = "unified",
    doc: str = "d",
    parent: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(doc, index, granularity),
        doc_id=doc,
        index=index,
        text=" ".join(sentences),
        granularity=granularity,
        parent_chunk_id=parent,
    )


def _unified(*groups: list[str], doc: str = "d") -> list[Chunk]:
    return [_chunk(i, g, doc=doc) for i, g in enumerate(groups)]


def _dual(doc: str = "d") -> list[Chunk]:
    """Two coarse chunks with one fine child each — the shape ``PrismChunker``
    emits, so repairs are exercised on realistic input rather than on unified
    chunks only."""
    coarse0 = _chunk(0, S[:2], granularity="coarse", doc=doc)
    coarse1 = _chunk(1, S[2:4], granularity="coarse", doc=doc)
    fine0 = _chunk(0, S[:2], granularity="fine", doc=doc, parent=coarse0.chunk_id)
    fine1 = _chunk(1, S[2:4], granularity="fine", doc=doc, parent=coarse1.chunk_id)
    return [coarse0, coarse1, fine0, fine1]


def _d(code: str, chunk_id: str, *, related=(), **metadata) -> ChunkDefect:
    from chimera_rag.core.types import DEFECT_REPAIR_ACTION

    return ChunkDefect(
        code=code,
        chunk_id=chunk_id,
        action=DEFECT_REPAIR_ACTION[code],
        detail=f"{code} on {chunk_id}",
        related_chunk_ids=list(related),
        metadata=dict(metadata),
    )


def _view(chunks, granularities) -> list[str]:
    """The sentence sequence one routing side actually sees."""
    out: list[str] = []
    for ch in chunks:
        if ch.granularity in granularities:
            out.extend(split_sentences(ch.text))
    return out


# ===========================================================================
# The R1 invariant: a repair moves boundaries, never text
# ===========================================================================
ACTIONS_UNDER_TEST = [
    ("split", _d("D5", "d::0")),
    ("split_by_topic", _d("D6", "d::0", objects=["one", "three"])),
    ("extend_left", _d("D2", "d::1", related=["d::0"])),
    ("merge_up", _d("D4", "d::0")),
    ("bind", _d("D3", "d::0", related=["d::1"])),
]


@pytest.mark.parametrize(
    ("action", "defect"), ACTIONS_UNDER_TEST, ids=[a for a, _ in ACTIONS_UNDER_TEST]
)
def test_no_action_changes_what_either_routing_side_sees(action, defect):
    """Headline R1 guard. Both the extraction view and the retrieval view hold
    the same sentences before and after the round, so whatever the loop buys
    comes from *where the cuts are*, not from feeding the extractor more text."""
    before = _unified(S[:2], S[2:4])
    outcome = apply_repairs([defect], before)
    assert outcome.defects_applied == 1
    for granularities in (EXTRACTED_GRANULARITIES, INDEXED_GRANULARITIES):
        assert _view(outcome.chunks, granularities) == _view(before, granularities)


def test_no_action_invents_text():
    """``bind`` legitimately duplicates coverage — the new coarse cover repeats
    what its fine children already hold — so the global sentence sequence is not
    conserved across every action. What must hold for all of them is that
    nothing *new* appears: a repair re-cuts the corpus, it never writes to it."""
    before = _unified(S[:2], S[2:4])
    known = set(split_sentences(" ".join(c.text for c in before)))
    for _action, defect in ACTIONS_UNDER_TEST:
        after = apply_repairs([defect], _unified(S[:2], S[2:4])).chunks
        produced = set(split_sentences(" ".join(c.text for c in after)))
        assert produced <= known, f"{defect.action} invented {produced - known}"


def test_apply_repairs_takes_no_extractor():
    """Structural form of the same claim: the repair entry point cannot call an
    extractor because it is not given one. If a future refactor adds an
    extractor parameter here, contribution 3 has quietly become
    ``extract-twice-union`` and this test should fail loudly."""
    import inspect

    assert list(inspect.signature(apply_repairs).parameters) == ["defects", "chunks"]


# ===========================================================================
# split / split_by_topic
# ===========================================================================
def test_split_partitions_one_segment_at_a_sentence_boundary():
    outcome = apply_repairs([_d("D5", "d::0")], _unified(S[:2], S[2:4]))
    assert [c.text for c in outcome.chunks] == ["A one.", "B two.", "C three. D four."]


def test_split_renumbers_the_whole_layer_rather_than_patching_ids():
    """Ids are derived from position after the round. A repair that patched the
    split segment in place would leave the surviving ``d::1`` claiming an index
    it no longer holds."""
    outcome = apply_repairs([_d("D5", "d::0")], _unified(S[:2], S[2:4]))
    assert [c.chunk_id for c in outcome.chunks] == ["d::0", "d::1", "d::2"]
    assert [c.index for c in outcome.chunks] == [0, 1, 2]


def test_split_uses_the_collision_boundary_when_d1_supplied_one():
    """D1 knows which neighbourhoods were unreachable, so the cut goes between
    them instead of at the midpoint — this is what makes the repair *directed*."""
    chunks = _unified(S[:4])
    defect = _d("D1", "d::0", component_members=[["A one.", "B two."], ["E five.", "F six."]])
    outcome = apply_repairs([defect], chunks)
    assert [c.text for c in outcome.chunks] == ["A one. B two.", "C three. D four."]


def test_split_falls_back_to_the_midpoint_without_component_info():
    outcome = apply_repairs([_d("D5", "d::0")], _unified(S[:4]))
    assert [c.text for c in outcome.chunks] == ["A one. B two.", "C three. D four."]


def test_split_on_a_single_sentence_chunk_is_deferred():
    """There is no boundary left to move. Deferring — rather than emitting an
    empty or duplicated chunk — is what lets the loop terminate."""
    outcome = apply_repairs([_d("D5", "d::0")], _unified(S[:1]))
    assert outcome.defects_applied == 0
    assert outcome.defects_deferred == 1
    assert [c.text for c in outcome.chunks] == ["A one."]


def test_split_by_topic_cuts_between_the_two_conflicting_values():
    text = ["Marie Curie was born in Warsaw.", "She later worked in Paris.", "Curie won two prizes."]
    defect = _d("D6", "d::0", objects=["Warsaw", "Paris"])
    outcome = apply_repairs([defect], _unified(text))
    assert [c.text for c in outcome.chunks] == [
        "Marie Curie was born in Warsaw.",
        "She later worked in Paris. Curie won two prizes.",
    ]


def test_split_by_topic_falls_back_to_the_midpoint_when_values_are_not_locatable():
    text = ["One claim here.", "Another claim there."]
    outcome = apply_repairs([_d("D6", "d::0", objects=["absent", "missing"])], _unified(text))
    assert [c.text for c in outcome.chunks] == ["One claim here.", "Another claim there."]


def test_splitting_a_fine_chunk_keeps_both_halves_under_the_same_parent():
    """The provenance chain must survive the split, or contribution 1 and
    contribution 3 would be working against each other."""
    outcome = apply_repairs([_d("D5", "d::fine::0")], _dual())
    fine = [c for c in outcome.chunks if c.granularity == "fine"]
    assert [c.text for c in fine] == ["A one.", "B two.", "C three. D four."]
    coarse_ids = {c.chunk_id for c in outcome.chunks if c.granularity == "coarse"}
    assert all(c.parent_chunk_id in coarse_ids for c in fine)
    assert fine[0].parent_chunk_id == fine[1].parent_chunk_id == "d::coarse::0"


# ===========================================================================
# extend_left / merge_up
# ===========================================================================
def test_extend_left_absorbs_the_predecessor():
    outcome = apply_repairs([_d("D2", "d::1", related=["d::0"])], _unified(S[:2], S[2:4]))
    assert [c.text for c in outcome.chunks] == ["A one. B two. C three. D four."]
    assert outcome.actions == {"extend_left": 1}


def test_extend_left_finds_its_predecessor_by_index_without_related_ids():
    """D2 normally names the predecessor, but the action must still work when it
    does not — the detectors and the repairs are separately configurable."""
    outcome = apply_repairs([_d("D2", "d::1")], _unified(S[:2], S[2:4]))
    assert outcome.defects_applied == 1
    assert [c.text for c in outcome.chunks] == ["A one. B two. C three. D four."]


def test_extend_left_at_the_start_of_a_document_is_deferred():
    outcome = apply_repairs([_d("D2", "d::0")], _unified(S[:2], S[2:4]))
    assert outcome.defects_applied == 0
    assert outcome.defects_deferred == 1
    assert len(outcome.chunks) == 2


def test_extend_left_on_a_coarse_chunk_keeps_its_fine_children_bound():
    """D2 audits the *retrieval* layer, so it fires on coarse chunks, which do
    have fine children. Absorbing a coarse cover must reparent them: left
    dangling they would be emitted with ``parent_chunk_id=None`` and a
    retrieval hit on the merged coarse chunk would no longer resolve to the
    triples they produced."""
    outcome = apply_repairs([_d("D2", "d::coarse::1", related=["d::coarse::0"])], _dual())
    coarse = [c for c in outcome.chunks if c.granularity == "coarse"]
    fine = [c for c in outcome.chunks if c.granularity == "fine"]
    assert len(coarse) == 1
    assert coarse[0].text == "A one. B two. C three. D four."
    assert [c.parent_chunk_id for c in fine] == [coarse[0].chunk_id] * 2


def test_merge_up_absorbs_the_successor():
    outcome = apply_repairs([_d("D4", "d::0")], _unified(S[:2], S[2:4]))
    assert [c.text for c in outcome.chunks] == ["A one. B two. C three. D four."]
    assert outcome.actions == {"merge_up": 1}


def test_merge_up_at_the_end_of_a_layer_falls_back_to_the_predecessor():
    """The point of D4 is to stop extracting this span alone; either direction
    achieves that, so refusing at the layer end would waste the signal."""
    outcome = apply_repairs([_d("D4", "d::1")], _unified(S[:2], S[2:4]))
    assert outcome.defects_applied == 1
    assert [c.text for c in outcome.chunks] == ["A one. B two. C three. D four."]


def test_merge_up_on_a_lone_chunk_is_deferred():
    outcome = apply_repairs([_d("D4", "d::0")], _unified(S[:2]))
    assert outcome.defects_applied == 0
    assert outcome.defects_deferred == 1


def test_merge_up_stays_inside_its_own_granularity_layer():
    """``index + 1`` exists twice under dual granularity; absorbing across
    layers would delete a chunk the other routing side depends on."""
    chunks = _dual()
    outcome = apply_repairs([_d("D4", "d::fine::0")], chunks)
    assert outcome.defects_applied == 1
    assert len([c for c in outcome.chunks if c.granularity == "coarse"]) == 2
    assert [c.text for c in outcome.chunks if c.granularity == "fine"] == [
        "A one. B two. C three. D four."
    ]


# ===========================================================================
# bind — the action that creates granularity where there was none
# ===========================================================================
def test_bind_turns_two_unified_chunks_into_a_coarse_cover_over_two_fine_children():
    """This is where a single-granularity document first acquires the
    dual-granularity structure. Index coverage is unchanged — the coarse chunk
    is what the vector store sees — and no text stops being extracted, because
    both originals survive as fine children rather than being folded away."""
    outcome = apply_repairs([_d("D3", "d::0", related=["d::1"])], _unified(S[:2], S[2:4]))
    assert [c.granularity for c in outcome.chunks] == ["coarse", "fine", "fine"]
    assert outcome.chunks[0].text == "A one. B two. C three. D four."
    assert outcome.chunks[0].parent_chunk_id is None
    assert [c.text for c in outcome.chunks[1:]] == ["A one. B two.", "C three. D four."]
    assert all(c.parent_chunk_id == "d::coarse::0" for c in outcome.chunks[1:])


def test_bind_merges_two_distinct_coarse_parents_and_moves_their_children():
    outcome = apply_repairs([_d("D3", "d::fine::0", related=["d::fine::1"])], _dual())
    coarse = [c for c in outcome.chunks if c.granularity == "coarse"]
    assert len(coarse) == 1
    assert coarse[0].text == "A one. B two. C three. D four."
    fine = [c for c in outcome.chunks if c.granularity == "fine"]
    assert [c.text for c in fine] == ["A one. B two.", "C three. D four."]
    assert all(c.parent_chunk_id == coarse[0].chunk_id for c in fine)


def test_bind_records_its_own_provenance_in_metadata():
    outcome = apply_repairs([_d("D3", "d::0", related=["d::1"])], _unified(S[:2], S[2:4]))
    assert outcome.chunks[0].metadata["bound_from"] == "D3"


def test_bind_refuses_a_cross_document_pair():
    chunks = [*_unified(S[:2], doc="d"), *_unified(S[2:4], doc="e")]
    outcome = apply_repairs([_d("D3", "d::0", related=["e::0"])], chunks)
    assert outcome.defects_applied == 0
    assert outcome.defects_deferred == 1


def test_bind_refuses_a_cross_layer_pair():
    """A fine chunk and a coarse chunk are not two halves of a straddle — one
    already covers the other."""
    chunks = _dual()
    outcome = apply_repairs([_d("D3", "d::fine::0", related=["d::coarse::1"])], chunks)
    assert outcome.defects_applied == 0


def test_bind_without_a_related_chunk_is_deferred():
    outcome = apply_repairs([_d("D3", "d::0")], _unified(S[:2], S[2:4]))
    assert outcome.defects_applied == 0
    assert outcome.defects_deferred == 1


def test_bind_refuses_a_pair_that_already_shares_a_cover():
    """D3 suppresses this itself; the repair refuses it too, so a stale or
    hand-written defect cannot rebuild a parent that already exists."""
    chunks = _dual()
    chunks[3] = _chunk(1, S[2:4], granularity="fine", parent="d::coarse::0")
    outcome = apply_repairs([_d("D3", "d::fine::0", related=["d::fine::1"])], chunks)
    assert outcome.defects_applied == 0


# ===========================================================================
# Round discipline
# ===========================================================================
def test_one_action_per_segment_per_round():
    """Conflicting edits on the same segment must not both land — "split X"
    followed by "merge X with Y" would lose text. The loser defers to the next
    round, where it is re-detected against the post-repair segmentation."""
    outcome = apply_repairs([_d("D5", "d::0"), _d("D4", "d::0")], _unified(S[:2], S[2:4]))
    assert outcome.defects_applied == 1
    assert outcome.defects_deferred == 1
    assert outcome.actions == {"split": 1}


def test_the_second_defect_on_a_split_segment_defers_even_via_its_old_id():
    """After a split the original unit is gone, so a second defect naming it
    resolves to a unit that is no longer in the document."""
    outcome = apply_repairs([_d("D5", "d::0"), _d("D1", "d::0")], _unified(S[:4]))
    assert outcome.defects_applied == 1
    assert outcome.defects_deferred == 1


def test_an_action_on_a_fine_child_blocks_an_action_on_its_coarse_parent():
    """A child edit moves the parent's provenance too — splitting a fine chunk
    changes which sentences its coarse cover is the evidence for — so the
    parent's own queued ``extend_left`` must defer to the next round rather than
    absorb a cover that has just been re-cut underneath it."""
    defects = [_d("D2", "d::coarse::1", related=["d::coarse::0"]), _d("D5", "d::fine::1")]
    outcome = apply_repairs(defects, _dual())
    # D5 outranks D2 in REPAIR_PRIORITY, so the child wins the segment.
    assert outcome.actions == {"split": 1}
    assert outcome.defects_deferred == 1


def test_defects_on_independent_segments_all_land_in_one_round():
    defects = [_d("D5", "d::0"), _d("D5", "d::1")]
    outcome = apply_repairs(defects, _unified(S[:2], S[2:4]))
    assert outcome.defects_applied == 2
    assert outcome.actions == {"split": 2}
    assert len(outcome.chunks) == 4


def test_the_most_informed_action_wins_a_contested_segment():
    """``REPAIR_PRIORITY`` puts D3 first because it names an exact sibling,
    while D5 only knows a count. On a contested segment the more informed
    action is the one that should get to move the boundary."""
    assert REPAIR_PRIORITY.index("D3") < REPAIR_PRIORITY.index("D5")
    defects = [_d("D5", "d::fine::0"), _d("D3", "d::fine::0", related=["d::fine::1"])]
    outcome = apply_repairs(defects, _dual())
    assert outcome.actions == {"bind": 1}
    assert outcome.defects_deferred == 1


def test_priority_order_is_independent_of_the_input_order():
    defects = [_d("D3", "d::fine::0", related=["d::fine::1"]), _d("D5", "d::fine::0")]
    assert apply_repairs(defects, _dual()).actions == {"bind": 1}


def test_every_code_has_a_rank_in_the_priority_order():
    """A new detector whose code is missing here would silently sort last and
    lose every contest, which is a behaviour change nobody asked for."""
    from chimera_rag.core.types import DEFECT_REPAIR_ACTION

    assert set(REPAIR_PRIORITY) == set(DEFECT_REPAIR_ACTION)


def test_every_repair_action_has_an_applicator():
    from chimera_rag.core.types import DEFECT_REPAIR_ACTION
    from chimera_rag.plugins.prism.repair import _APPLICATORS

    assert set(_APPLICATORS) == set(DEFECT_REPAIR_ACTION.values())


def test_a_defect_naming_an_unknown_chunk_is_deferred():
    outcome = apply_repairs([_d("D5", "nope::0")], _unified(S[:2]))
    assert outcome.defects_applied == 0
    assert outcome.defects_deferred == 1
    assert len(outcome.chunks) == 1


def test_repairs_are_scoped_per_document():
    """Two documents must not absorb each other's segments even when their ids
    line up."""
    chunks = [*_unified(S[:2], S[2:4], doc="d"), *_unified(S[:2], S[2:4], doc="e")]
    outcome = apply_repairs(
        [_d("D4", "d::0"), _d("D2", "e::1", related=["e::0"])], chunks
    )
    assert outcome.defects_applied == 2
    for doc in ("d", "e"):
        ours = [c for c in outcome.chunks if c.doc_id == doc]
        assert [c.text for c in ours] == ["A one. B two. C three. D four."]


def test_document_order_is_preserved_in_the_output():
    chunks = [*_unified(S[:2], doc="d"), *_unified(S[2:4], doc="e")]
    outcome = apply_repairs([_d("D5", "d::0")], chunks)
    assert [c.doc_id for c in outcome.chunks] == ["d", "d", "e"]


def test_no_defects_returns_the_input_segmentation_reindexed():
    before = _unified(S[:2], S[2:4])
    outcome = apply_repairs([], before)
    assert outcome.defects_applied == 0
    assert outcome.actions == {}
    assert [c.text for c in outcome.chunks] == [c.text for c in before]


def test_repairing_an_empty_chunk_list_is_a_no_op():
    outcome = apply_repairs([], [])
    assert outcome.chunks == []
    assert outcome.defects_applied == 0


# ===========================================================================
# The pipeline loop
# ===========================================================================
class CountingExtractor(BaseTripleExtractor):
    """Records the text of every chunk it is asked to extract.

    Entity names are derived from the chunk's own head *and* tail, so two
    different chunks never share an entity. That keeps D1 and D3 silent in the
    end-to-end test, where the only defect meant to fire is D5 — otherwise a
    fake extractor that emitted constant names would manufacture collisions the
    detectors would rightly flag, and the test would be asserting on noise.
    """

    def __init__(self, triples_per_chunk: int = 1) -> None:
        self.calls: list[str] = []
        self._n = triples_per_chunk

    def extract(self, chunk: Chunk) -> list[Triple]:
        self.calls.append(chunk.text)
        marker = f"{chunk.text[:12]}|{chunk.text[-12:]}"
        return [
            Triple(
                subject=f"{marker}#s{i}",
                predicate="mentions",
                object=f"{marker}#o{i}",
                source_chunk_id=chunk.chunk_id,
            )
            for i in range(self._n)
        ]


class RecordingVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.added_ids: list[str] = []

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        self.added_ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        return []

    def persist(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


class RecordingGraphStore(BaseGraphStore):
    def __init__(self) -> None:
        self.triples: list[Triple] = []

    def add_triple(self, triple: Triple) -> None:
        self.triples.append(triple)

    def get_neighbors(self, entity: str, hops: int = 1) -> list[Triple]:
        return []

    def all_triples(self) -> list[Triple]:
        return list(self.triples)

    def persist(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


class FakeEmbedder:
    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.ones((len(texts), self.dim), dtype=np.float32)


class ScriptedChunker(BaseChunker):
    """No audit capability — the legacy single-pass path."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def chunk(self, document: Document) -> list[Chunk]:
        return list(self._chunks)


class ScriptedAuditor(BaseChunker, BaseChunkAuditor):
    """A chunker double whose audit findings and repairs the test scripts."""

    def __init__(
        self,
        chunks: list[Chunk],
        *,
        findings: list[list[ChunkDefect]] | None = None,
        repair_fn=None,
    ) -> None:
        self._chunks = chunks
        self._findings = list(findings or [])
        self._repair_fn = repair_fn
        self.audit_calls: list[list[Chunk]] = []
        self.repair_calls: list[list[ChunkDefect]] = []

    def chunk(self, document: Document) -> list[Chunk]:
        return list(self._chunks)

    def audit(self, chunks, triples_by_chunk, graph=None) -> list[ChunkDefect]:
        self.audit_calls.append(list(chunks))
        return self._findings.pop(0) if self._findings else []

    def repair(self, defects, chunks) -> RepairOutcome:
        self.repair_calls.append(list(defects))
        if self._repair_fn is not None:
            return self._repair_fn(defects, chunks)
        return RepairOutcome(chunks=list(chunks), defects_deferred=len(defects))


def _run(
    chunker: BaseChunker,
    *,
    audit_rounds: int = 2,
    extractor: CountingExtractor | None = None,
    content: str = "scripted; chunks come from the double",
):
    extractor = extractor or CountingExtractor()
    graph = RecordingGraphStore()
    vectors = RecordingVectorStore()
    pipe = IngestionPipeline(
        chunker=chunker,
        extractor=extractor,
        pruner=NoOpPruner(),
        embedder=FakeEmbedder(),
        state=PipelineState(graph_store=graph, vector_store=vectors),
        show_progress=False,
        audit_rounds=audit_rounds,
    )
    stats = pipe.run([Document(doc_id="d", content=content)])
    return pipe, stats, extractor, graph, vectors


def _split_repair(defects, chunks) -> RepairOutcome:
    """The real repair engine, used so the pipeline tests exercise composition
    rather than a mock."""
    return apply_repairs(defects, chunks)


def test_a_chunker_without_the_audit_capability_never_loops():
    _, stats, extractor, _, _ = _run(ScriptedChunker(_unified(S[:2], S[2:4])))
    assert stats["audit_rounds"] == 0
    assert stats["repairs_applied"] == 0
    assert stats["defects"] == 0
    assert len(extractor.calls) == 2


def test_audit_rounds_zero_disables_the_loop():
    """E2 ablation row 4: dual granularity with the closed loop switched off."""
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4]), findings=[[_d("D5", "d::0")]], repair_fn=_split_repair
    )
    _, stats, extractor, _, _ = _run(chunker, audit_rounds=0)
    assert chunker.audit_calls == []
    assert chunker.repair_calls == []
    assert stats["audit_rounds"] == 0
    assert len(extractor.calls) == 2


def test_audit_disabled_on_a_real_prism_chunker_still_ingests():
    from chimera_rag.plugins.prism.chunker import PrismChunker

    chunker = PrismChunker(audit_enabled=False)
    _, stats, _, _, _ = _run(chunker)
    assert stats["defects"] == 0
    assert stats["repairs_applied"] == 0


def test_only_the_recut_segments_are_extracted_again():
    """Risk R2: a repair round must cost the marginal segments, not the corpus.
    The cache is keyed by *text* because a repair renumbers ids, so id-keyed
    reuse could not tell a re-cut segment from an untouched one."""
    extractor = CountingExtractor()
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4], S[4:6]),
        findings=[[_d("D5", "d::0")]],
        repair_fn=_split_repair,
    )
    _, stats, _, _, _ = _run(chunker, extractor=extractor)
    assert stats["audit_rounds"] == 1
    # Round 0 extracts all three; round 1 only the two halves of d::0.
    assert extractor.calls == [
        "A one. B two.",
        "C three. D four.",
        "E five. F six.",
        "A one.",
        "B two.",
    ]


def test_an_untouched_chunk_is_not_re_extracted_across_several_rounds():
    extractor = CountingExtractor()
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4]),
        findings=[[_d("D5", "d::0")], [_d("D5", "d::2")]],
        repair_fn=_split_repair,
    )
    _, stats, _, _, _ = _run(chunker, extractor=extractor)
    assert stats["audit_rounds"] == 2
    assert extractor.calls.count("C three. D four.") == 1
    # Every extraction is of a text nobody has seen before — the R2 claim.
    assert len(set(extractor.calls)) == len(extractor.calls)


def test_low_confidence_defects_are_recorded_but_never_drive_a_repair():
    """Risk R3, made enforceable: the loop must not even call ``repair`` when
    every finding is record-only, because a detector that could not falsify its
    own candidate is the one whose noise the loop would amplify."""
    low = _d("D1", "d::0", confidence="low")
    assert low.repair_driving is False
    chunker = ScriptedAuditor(_unified(S[:2], S[2:4]), findings=[[low]], repair_fn=_split_repair)
    _, stats, _, _, _ = _run(chunker)
    assert chunker.repair_calls == []
    assert stats["defects"] == 1
    assert stats["repairs_applied"] == 0
    assert stats["audit_rounds"] == 0


def test_only_repair_driving_defects_reach_the_auditor():
    low = _d("D1", "d::0", confidence="low")
    high = _d("D5", "d::1")
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4]), findings=[[low, high]], repair_fn=_split_repair
    )
    _run(chunker)
    assert [d.chunk_id for d in chunker.repair_calls[0]] == ["d::1"]


def test_a_defect_without_a_confidence_is_repair_driving():
    """``confidence`` is opt-in per detector; D5 and D3 set none, and treating
    an absent value as low would switch the whole loop off by default."""
    assert _d("D5", "d::0").repair_driving is True


def test_the_loop_stops_when_a_round_applies_nothing():
    """A repair that cannot move any boundary (a split on a single sentence)
    returns ``defects_applied == 0``; looping again would repeat it forever."""
    chunker = ScriptedAuditor(
        _unified(S[:1], S[1:2]),
        findings=[[_d("D5", "d::0")], [_d("D5", "d::0")], [_d("D5", "d::0")]],
        repair_fn=_split_repair,
    )
    _, stats, _, _, _ = _run(chunker, audit_rounds=5)
    assert stats["audit_rounds"] == 0
    assert stats["defects"] == 1
    assert len(chunker.audit_calls) == 1


def test_the_loop_stops_when_a_round_finds_nothing():
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4]), findings=[[_d("D5", "d::0")], []], repair_fn=_split_repair
    )
    _, stats, _, _, _ = _run(chunker, audit_rounds=5)
    assert stats["audit_rounds"] == 1
    assert len(chunker.audit_calls) == 2


def _split_the_longest(defects, chunks) -> RepairOutcome:
    """Always re-cuts the longest segment, so the loop can only be stopped by
    ``audit_rounds`` — which is exactly what the k_max test needs."""
    target = max(chunks, key=lambda c: len(split_sentences(c.text)))
    return apply_repairs([_d("D5", target.chunk_id)], chunks)


def test_the_loop_stops_at_k_max_even_while_defects_remain():
    """k_max is the cost ceiling. Every round here still finds a defect worth
    acting on, so nothing but ``audit_rounds`` ends the loop."""
    findings = [[_d("D5", "d::0")] for _ in range(6)]
    chunker = ScriptedAuditor(_unified(S[:6]), findings=findings, repair_fn=_split_the_longest)
    _, stats, extractor, _, _ = _run(chunker, audit_rounds=3)
    assert stats["audit_rounds"] == 3
    assert stats["repairs_applied"] == 3
    assert len(chunker.audit_calls) == 3
    # Four extraction passes for three repair rounds, and no text paid for twice.
    assert len(set(extractor.calls)) == len(extractor.calls)


def test_negative_audit_rounds_are_clamped_to_zero():
    assert IngestionPipeline(
        chunker=ScriptedChunker([]),
        extractor=CountingExtractor(),
        pruner=NoOpPruner(),
        embedder=FakeEmbedder(),
        state=PipelineState(graph_store=RecordingGraphStore(), vector_store=RecordingVectorStore()),
        audit_rounds=-3,
    ).audit_rounds == 0


def test_stats_report_the_defect_census_by_code():
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4], S[4:6]),
        findings=[[_d("D5", "d::0"), _d("D5", "d::1"), _d("D1", "d::2", confidence="low")]],
        repair_fn=_split_repair,
    )
    _, stats, _, _, _ = _run(chunker)
    assert stats["defects"] == 3
    assert stats["defects_by_code"] == {"D5": 2, "D1": 1}
    assert stats["repairs_applied"] == 2
    assert stats["chunks"] == 5


def test_defects_from_every_round_are_counted_not_just_the_last():
    chunker = ScriptedAuditor(
        _unified(S[:4]),
        findings=[[_d("D5", "d::0")], [_d("D5", "d::0")]],
        repair_fn=_split_repair,
    )
    _, stats, _, _, _ = _run(chunker)
    assert stats["audit_rounds"] == 2
    assert stats["defects"] == 2


# ---------------------------------------------------------------------------
# Post-repair state must be the only state anybody reads
# ---------------------------------------------------------------------------
def test_chunk_lookup_holds_only_post_repair_ids():
    """A repair renumbers and can *remove* ids. Populating the lookup before the
    loop settles would leave an entry for a chunk that no longer exists, and a
    retrieval hit on it would resolve to stale text."""
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4]),
        findings=[[_d("D2", "d::1", related=["d::0"])]],
        repair_fn=_split_repair,
    )
    pipe, _, _, _, _ = _run(chunker)
    assert set(pipe.state.chunk_lookup) == {"d::0"}
    assert pipe.state.chunk_lookup["d::0"].text == "A one. B two. C three. D four."


def test_the_vector_store_receives_post_repair_ids():
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4]),
        findings=[[_d("D5", "d::0")]],
        repair_fn=_split_repair,
    )
    _, _, _, _, vectors = _run(chunker)
    assert vectors.added_ids == ["d::0", "d::1", "d::2"]


def test_only_indexed_granularities_reach_the_vector_store_after_a_bind():
    """A D3 bind converts ``unified`` chunks into a ``coarse`` cover plus two
    ``fine`` children, so the post-repair index holds one id, not three."""
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4]),
        findings=[[_d("D3", "d::0", related=["d::1"])]],
        repair_fn=_split_repair,
    )
    _, stats, _, _, vectors = _run(chunker)
    assert vectors.added_ids == ["d::coarse::0"]
    assert stats["chunks"] == 3


def test_every_committed_triple_cites_a_chunk_that_still_exists():
    """Provenance integrity across the loop: round 0 extracted ``d::1``, then a
    repair absorbed it. Its triples must not survive into the graph under an id
    the lookup no longer knows."""
    chunker = ScriptedAuditor(
        _unified(S[:2], S[2:4]),
        findings=[[_d("D2", "d::1", related=["d::0"])]],
        repair_fn=_split_repair,
    )
    pipe, _, _, graph, _ = _run(chunker)
    assert graph.triples
    for t in graph.triples:
        assert t.source_chunk_id in pipe.state.chunk_lookup


def test_a_cached_triple_is_restamped_to_the_chunk_it_is_handed_to():
    """When two chunks share a text — routine once overlap is involved — the
    second one reuses the first's triples. Without re-stamping, the graph would
    cite a chunk the retrieval side may never surface."""
    chunks = _unified(S[:2], S[:2], S[2:4])
    extractor = CountingExtractor()
    pipe, _, _, graph, _ = _run(ScriptedChunker(chunks), extractor=extractor)
    assert len(extractor.calls) == 3
    cited = sorted({t.source_chunk_id for t in graph.triples})
    assert cited == ["d::0", "d::1", "d::2"]
    assert set(cited) <= set(pipe.state.chunk_lookup)


# ===========================================================================
# End-to-end with the real PrismRAG chunker
# ===========================================================================
def test_a_real_prism_chunker_runs_detect_and_repair_end_to_end():
    """Composition of all three contributions: the dual-granularity chunker
    emits fine/coarse, an over-budget extractor trips D5 on the fine layer, and
    the loop re-cuts the fine boundaries while the coarse retrieval layer stays
    put. Nothing here is scripted — the detector decides."""
    from chimera_rag.plugins.prism.chunker import PrismChunker

    text = " ".join(f"Sentence number {i} about topic {i}." for i in range(12))
    chunker = PrismChunker(
        fine_min_size=120,
        fine_max_size=120,
        coarse_min_size=300,
        coarse_max_size=300,
        triple_budget=2,
    )
    extractor = CountingExtractor(triples_per_chunk=5)
    initial = chunker.chunk(Document(doc_id="d", content=text))
    initial_fine = len([c for c in initial if c.granularity == "fine"])
    initial_coarse = len([c for c in initial if c.granularity == "coarse"])

    pipe, stats, _, _, vectors = _run(chunker, extractor=extractor, audit_rounds=2, content=text)

    assert stats["defects"] > 0
    assert stats["defects_by_code"].get("D5", 0) > 0
    assert stats["repairs_applied"] > 0
    assert stats["audit_rounds"] == 2

    fine = [c for c in pipe.state.chunk_lookup.values() if c.granularity == "fine"]
    coarse = [c for c in pipe.state.chunk_lookup.values() if c.granularity == "coarse"]
    assert fine and coarse
    # The extraction layer was re-cut; the retrieval layer was not.
    assert len(fine) > initial_fine
    assert len(coarse) == initial_coarse
    assert all(c.chunk_id in vectors.added_ids for c in coarse)
    assert not any(c.chunk_id in vectors.added_ids for c in fine)
    # Every fine chunk still resolves to a coarse parent that exists.
    coarse_ids = {c.chunk_id for c in coarse}
    assert all(c.parent_chunk_id in coarse_ids for c in fine)
    # And the coarse layer still covers the whole document.
    doc = " ".join(split_sentences(text))
    assert " ".join(c.text for c in sorted(coarse, key=lambda c: c.index)) == doc
    # So does the re-cut fine layer — a repair moved boundaries, it did not
    # change what the extractor is asked to read.
    assert " ".join(c.text for c in sorted(fine, key=lambda c: c.index)) == doc
