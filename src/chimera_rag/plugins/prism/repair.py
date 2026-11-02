"""Boundary-change repairs for typed chunk defects — PrismRAG contribution 3.

Every action in this module **re-cuts a segmentation boundary**. None of them
calls the extractor. That distinction is the whole contribution: risk R1 in
``docs/innovation-1/design.md`` warns that if "repair" meant extracting again
and unioning the results, PrismRAG would collapse into the
``extract-twice-union`` counter-baseline and prove nothing.

==========  ==========================================  ======================
Action      Boundary change                             Fixes
==========  ==========================================  ======================
``split``   one segment becomes two at a sentence cut    D1, D5
``split_by_topic``  as above, cut placed between topics  D6
``extend_left``     segment absorbs its predecessor      D2
``merge_up``        segment absorbs its successor        D4
``bind``      two segments gain a shared coarse parent   D3
==========  ==========================================  ======================

Two rules keep a round from corrupting itself:

* **One action per segment per round.** Defects are applied in
  :data:`REPAIR_PRIORITY` order and any segment already edited — or whose
  parent was — defers to the next round. Without this, "split X" and "merge X
  with Y" could both land and lose text.
* **Text is conserved.** Merging concatenates, splitting partitions, binding
  only adds a cover. No repair ever drops a sentence, which
  ``tests/test_prism_repair_loop.py`` asserts directly.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from chimera_rag.core.types import Chunk, ChunkDefect, ChunkGranularity, RepairOutcome
from chimera_rag.plugins.prism.features import split_sentences

#: Applied most-specific first. D3 names an exact sibling and D6 an exact
#: conflicting pair, so they carry more information than D5 (a count) or D2/D4
#: (a missing antecedent). The most informed action wins the right to touch a
#: segment in a given round; the rest defer.
REPAIR_PRIORITY: tuple[str, ...] = ("D3", "D6", "D1", "D5", "D2", "D4")

#: Emission order within a document. Coarse ids must be assigned before fine
#: ones because a fine chunk stores its parent's id.
_EMIT_ORDER: tuple[ChunkGranularity, ...] = ("coarse", "fine", "unified")


@dataclass(eq=False)
class _Unit:
    """One mutable segment of one document during a repair round.

    ``eq=False`` is load-bearing: repairs locate units with ``list.index`` and
    ``list.remove``, which must compare by identity. Two units holding the same
    text are still two different segments.
    """

    doc_id: str
    index: int
    sentences: list[str]
    granularity: ChunkGranularity
    parent: _Unit | None = None
    origin_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(self.sentences).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def apply_repairs(
    defects: Sequence[ChunkDefect], chunks: Sequence[Chunk]
) -> RepairOutcome:
    """Re-segment *chunks* to fix *defects*.

    Returns the full new chunk list — repairs renumber a layer, so ids are
    re-derived from position rather than patched in place. The caller is
    expected to have filtered out low-confidence defects already (risk R3).
    """
    by_doc, doc_order = _to_units(chunks)
    by_origin = {u.origin_id: u for units in by_doc.values() for u in units}

    applied = 0
    deferred = 0
    actions: dict[str, int] = defaultdict(int)
    touched: set[int] = set()

    for defect in _in_priority_order(defects):
        target = by_origin.get(defect.chunk_id)
        if target is None:
            deferred += 1
            continue
        siblings = [by_origin[r] for r in defect.related_chunk_ids if r in by_origin]
        units = by_doc[target.doc_id]
        if target not in units:
            # An earlier action in this round already absorbed or replaced it.
            deferred += 1
            continue

        # A unit already edited this round — or one whose coarse cover was —
        # defers, so conflicting edits cannot both land.
        implicated = [target, *siblings, *(u.parent for u in (target, *siblings) if u.parent)]
        if any(id(u) in touched for u in implicated):
            deferred += 1
            continue

        applicator = _APPLICATORS.get(defect.action)
        if applicator is None or not applicator(defect, target, siblings, units):
            deferred += 1
            continue

        touched.update(id(u) for u in implicated)
        applied += 1
        actions[defect.action] += 1

    return RepairOutcome(
        chunks=_emit(by_doc, doc_order),
        defects_applied=applied,
        defects_deferred=deferred,
        actions=dict(actions),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
def _split(defect: ChunkDefect, target: _Unit, siblings: list[_Unit], units: list[_Unit]) -> bool:
    """Halve *target* at a sentence boundary (D1, D5)."""
    if len(target.sentences) < 2:
        return False
    # D1 knows which neighbourhoods collided, so try to cut between them before
    # falling back to the midpoint.
    groups = [list(g) for g in defect.metadata.get("component_members", [])]
    cut = _separating_cut(target.sentences, groups) if groups else None
    return _split_at(units, target, cut if cut is not None else len(target.sentences) // 2)


def _split_by_topic(
    defect: ChunkDefect, target: _Unit, siblings: list[_Unit], units: list[_Unit]
) -> bool:
    """Cut between the two conflicting values (D6)."""
    if len(target.sentences) < 2:
        return False
    objects = list(defect.metadata.get("objects", []))
    cut = _separating_cut(target.sentences, [[o] for o in objects])
    return _split_at(units, target, cut if cut is not None else len(target.sentences) // 2)


def _extend_left(
    defect: ChunkDefect, target: _Unit, siblings: list[_Unit], units: list[_Unit]
) -> bool:
    """Absorb the predecessor so the opening anaphor resolves (D2)."""
    prev = siblings[0] if siblings else _sibling(units, target, target.index - 1)
    if prev is None or prev.doc_id != target.doc_id or prev not in units:
        return False
    _absorb(units, keeper=prev, absorbed=target)
    return True


def _merge_up(
    defect: ChunkDefect, target: _Unit, siblings: list[_Unit], units: list[_Unit]
) -> bool:
    """Absorb the successor so a starved chunk gains context (D4).

    Falls back to the predecessor at the end of a layer — the point is to stop
    extracting this span alone, and either direction does that.
    """
    nxt = _sibling(units, target, target.index + 1)
    if nxt is not None:
        _absorb(units, keeper=target, absorbed=nxt)
        return True
    prev = _sibling(units, target, target.index - 1)
    if prev is None:
        return False
    _absorb(units, keeper=prev, absorbed=target)
    return True


def _bind(
    defect: ChunkDefect, target: _Unit, siblings: list[_Unit], units: list[_Unit]
) -> bool:
    """Give two segments a shared coarse parent (D3).

    Three shapes collapse into one algorithm:

    * both sides already have distinct coarse parents -> the parents merge and
      every fine child follows;
    * a side has no parent -> that side becomes a ``fine`` child of the new
      coarse cover, which is where a ``unified`` chunk first acquires the
      dual-granularity structure;
    * both sides are parentless -> two ``unified`` chunks become two ``fine``
      children of one new ``coarse`` chunk. Index coverage is unchanged: the
      coarse chunk is what the vector store sees.
    """
    if not siblings:
        return False
    other = siblings[0]
    if other.doc_id != target.doc_id or other.granularity != target.granularity:
        return False

    first, second = (
        (target, other) if target.index <= other.index else (other, target)
    )
    cover_a = first.parent if first.parent is not None else first
    cover_b = second.parent if second.parent is not None else second
    if cover_a is cover_b:
        return False

    coarse = _Unit(
        doc_id=target.doc_id,
        index=min(first.index, second.index),
        sentences=[*cover_a.sentences, *cover_b.sentences],
        granularity="coarse",
        parent=None,
        origin_id="",
        metadata={**cover_a.metadata, "layer": "coarse", "bound_from": "D3"},
    )

    # Drop the old covers, but never a side that had no parent of its own —
    # that side stays put and becomes a fine child below.
    for old in (cover_a, cover_b):
        if old is not target and old is not other and old in units:
            units.remove(old)
    units.append(coarse)

    for u in units:
        if u is not coarse and (u.parent is cover_a or u.parent is cover_b):
            u.parent = coarse
    for side in (target, other):
        if side in units and side.parent is None:
            side.granularity = "fine"
            side.parent = coarse
    return True


_APPLICATORS = {
    "split": _split,
    "split_by_topic": _split_by_topic,
    "extend_left": _extend_left,
    "merge_up": _merge_up,
    "bind": _bind,
}


# ---------------------------------------------------------------------------
# Segment surgery (module-private)
# ---------------------------------------------------------------------------
def _split_at(units: list[_Unit], target: _Unit, cut: int) -> bool:
    if not 1 <= cut < len(target.sentences):
        return False
    left = _Unit(
        doc_id=target.doc_id,
        index=target.index,
        sentences=target.sentences[:cut],
        granularity=target.granularity,
        parent=target.parent,
        origin_id=target.origin_id,
        metadata=dict(target.metadata),
    )
    right = _Unit(
        doc_id=target.doc_id,
        index=target.index,
        sentences=target.sentences[cut:],
        granularity=target.granularity,
        parent=target.parent,
        origin_id="",
        metadata=dict(target.metadata),
    )
    units[units.index(target) : units.index(target) + 1] = [left, right]
    return True


def _absorb(units: list[_Unit], *, keeper: _Unit, absorbed: _Unit) -> None:
    """Concatenate *absorbed* into *keeper*, keeping the earlier position.

    Also reparents *absorbed*'s children: D2 audits the coarse layer, so
    ``extend_left`` can remove a coarse cover that fine chunks hang from. Left
    dangling, those children would be emitted with ``parent_chunk_id=None`` and
    contribution 1's provenance chain would break silently — the retrieval hit
    on the merged coarse chunk would no longer resolve to them.
    """
    first, second = (
        (keeper, absorbed) if keeper.index <= absorbed.index else (absorbed, keeper)
    )
    keeper.sentences = [*first.sentences, *second.sentences]
    keeper.index = min(keeper.index, absorbed.index)
    keeper.origin_id = keeper.origin_id or absorbed.origin_id
    units.remove(absorbed)
    for u in units:
        if u.parent is absorbed:
            u.parent = keeper


def _sibling(units: Sequence[_Unit], target: _Unit, index: int) -> _Unit | None:
    """The same-layer unit at *index*, or ``None``.

    Layer-scoped because fine and coarse each number from 0 within a document.
    """
    for u in units:
        if u.granularity == target.granularity and u.index == index:
            return u
    return None


def _separating_cut(sentences: Sequence[str], groups: Sequence[Sequence[str]]) -> int | None:
    """Earliest sentence index that confines every group's mentions to one side.

    Returns ``None`` when fewer than two groups actually appear in the text, or
    when no cut separates them — the caller then falls back to the midpoint.
    """
    lowered = [s.lower() for s in sentences]
    placements: list[set[int]] = []
    for group in groups:
        terms = [t.lower() for t in group if t]
        hits = {i for i, s in enumerate(lowered) if any(t in s for t in terms)}
        if hits:
            placements.append(hits)
    if len(placements) < 2:
        return None
    for k in range(1, len(sentences)):
        if all(max(p) < k or min(p) >= k for p in placements):
            return k
    return None


def _in_priority_order(defects: Sequence[ChunkDefect]) -> list[ChunkDefect]:
    rank = {code: i for i, code in enumerate(REPAIR_PRIORITY)}
    fallback = len(REPAIR_PRIORITY)
    return sorted(defects, key=lambda d: (rank.get(d.code, fallback), d.chunk_id))


# ---------------------------------------------------------------------------
# Unit <-> Chunk conversion
# ---------------------------------------------------------------------------
def _to_units(chunks: Sequence[Chunk]) -> tuple[dict[str, list[_Unit]], list[str]]:
    doc_order: list[str] = []
    by_doc: dict[str, list[_Unit]] = {}
    pairs: list[tuple[Chunk, _Unit]] = []

    for ch in chunks:
        if ch.doc_id not in by_doc:
            by_doc[ch.doc_id] = []
            doc_order.append(ch.doc_id)
        unit = _Unit(
            doc_id=ch.doc_id,
            index=ch.index,
            sentences=split_sentences(ch.text),
            granularity=ch.granularity,
            origin_id=ch.chunk_id,
            metadata=dict(ch.metadata),
        )
        by_doc[ch.doc_id].append(unit)
        pairs.append((ch, unit))

    # Second pass: parent_chunk_id is a string, so it can only be resolved once
    # every unit in the document exists.
    for doc_id, units in by_doc.items():
        by_origin = {u.origin_id: u for u in units}
        for ch, unit in pairs:
            if ch.doc_id == doc_id and ch.parent_chunk_id is not None:
                unit.parent = by_origin.get(ch.parent_chunk_id)

    return by_doc, doc_order


def _emit(by_doc: dict[str, list[_Unit]], doc_order: Sequence[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id in doc_order:
        units = by_doc[doc_id]
        layers = {
            layer: sorted((u for u in units if u.granularity == layer), key=lambda u: u.index)
            for layer in _EMIT_ORDER
        }

        ids: dict[int, str] = {}
        for layer in _EMIT_ORDER:
            for i, unit in enumerate(layers[layer]):
                ids[id(unit)] = (
                    f"{doc_id}::{i}" if layer == "unified" else f"{doc_id}::{layer}::{i}"
                )

        for layer in _EMIT_ORDER:
            for i, unit in enumerate(layers[layer]):
                parent_id = ids.get(id(unit.parent)) if unit.parent is not None else None
                chunks.append(
                    Chunk(
                        chunk_id=ids[id(unit)],
                        doc_id=doc_id,
                        index=i,
                        text=unit.text,
                        granularity=layer,
                        parent_chunk_id=parent_id,
                        metadata={**unit.metadata, "layer": layer},
                    )
                )
    return chunks


__all__ = ["REPAIR_PRIORITY", "apply_repairs"]
