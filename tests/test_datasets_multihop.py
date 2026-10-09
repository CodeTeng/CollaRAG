"""Tests for the HotpotQA / 2Wiki / Musique dataset loaders.

These three are the canonical multi-hop QA benchmarks for the thesis;
the loaders only need to read their native JSON/JSONL shape and map to
:class:`QAExample` / :class:`Document`.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------
HOTPOT_SAMPLE = [
    {
        "_id": "h1",
        "question": "Who invented dynamite?",
        "answer": "Alfred Nobel",
        "context": [
            ["Alfred Nobel", ["Alfred Nobel was a Swedish chemist.",
                              "He invented dynamite in 1867."]],
            ["Nitroglycerin", ["Nitroglycerin is a chemical compound."]],
        ],
        "type": "bridge",
        "level": "medium",
    },
    {
        "_id": "h2",
        "question": "Who formulated relativity?",
        "answer": "Albert Einstein",
        "context": [
            ["Albert Einstein", ["Albert Einstein was a physicist."]],
        ],
    },
]


def test_hotpotqa_loader_reads_examples(tmp_path: Path):
    from chimera_rag.datasets.hotpotqa import HotpotQALoader

    p = tmp_path / "hotpot.json"
    p.write_text(json.dumps(HOTPOT_SAMPLE))

    examples = HotpotQALoader(path=str(p)).load_examples()
    assert len(examples) == 2
    assert examples[0].qid == "h1"
    assert examples[0].question.startswith("Who invented")
    assert examples[0].answer == "Alfred Nobel"
    # contexts should be flattened sentence lists
    assert any("dynamite" in c.lower() for c in examples[0].contexts)


def test_hotpotqa_loader_load_documents_emits_one_doc_per_title(tmp_path: Path):
    from chimera_rag.datasets.hotpotqa import HotpotQALoader

    p = tmp_path / "hotpot.json"
    p.write_text(json.dumps(HOTPOT_SAMPLE))

    docs = HotpotQALoader(path=str(p)).load_documents()
    # Titles seen: "Alfred Nobel", "Nitroglycerin", "Albert Einstein" (3 unique)
    doc_ids = {d.doc_id for d in docs}
    assert "Alfred Nobel" in doc_ids
    assert "Nitroglycerin" in doc_ids
    assert "Albert Einstein" in doc_ids


def test_hotpotqa_loader_respects_limit(tmp_path: Path):
    from chimera_rag.datasets.hotpotqa import HotpotQALoader

    p = tmp_path / "hotpot.json"
    p.write_text(json.dumps(HOTPOT_SAMPLE))

    assert len(HotpotQALoader(path=str(p)).load_examples(limit=1)) == 1


# ---------------------------------------------------------------------------
# 2Wiki
# ---------------------------------------------------------------------------
TWOWIKI_SAMPLE = [
    {
        "_id": "t1",
        "question": "Where was Alfred Nobel born?",
        "answer": "Stockholm",
        "context": [
            ["Alfred Nobel", ["Alfred Nobel was born in Stockholm in 1833."]],
        ],
    }
]


def test_two_wiki_loader_reads_examples(tmp_path: Path):
    from chimera_rag.datasets.two_wiki import TwoWikiLoader

    p = tmp_path / "2wiki.json"
    p.write_text(json.dumps(TWOWIKI_SAMPLE))

    examples = TwoWikiLoader(path=str(p)).load_examples()
    assert len(examples) == 1
    assert examples[0].qid == "t1"
    assert examples[0].answer == "Stockholm"


# ---------------------------------------------------------------------------
# Musique
# ---------------------------------------------------------------------------
MUSIQUE_SAMPLE = [
    {"id": "m1", "question": "What element did Curie discover?", "answer": "radium"},
    {"id": "m2", "question": "Who founded the Nobel Prize?", "answer": "Alfred Nobel"},
]


def test_musique_loader_reads_jsonl(tmp_path: Path):
    from chimera_rag.datasets.musique import MusiqueLoader

    p = tmp_path / "musique.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in MUSIQUE_SAMPLE))

    examples = MusiqueLoader(path=str(p)).load_examples()
    assert len(examples) == 2
    assert examples[0].qid == "m1"


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------
def test_registered_loaders_cover_all_three_multi_hop_benchmarks():
    from chimera_rag.datasets import get_dataset_loader

    assert get_dataset_loader("hotpotqa").__name__ == "HotpotQALoader"
    assert get_dataset_loader("two_wiki").__name__ == "TwoWikiLoader"
    assert get_dataset_loader("musique").__name__ == "MusiqueLoader"
