"""Generate 100 synthetic wiki-style multi-hop QA items.

Output layout (deterministic, seeded):

data/mock_wiki/
├── corpus.txt   — one "paragraph" per blank-line-separated block, used
│                   by PlainTextLoader for ingestion
└── qa.jsonl     — one JSON object per line, {qid, question, answer}
                   used by GenericJsonlLoader for evaluation

The facts form a small closed-world knowledge graph so each QA pair can
be unambiguously answered from the corpus. Multi-hop questions
(``linked_by``) force the retriever/reasoning path to bridge two
entities via an intermediate node.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "data" / "mock_wiki"

# ---------------------------------------------------------------------------
# Small closed-world KG
# ---------------------------------------------------------------------------
PEOPLE = [
    ("Ada Verdant",    "Lumentech Corp",   "London",       "machine learning"),
    ("Bruno Falk",     "Helios Systems",   "Berlin",       "quantum computing"),
    ("Cara Solis",     "Nimbus Labs",      "Barcelona",    "cryptography"),
    ("Dmitri Varga",   "Obsidian AI",      "Prague",       "robotics"),
    ("Elena Moss",     "Polaris Networks", "Stockholm",    "distributed systems"),
    ("Farid Azar",     "Quillfire Corp",   "Cairo",        "natural language processing"),
    ("Greta Halden",   "Riverbend Studios","Oslo",         "computer graphics"),
    ("Hiro Tanaka",    "Sendai Robotics",  "Osaka",        "reinforcement learning"),
    ("Iris Orlova",    "Tundra Dynamics",  "Moscow",       "optimization"),
    ("Jonah Reed",     "Vanta Analytics",  "Dublin",       "time-series forecasting"),
    ("Kira Lang",      "Winterfell Labs",  "Reykjavik",    "graph neural networks"),
    ("Lionel Costa",   "Xanadu Media",     "Lisbon",       "recommendation systems"),
    ("Mira Shahin",    "Yggdrasil AI",     "Istanbul",     "speech synthesis"),
    ("Noa Kessler",    "Zenith Robotics",  "Tel Aviv",     "computer vision"),
    ("Omar Diallo",    "Aether Graph",     "Dakar",        "knowledge graphs"),
]

# Edges: companies -> their acquiring parents (some are independent).
# This creates a second hop: person -> company -> parent.
ACQUISITIONS = {
    "Lumentech Corp":   "Helios Systems",
    "Nimbus Labs":      "Obsidian AI",
    "Polaris Networks": "Obsidian AI",
    "Quillfire Corp":   "Riverbend Studios",
    "Sendai Robotics":  "Tundra Dynamics",
    "Vanta Analytics":  "Winterfell Labs",
    "Xanadu Media":     "Yggdrasil AI",
    "Zenith Robotics":  "Aether Graph",
    # The rest (Helios, Obsidian, Riverbend, Tundra, Winterfell,
    # Yggdrasil, Aether) remain independent.
}

# Each company was founded in a specific year.
FOUNDATION_YEARS = {c: 1995 + (i * 2) for i, (_, c, _, _) in enumerate(PEOPLE)}


# ---------------------------------------------------------------------------
# Corpus generation (one paragraph per entity, wiki-style)
# ---------------------------------------------------------------------------
def build_corpus() -> str:
    paragraphs: list[str] = []

    for person, company, city, field in PEOPLE:
        year = FOUNDATION_YEARS[company]
        paragraphs.append(
            f"{person} is a researcher based in {city}. "
            f"{person} works at {company}, where they lead research on {field}. "
            f"{person} earned a PhD before joining {company}."
        )
        parent = ACQUISITIONS.get(company)
        if parent:
            paragraphs.append(
                f"{company} is a technology company headquartered in {city}. "
                f"{company} was founded in {year}. "
                f"In 2021, {company} was acquired by {parent}."
            )
        else:
            paragraphs.append(
                f"{company} is an independent technology company headquartered in {city}. "
                f"{company} was founded in {year} and remains independently owned."
            )

    # Add parent-company profiles (only include once).
    parents = sorted(set(ACQUISITIONS.values()))
    for parent in parents:
        # Where is the parent? use the city of its own founder if it appears in PEOPLE.
        parent_city = next((c for _, co, c, _ in PEOPLE if co == parent), "Zurich")
        paragraphs.append(
            f"{parent} is a global technology group with headquarters in {parent_city}. "
            f"{parent} operates multiple subsidiaries across Europe and Asia."
        )

    return "\n\n".join(paragraphs) + "\n"


# ---------------------------------------------------------------------------
# QA generation — 100 items with varied intents
# ---------------------------------------------------------------------------
def build_qa() -> list[dict]:
    qa: list[dict] = []

    # --- factual, 1-hop (30) ---
    for person, company, city, field in PEOPLE[:10]:
        qa.append({"question": f"Where is {person} based?",               "answer": city})
        qa.append({"question": f"Which company does {person} work at?",   "answer": company})
        qa.append({"question": f"What field does {person} research?",     "answer": field})

    # --- factual about companies (15) ---
    for _person, company, _city, _field in PEOPLE[:15]:
        year = FOUNDATION_YEARS[company]
        qa.append({"question": f"In what year was {company} founded?",    "answer": str(year)})

    # --- multi-hop: person -> company -> parent (15) ---
    mh_count = 0
    for person, company, _city, _field in PEOPLE:
        parent = ACQUISITIONS.get(company)
        if parent and mh_count < 15:
            qa.append({
                "question": f"Which company acquired the employer of {person}?",
                "answer": parent,
            })
            mh_count += 1

    # --- multi-hop: person -> company -> city (10) ---
    mh_city_count = 0
    for person, _company, city, _field in PEOPLE:
        if mh_city_count < 10:
            qa.append({
                "question": f"In which city is the company that {person} works at headquartered?",
                "answer": city,
            })
            mh_city_count += 1

    # --- comparative (10) ---
    for i in range(10):
        a, b = PEOPLE[i], PEOPLE[(i + 7) % len(PEOPLE)]
        # Who researches X and who researches Y — answer: a's company
        qa.append({
            "question": f"Compare the research fields of {a[0]} and {b[0]}. What does {a[0]} research?",
            "answer": a[3],
        })

    # --- analytical / why-style (10) ---
    for i in range(10):
        person, company, city, field = PEOPLE[i]
        qa.append({
            "question": f"Why is {person} associated with {field}?",
            "answer": f"because {person} leads research on {field} at {company}",
        })

    # --- exploratory (10) ---
    for i in range(10):
        person, company, _city, field = PEOPLE[i]
        qa.append({
            "question": f"Tell me about {person}'s role at {company}.",
            "answer": f"{person} leads research on {field} at {company}",
        })

    # --- extra factual to reach 100 (7) ---
    for _person, company, city, _field in PEOPLE[:7]:
        qa.append({
            "question": f"What is the headquarters city of {company}?",
            "answer": city,
        })

    # Clip or pad to exactly 100.
    qa = qa[:100]
    assert len(qa) == 100, f"expected 100, got {len(qa)}"

    # Attach stable qids.
    rng = random.Random(42)
    rng.shuffle(qa)  # shuffle so intent types are mixed across evaluation order
    for i, item in enumerate(qa):
        item["qid"] = f"mock-{i:03d}"

    return qa


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = OUT_DIR / "corpus.txt"
    qa_path = OUT_DIR / "qa.jsonl"

    corpus_path.write_text(build_corpus(), encoding="utf-8")
    with qa_path.open("w", encoding="utf-8") as f:
        for item in build_qa():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"corpus -> {corpus_path}  ({corpus_path.stat().st_size} bytes)")
    print(f"qa     -> {qa_path}  (100 examples)")


if __name__ == "__main__":
    main()
