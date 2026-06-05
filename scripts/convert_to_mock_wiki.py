"""Convert downloaded datasets into the mock_wiki layout.

Target layout (one per dataset), matching MockWikiLoader's contract:

    data/<name>/corpus.txt   # paragraphs separated by a blank line (\\n\\n)
    data/<name>/qa.jsonl     # one {"question","answer","qid"} per line

Source files (produced by scripts/download_datasets.py):
    hotpotqa : data/hotpotqa/hotpotqa.json            (JSON array, context=[[title,[sents]]])
    two_wiki : data/two_wiki/dev.json                 (same shape as hotpotqa)
    musique  : data/musique/musique_ans_v1.0_dev.jsonl(JSONL, paragraphs=[{title,paragraph_text}])
    nq       : data/nq/nq.jsonl                        (open-domain QA, no corpus)
    popqa    : data/popqa/popqa.jsonl                  (open-domain QA, no corpus)

NQ / PopQA are open-domain (no bundled passages), so an empty placeholder
corpus.txt is written to keep all five datasets structurally uniform.

Usage::

    .venv/bin/python scripts/convert_to_mock_wiki.py                 # all
    .venv/bin/python scripts/convert_to_mock_wiki.py --only nq popqa # subset
    .venv/bin/python scripts/convert_to_mock_wiki.py --limit 500     # cap QA rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _write_corpus(paragraphs: list[str], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "corpus.txt"
    cleaned = [p.strip() for p in paragraphs if p and p.strip()]
    path.write_text("\n\n".join(cleaned), encoding="utf-8")
    print(f"  [corpus] {len(cleaned)} paragraphs -> {path.relative_to(REPO_ROOT)}")
    return len(cleaned)


def _write_qa(rows: list[dict], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "qa.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  [qa]     {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")
    return len(rows)


def _read_json_array(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"source not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"source not found: {path}")
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _qa_row(question: str, answer: str, qid: str) -> dict:
    return {"question": str(question), "answer": str(answer), "qid": str(qid)}


# ---------------------------------------------------------------------------
# HotpotQA / 2Wiki: context = [[title, [sentences...]], ...]
# corpus = one paragraph per unique title (sentences joined); dedup by title.
# ---------------------------------------------------------------------------
def _convert_hotpot_like(src: Path, out_dir: Path, source_tag: str, limit: int | None) -> None:
    raw = _read_json_array(src)
    by_title: dict[str, str] = {}
    qa_rows: list[dict] = []
    for item in raw:
        for ctx in item.get("context", []):
            if isinstance(ctx, (list, tuple)) and len(ctx) >= 2 and isinstance(ctx[1], list):
                title = str(ctx[0])
                if title not in by_title:
                    sentences = [s for s in ctx[1] if isinstance(s, str)]
                    by_title[title] = " ".join(sentences).strip()
        if limit is None or len(qa_rows) < limit:
            qa_rows.append(
                _qa_row(
                    item["question"],
                    item["answer"],
                    item.get("_id", item.get("id", f"{source_tag}-{len(qa_rows)}")),
                )
            )
    _write_corpus([f"{title}. {body}" for title, body in by_title.items() if body], out_dir)
    _write_qa(qa_rows, out_dir)


def convert_hotpotqa(limit: int | None) -> None:
    print("[hotpotqa]")
    _convert_hotpot_like(
        DATA_DIR / "hotpotqa" / "hotpotqa.json", DATA_DIR / "hotpotqa", "hpq", limit
    )


def convert_two_wiki(limit: int | None) -> None:
    print("[two_wiki]")
    _convert_hotpot_like(
        DATA_DIR / "two_wiki" / "dev.json", DATA_DIR / "two_wiki", "2wiki", limit
    )


# ---------------------------------------------------------------------------
# MuSiQue: paragraphs = [{title, paragraph_text}]
# corpus = one paragraph per unique (title, paragraph_text); dedup by text.
# ---------------------------------------------------------------------------
def convert_musique(limit: int | None) -> None:
    print("[musique]")
    src = DATA_DIR / "musique" / "musique_ans_v1.0_dev.jsonl"
    seen: set[str] = set()
    paragraphs: list[str] = []
    qa_rows: list[dict] = []
    for row in _iter_jsonl(src):
        for para in row.get("paragraphs", []):
            text = str(para.get("paragraph_text", "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            title = str(para.get("title", "")).strip()
            paragraphs.append(f"{title}. {text}" if title else text)
        if limit is None or len(qa_rows) < limit:
            qa_rows.append(
                _qa_row(row["question"], row["answer"], row.get("id", f"musique-{len(qa_rows)}"))
            )
    _write_corpus(paragraphs, DATA_DIR / "musique")
    _write_qa(qa_rows, DATA_DIR / "musique")


# ---------------------------------------------------------------------------
# NQ / PopQA: open-domain, no bundled corpus -> empty placeholder corpus.txt
# ---------------------------------------------------------------------------
def _convert_open_domain(src: Path, out_dir: Path, tag: str, limit: int | None) -> None:
    qa_rows: list[dict] = []
    for row in _iter_jsonl(src):
        if limit is not None and len(qa_rows) >= limit:
            break
        qa_rows.append(
            _qa_row(row["question"], row["answer"], row.get("qid", f"{tag}-{len(qa_rows)}"))
        )
    # open-domain: no passages ship with the dataset; keep an empty corpus so
    # the directory layout matches the other four datasets.
    _write_corpus([], out_dir)
    _write_qa(qa_rows, out_dir)


def convert_nq(limit: int | None) -> None:
    print("[nq] (open-domain: empty corpus placeholder)")
    _convert_open_domain(DATA_DIR / "nq" / "nq.jsonl", DATA_DIR / "nq", "nq", limit)


def convert_popqa(limit: int | None) -> None:
    print("[popqa] (open-domain: empty corpus placeholder)")
    _convert_open_domain(DATA_DIR / "popqa" / "popqa.jsonl", DATA_DIR / "popqa", "popqa", limit)


CONVERTERS = {
    "hotpotqa": convert_hotpotqa,
    "two_wiki": convert_two_wiki,
    "musique": convert_musique,
    "nq": convert_nq,
    "popqa": convert_popqa,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(CONVERTERS),
        help="only convert these datasets (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap number of QA rows per dataset (default: all)",
    )
    args = parser.parse_args()

    targets = args.only or list(CONVERTERS)
    failures: list[str] = []
    for name in targets:
        try:
            CONVERTERS[name](args.limit)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"[{name}] ERROR: {type(e).__name__}: {e}")
            failures.append(name)

    print("\n=== summary ===")
    for name in targets:
        status = "FAILED" if name in failures else "ok"
        print(f"  {name:10s} {status}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
