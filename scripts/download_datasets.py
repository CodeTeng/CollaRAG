"""Download GraphRAG evaluation datasets into ``data/`` in Chimera-RAG format.

Datasets handled (dev splits):
  - HotpotQA       -> data/hotpotqa/hotpotqa.json   (HotpotQA JSON-array shape)
  - 2WikiMultihopQA-> data/two_wiki/dev.json        (HotpotQA-compatible shape)
  - MuSiQue        -> data/musique/musique_ans_v1.0_dev.jsonl (JSONL w/ paragraphs)
  - Natural Q.(NQ) -> data/nq/nq.jsonl              (generic_jsonl: question/answer)
  - PopQA          -> data/popqa/popqa.jsonl        (generic_jsonl: question/answer)

The target shapes are dictated by the loaders in
``src/chimera_rag/datasets/`` so the produced files load without extra config
changes (NQ / PopQA additionally need a ``datasets:`` entry in the yaml).

Usage::

    .venv/bin/python scripts/download_datasets.py                 # all, full dev
    .venv/bin/python scripts/download_datasets.py --only nq popqa # subset
    .venv/bin/python scripts/download_datasets.py --limit 200     # cap per dataset
    HF_ENDPOINT=https://hf-mirror.com .venv/bin/python scripts/download_datasets.py

If the primary HuggingFace dataset id fails, the next candidate id is tried.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Force all HuggingFace traffic through the mirror; the direct huggingface.co
# host is unreachable from CN networks while hf-mirror.com is stable.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

def _ensure_ca_bundle() -> str | None:
    """Return a CA bundle that trusts both public CAs and the corp proxy CA.

    Corporate networks (e.g. AntGroup StarPoint) MITM TLS with an internal
    root CA that lives in the macOS keychain but not in certifi's bundle.
    We merge certifi + the system keychain roots into one file so Python's
    requests/gdown can verify proxied connections the same way ``curl`` does.
    """
    try:
        import certifi
    except ModuleNotFoundError:
        return None

    combined = REPO_ROOT / ".certs" / "combined_ca.pem"
    if combined.is_file() and combined.stat().st_size > 0:
        return str(combined)

    combined.parent.mkdir(parents=True, exist_ok=True)
    parts = [Path(certifi.where()).read_text(encoding="utf-8")]

    import subprocess

    keychains = [
        "/System/Library/Keychains/SystemRootCertificates.keychain",
        "/Library/Keychains/System.keychain",
    ]
    for kc in keychains:
        try:
            out = subprocess.run(
                ["security", "find-certificate", "-a", "-p", kc],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip():
                parts.append(out.stdout)
        except (OSError, subprocess.SubprocessError):
            continue

    combined.write_text("\n".join(parts), encoding="utf-8")
    return str(combined)


_CA_BUNDLE = _ensure_ca_bundle()
if _CA_BUNDLE:
    os.environ.setdefault("SSL_CERT_FILE", _CA_BUNDLE)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA_BUNDLE)
    os.environ.setdefault("CURL_CA_BUNDLE", _CA_BUNDLE)

# Google Drive file id for the official MuSiQue distribution
# (from StonyBrookNLP/musique download_data.sh -> musique_v1.0.zip).
# 2Wiki's official files are on Dropbox (unreachable here) so it uses the HF
# mirror instead; see build_two_wiki.
GDRIVE_FILES = {
    "musique": "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h",
}


def _load_hf(candidates: list[tuple[str, str | None, str]], split: str):
    """Try each (repo_id, config, split_override) candidate until one loads."""
    from datasets import load_dataset

    last_err: Exception | None = None
    for repo_id, config, split_name in candidates:
        use_split = split_name or split
        try:
            print(f"  [hf] trying {repo_id} (config={config}, split={use_split}) ...")
            if config:
                return load_dataset(repo_id, config, split=use_split)
            return load_dataset(repo_id, split=use_split)
        except Exception as e:  # noqa: BLE001 - we intentionally fall through
            print(f"  [hf] failed: {type(e).__name__}: {e}")
            last_err = e
    raise RuntimeError(f"all candidates failed; last error: {last_err}")


def _write_json_array(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [write] {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  [write] {len(rows)} rows -> {path.relative_to(REPO_ROOT)}")


def _capped(iterable, limit: int | None):
    for i, item in enumerate(iterable):
        if limit is not None and i >= limit:
            break
        yield item


# ---------------------------------------------------------------------------
# HotpotQA -> HotpotQALoader JSON-array shape
# ---------------------------------------------------------------------------
def build_hotpotqa(limit: int | None) -> None:
    print("[hotpotqa]")
    ds = _load_hf(
        [
            ("hotpotqa/hotpot_qa", "distractor", "validation"),
            ("hotpot_qa", "distractor", "validation"),
        ],
        split="validation",
    )
    rows: list[dict] = []
    for ex in _capped(ds, limit):
        ctx = ex.get("context", {})
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        context = [[t, list(s)] for t, s in zip(titles, sentences)]
        rows.append(
            {
                "_id": str(ex.get("id", f"hpq-{len(rows)}")),
                "question": ex["question"],
                "answer": ex["answer"],
                "context": context,
                "type": ex.get("type"),
                "level": ex.get("level"),
            }
        )
    _write_json_array(rows, DATA_DIR / "hotpotqa" / "hotpotqa.json")


def _gdrive_download(file_id: str, dest: Path) -> Path:
    """Download a Google Drive file by id into ``dest`` (skip if present)."""
    import gdown

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  [gdrive] cached {dest.relative_to(REPO_ROOT)}")
        return dest

    # Use the merged CA bundle (public + corp proxy roots) so gdown can
    # verify the MITM'd Google Drive connection.
    verify: bool | str = _CA_BUNDLE if _CA_BUNDLE else True

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  [gdrive] downloading {file_id} -> {dest.relative_to(REPO_ROOT)}")
    gdown.download(url, str(dest), quiet=False, verify=verify)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise RuntimeError(f"gdown produced no file for {file_id}")
    return dest


def _extract_zip(zip_path: Path, member_substr: str) -> str:
    """Return text of the first zip member whose name contains ``member_substr``."""
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        match = next((n for n in names if member_substr in n and not n.endswith("/")), None)
        if match is None:
            raise RuntimeError(
                f"no member matching {member_substr!r} in {zip_path.name}; have={names[:10]}"
            )
        print(f"  [zip] reading member {match}")
        return zf.read(match).decode("utf-8")


# ---------------------------------------------------------------------------
# 2WikiMultihopQA -> HotpotQA-compatible shape (two_wiki loader subclasses it)
# The official files live on Dropbox (unreachable here), so we use the
# faithful HF mirror. Its ``context`` field is a JSON *string* encoding
# [[title, [sentences]], ...], which we parse back into nested lists.
# ---------------------------------------------------------------------------
def _coerce_2wiki_context(raw_ctx) -> list[list]:
    if isinstance(raw_ctx, str):
        try:
            raw_ctx = json.loads(raw_ctx)
        except json.JSONDecodeError:
            return []
    context: list[list] = []
    for item in raw_ctx or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            sents = item[1]
            if isinstance(sents, str):
                sents = [sents]
            context.append([str(item[0]), list(sents)])
    return context


def build_two_wiki(limit: int | None) -> None:
    print("[two_wiki]")
    ds = _load_hf(
        [
            ("scholarly-shadows-syndicate/2wikimultihopqa", None, "validation"),
        ],
        split="validation",
    )
    rows: list[dict] = []
    for ex in _capped(ds, limit):
        rows.append(
            {
                "_id": str(ex.get("_id", ex.get("id", f"2wiki-{len(rows)}"))),
                "question": ex["question"],
                "answer": ex["answer"],
                "context": _coerce_2wiki_context(ex.get("context")),
                "type": ex.get("type"),
                "level": None,
            }
        )
    _write_json_array(rows, DATA_DIR / "two_wiki" / "dev.json")


# ---------------------------------------------------------------------------
# MuSiQue -> JSONL with id/question/answer/paragraphs[{title,paragraph_text}]
# Official release: Google Drive zip containing musique_ans_v1.0_dev.jsonl.
# ---------------------------------------------------------------------------
def build_musique(limit: int | None) -> None:
    print("[musique]")
    zip_path = DATA_DIR / "musique" / "_musique_release.zip"
    _gdrive_download(GDRIVE_FILES["musique"], zip_path)
    text = _extract_zip(zip_path, "musique_ans_v1.0_dev.jsonl")
    rows: list[dict] = []
    for i, line in enumerate(text.splitlines()):
        if limit is not None and len(rows) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        ex = json.loads(line)
        paragraphs = [
            {
                "title": p.get("title", ""),
                "paragraph_text": p.get("paragraph_text", ""),
            }
            for p in ex.get("paragraphs", [])
        ]
        rows.append(
            {
                "id": str(ex.get("id", f"musique-{len(rows)}")),
                "question": ex["question"],
                "answer": ex["answer"],
                "paragraphs": paragraphs,
            }
        )
    _write_jsonl(rows, DATA_DIR / "musique" / "musique_ans_v1.0_dev.jsonl")


# ---------------------------------------------------------------------------
# NQ -> generic_jsonl: question/answer
# ---------------------------------------------------------------------------
def build_nq(limit: int | None) -> None:
    print("[nq]")
    ds = _load_hf(
        [
            ("google-research-datasets/nq_open", None, "validation"),
            ("nq_open", None, "validation"),
        ],
        split="validation",
    )
    rows: list[dict] = []
    for ex in _capped(ds, limit):
        answers = ex.get("answer", [])
        answer = answers[0] if isinstance(answers, list) and answers else str(answers)
        rows.append(
            {
                "qid": f"nq-{len(rows)}",
                "question": ex["question"],
                "answer": answer,
            }
        )
    _write_jsonl(rows, DATA_DIR / "nq" / "nq.jsonl")


# ---------------------------------------------------------------------------
# PopQA -> generic_jsonl: question/answer
# ---------------------------------------------------------------------------
def build_popqa(limit: int | None) -> None:
    print("[popqa]")
    ds = _load_hf(
        [
            ("akariasai/PopQA", None, "test"),
        ],
        split="test",
    )
    rows: list[dict] = []
    for ex in _capped(ds, limit):
        # possible answers stored as a stringified list in "possible_answers"
        raw = ex.get("possible_answers") or ex.get("obj") or ex.get("answers")
        answer = ""
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                answer = parsed[0] if isinstance(parsed, list) and parsed else raw
            except json.JSONDecodeError:
                answer = raw
        elif isinstance(raw, list) and raw:
            answer = raw[0]
        rows.append(
            {
                "qid": str(ex.get("id", f"popqa-{len(rows)}")),
                "question": ex["question"],
                "answer": str(answer),
            }
        )
    _write_jsonl(rows, DATA_DIR / "popqa" / "popqa.jsonl")


BUILDERS = {
    "hotpotqa": build_hotpotqa,
    "two_wiki": build_two_wiki,
    "musique": build_musique,
    "nq": build_nq,
    "popqa": build_popqa,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(BUILDERS),
        help="only build these datasets (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap number of rows per dataset (default: full dev split)",
    )
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ModuleNotFoundError:
        print("ERROR: `datasets` not installed. Run: uv pip install datasets")
        return 1

    targets = args.only or list(BUILDERS)
    failures: list[str] = []
    for name in targets:
        try:
            BUILDERS[name](args.limit)
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
