"""LLM-only baseline for Chimera-RAG.

直接把 question 丢给 LLM 拿答案，不走任何检索/图谱/意图分类，用来：
  - 复现 Relink 论文 Table 1 里的 "LLM only" 行
  - 在本地 ollama (Qwen3-8B) 与云端 DeepSeek-V3 之间做同口径对比

产物与 ``main.py infer`` 相同（predictions.jsonl + meta.json），可直接喂给
``experiments/eval.py`` 打 EM/F1/ROUGE-L。

用法::

    # 1) 启动 ollama 并 pull qwen3:8b（仅 ollama 路径需要）
    brew services start ollama && ollama pull qwen3:8b

    # 2) 跑一个数据集
    uv run python scripts/run_llm_only.py \\
        --dataset hotpotqa --provider ollama --model qwen3:8b \\
        --limit 500 --out output/llm_only/qwen3_8b/hotpotqa

    # 3) 打分
    uv run python experiments/eval.py --predictions output/llm_only/qwen3_8b/hotpotqa

依赖：openai>=1.0（已在 uv 主依赖里）；ollama 仅需要本地 11434 端口可访问。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a hard dep
    tqdm = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPT_TEMPLATE = (
    "You are answering a multi-hop question using only your own knowledge. "
    "Output ONLY the final answer as short as possible (a name, a date, "
    "yes/no, or a short phrase). No explanation, no quotes, no punctuation "
    "around it.\n\n"
    "Question: {question}\n"
    "Answer:"
)

# provider -> (base_url, api_key_env, default_model)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "ollama":   ("http://211.71.76.154:11434/v1", "OLLAMA_KEY",       "qwen3:32b"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "openai":   ("https://api.openai.com/v1",   "OPENAI_API_KEY",   "gpt-4o-mini"),
}

DATASETS = {
    "hotpotqa": PROJECT_ROOT / "data" / "hotpotqa" / "qa.jsonl",
    "two_wiki": PROJECT_ROOT / "data" / "two_wiki" / "qa.jsonl",
    "musique":  PROJECT_ROOT / "data" / "musique"  / "qa.jsonl",
}


def load_qa(dataset: str, limit: int, seed: int) -> list[dict[str, Any]]:
    path = DATASETS[dataset]
    if not path.is_file():
        sys.exit(
            f"qa file missing: {path}\n"
            f"  run `python scripts/download_datasets.py` and "
            f"`python scripts/convert_to_mock_wiki.py` first."
        )
    rows = [
        json.loads(ln)
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    if limit and 0 < limit < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def _strip_thinking(text: str) -> str:
    """Qwen3 thinking-mode 可能留下 <think>...</think> 包裹。去掉它。"""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    # 去掉残留的开闭标签
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def _strip_quotes(text: str) -> str:
    text = text.strip()
    for q in ("'", '"', "“", "”", "‘", "’"):
        if text.startswith(q) and text.endswith(q) and len(text) > 1:
            text = text[1:-1].strip()
    # 去掉常见前缀
    for prefix in ("Answer:", "answer:", "ANSWER:", "Final answer:", "答案:", "答案：" ):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def build_client(provider: str):
    try:
        from openai import OpenAI
    except ImportError as e:
        sys.exit(f"openai SDK not installed: {e}\n  run `uv sync --extra llm`")
    base_url, key_env, _ = PROVIDERS[provider]
    api_key = os.getenv(key_env, "") or ("ollama" if provider == "ollama" else "")
    if not api_key:
        sys.exit(f"env var {key_env} not set for provider={provider}")
    return OpenAI(base_url=base_url, api_key=api_key)


def _truncate(text: str, width: int) -> str:
    """单行展示用：去换行，超长截断，便于 tqdm postfix / 日志整齐显示。"""
    text = (text or "").replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM-only baseline runner")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--provider", default="ollama", choices=list(PROVIDERS))
    ap.add_argument("--model", default="qwen3:8b",
                    help="override default model for the provider")
    ap.add_argument("--limit", type=int, default=500,
                    help="number of questions to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-request timeout in seconds")
    ap.add_argument("--retries", type=int, default=2,
                    help="retries on transient errors")
    ap.add_argument("--no-think", action="store_true",
                    help="append /no_think to prompt (Qwen3 only)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="print a verbose example every N steps (set 0 to disable)")
    ap.add_argument("--no-progress-bar", action="store_true",
                    help="disable tqdm progress bar (use plain prints only)")
    args = ap.parse_args()

    _, _, default_model = PROVIDERS[args.provider]
    model = args.model or default_model

    rows = load_qa(args.dataset, args.limit, args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args.provider)

    per_example: list[dict[str, Any]] = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    n_errors = 0
    n_empty = 0  # 模型返回空字符串的次数

    # ---------- run header ----------
    print("=" * 72)
    print("LLM-only baseline")
    print("=" * 72)
    print(f"  dataset      : {args.dataset}  ({len(rows)} questions, seed={args.seed})")
    print(f"  provider     : {args.provider}")
    print(f"  model        : {model}")
    print(f"  temperature  : {args.temperature}")
    print(f"  max_tokens   : {args.max_tokens}")
    print(f"  timeout (s)  : {args.timeout}")
    print(f"  retries      : {args.retries}")
    print(f"  no_think     : {args.no_think}")
    print(f"  out          : {out_dir}")
    print("-" * 72)

    use_bar = (tqdm is not None) and (not args.no_progress_bar)
    iterator = tqdm(
        rows,
        total=len(rows),
        desc=f"{args.dataset}/{args.provider}",
        unit="q",
        dynamic_ncols=True,
    ) if use_bar else rows

    t_run = time.time()
    for i, row in enumerate(iterator, 1):
        question = row["question"]
        prompt_q = question + (" /no_think" if args.no_think else "")
        prompt = PROMPT_TEMPLATE.format(question=prompt_q)

        pred = ""
        t_call = time.time()
        last_err: Exception | None = None
        attempts_used = 0
        for attempt in range(args.retries + 1):
            attempts_used = attempt + 1
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                raw = (resp.choices[0].message.content or "")
                pred = _strip_quotes(_strip_thinking(raw))
                if resp.usage:
                    usage_total["prompt_tokens"] += resp.usage.prompt_tokens or 0
                    usage_total["completion_tokens"] += resp.usage.completion_tokens or 0
                    usage_total["total_tokens"] += resp.usage.total_tokens or 0
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < args.retries:
                    # 重试日志：用 tqdm.write 避免把进度条搞乱
                    msg = (
                        f"  [{i}/{len(rows)}] retry {attempt + 1}/{args.retries} "
                        f"after error: {type(e).__name__}: {_truncate(str(e), 120)}"
                    )
                    (tqdm.write if use_bar else print)(msg)
                    time.sleep(1.5 * (attempt + 1))

        if last_err is not None:
            n_errors += 1
            msg = (
                f"  [{i}/{len(rows)}] ERROR after {args.retries} retries: "
                f"{type(last_err).__name__}: {_truncate(str(last_err), 120)}"
            )
            (tqdm.write if use_bar else print)(msg)

        if not pred:
            n_empty += 1

        latency_s = time.time() - t_call
        per_example.append({
            "qid": row.get("qid", str(i)),
            "question": question,
            "reference": row["answer"],   # 项目约定字段名（被 scorer / reporter 读取）
            "prediction": pred,           # 项目约定字段名（被 scorer / reporter 读取）
            "latency_s": round(latency_s, 3),
        })

        # ---------- live progress signals ----------
        if use_bar:
            elapsed = time.time() - t_run
            avg = elapsed / i
            iterator.set_postfix(  # type: ignore[union-attr]
                {
                    "avg": f"{avg:.2f}s",
                    "err": n_errors,
                    "empty": n_empty,
                    "tok": usage_total["total_tokens"],
                    "last": _truncate(pred or "<empty>", 40),
                },
                refresh=False,
            )

        # 每 N 条详细日志一次（用 tqdm.write 不破坏进度条）
        if args.progress_every and (i % args.progress_every == 0 or i == len(rows)):
            line = (
                f"  [{i:>4}/{len(rows)}] "
                f"lat={latency_s:5.2f}s "
                f"tries={attempts_used} "
                f"| Q={_truncate(question, 60)!r}"
                f" | ref={_truncate(row['answer'], 30)!r}"
                f" | pred={_truncate(pred, 40)!r}"
            )
            (tqdm.write if use_bar else print)(line)

    if use_bar and hasattr(iterator, "close"):
        iterator.close()  # type: ignore[union-attr]

    elapsed_s = time.time() - t_run
    meta = {
        "config_name": f"llm_only.{args.provider}.{model.replace(':', '_').replace('/', '_')}",
        "method": "llm_only",
        "dataset": args.dataset,
        "n_samples": len(per_example),
        "seed": args.seed,
        "limit": args.limit,
        "provider": args.provider,
        "model": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "no_think": args.no_think,
        "n_errors": n_errors,
        "n_empty": n_empty,
        "usage_total": usage_total,
        "elapsed_s": round(elapsed_s, 2),
        "prompt_template": PROMPT_TEMPLATE,
    }

    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in per_example:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------- run footer ----------
    avg_latency = elapsed_s / max(len(per_example), 1)
    print("-" * 72)
    print(f"  wrote        : {pred_path}")
    print(f"  rows         : {len(per_example)}")
    print(f"  errors       : {n_errors}")
    print(f"  empty preds  : {n_empty}")
    print(f"  elapsed      : {elapsed_s:.1f}s  (avg {avg_latency:.2f}s / q)")
    print(f"  usage        : {usage_total}")
    print(f"  next         : uv run python experiments/eval.py --predictions {out_dir} --metrics em,f1,rouge_l")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    # uv run python scripts/run_llm_only.py --dataset hotpotqa/two_wiki/musique --provider ollama --model qwen3:8b --limit 500 --out output/llm_only/qwen3_8b/hotpotqa
    # uv run python scripts/run_llm_only.py --dataset musique --provider ollama --model gemma4:12b --limit 500 --out output/llm_only/gemma4_12b/musique
    raise SystemExit(main())
