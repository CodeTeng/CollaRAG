# Plan: Move eval.py & eval_ragas.py into experiments/

## Context

The user wants all experiment/evaluation scripts consolidated under `experiments/` instead of cluttering the project root. `eval.py` (offline EM/F1/ROUGE-L scorer) and `eval_ragas.py` (RAGAS semantic evaluator) currently sit at the repo root. They need to be moved to `experiments/` and all references updated.

## Steps

### 1. Move files

```
eval.py        → experiments/eval.py
eval_ragas.py  → experiments/eval_ragas.py
```

Use `git mv` to preserve history.

### 2. Fix runtime-critical code references

| File | Line | Change |
|------|------|--------|
| `experiments/utils.py` | 148 | `"eval.py"` → `"experiments/eval.py"` |
| `tests/test_eval_script.py` | 39 | `"eval.py"` → `"experiments/eval.py"` |
| `scripts/run_llm_only.py` | 316 | `eval.py` → `experiments/eval.py` in print() |

### 3. Update docstrings inside moved files

- `experiments/eval.py`: update self-referencing docstring (lines 7, 14, 19)
- `experiments/eval_ragas.py`: update self-references and cross-ref to eval.py (lines 4, 9, 17)

### 4. Update documentation

| File | What to change |
|------|---------------|
| `README.md` | Lines 77, 89: `eval.py` → `experiments/eval.py`; Lines 221-222: move from root tree to experiments subtree |
| `CLAUDE.md` | Line 42: update eval command path |
| `configs/README.md` | Lines 56, 67, 68: update eval command paths |
| `scripts/run_llm_only.py` | Lines 8, 21: update docstring paths |
| `src/chimera_rag/cli.py` | Lines 9, 16: update docstring refs |
| `src/chimera_rag/core/types.py` | Line 173: update comment |
| `tests/test_cli_main.py` | Line 33: update comment |
| `tests/test_eval_script.py` | Line 1: update docstring |

### 5. Skip (no-op)

- `.aone_copilot/plans/` — historical planning docs, not worth touching
- argparse `prog=` names and logger names — cosmetic, keep as-is

### 6. Commit

Single commit: `refactor: move eval.py and eval_ragas.py into experiments/`

## Verification

1. `uv run pytest tests/test_eval_script.py -q` — must pass (subprocess path fixed)
2. `uv run pytest -q` — full suite green
3. `grep -rn "uv run python eval\." --include="*.py" --include="*.md" --include="*.sh"` from root — no stale root-level references remain
