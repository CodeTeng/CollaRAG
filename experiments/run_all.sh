#!/usr/bin/env bash
# =============================================================================
# run_all.sh — 一键运行所有 Chimera-RAG 论文实验
#
# Usage:
#   bash experiments/run_all.sh               # 运行全部（默认 500 题/数据集）
#   bash experiments/run_all.sh --limit 50    # 快速烟雾测试（50 题/数据集）
#   bash experiments/run_all.sh --exp 1 2 3   # 只运行指定实验
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
LIMIT=500
EXPERIMENTS=""
LOG_LEVEL="INFO"

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --exp)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXPERIMENTS="$EXPERIMENTS $1"
                shift
            done
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# 默认运行所有实验
if [ -z "$EXPERIMENTS" ]; then
    EXPERIMENTS="1 2 3 4 5 6 7 8 9"
fi

# ---------------------------------------------------------------------------
# 检查前置条件
# ---------------------------------------------------------------------------
echo "=============================================="
echo "Chimera-RAG Experiment Suite"
echo "=============================================="
echo "Limit:       $LIMIT samples/dataset"
echo "Experiments: $EXPERIMENTS"
echo "Log Level:   $LOG_LEVEL"
echo "=============================================="

if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 确保依赖已安装
echo ">>> Checking dependencies..."
uv sync --all-extras --dev --quiet 2>/dev/null || true

# 创建输出目录
OUTPUT_BASE="output/experiments"
mkdir -p "$OUTPUT_BASE"

# 记录开始时间
START_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# 运行实验
# ---------------------------------------------------------------------------
run_exp() {
    local exp_num=$1
    local script=$2
    local desc=$3
    shift 3
    local extra_args=("$@")

    echo ""
    echo "======================================================"
    echo "  Experiment $exp_num: $desc"
    echo "======================================================"

    local exp_start=$(date +%s)

    if uv run python "experiments/$script" \
        --limit "$LIMIT" \
        --log-level "$LOG_LEVEL" \
        "${extra_args[@]}" 2>&1 | tee "$OUTPUT_BASE/exp${exp_num}.log"; then
        local exp_end=$(date +%s)
        local elapsed=$((exp_end - exp_start))
        echo "  [OK] Experiment $exp_num completed in ${elapsed}s"
    else
        local exp_end=$(date +%s)
        local elapsed=$((exp_end - exp_start))
        echo "  [FAIL] Experiment $exp_num failed after ${elapsed}s (see log)"
    fi
}

for exp in $EXPERIMENTS; do
    case $exp in
        1)
            run_exp 1 exp1_main_results.py \
                "Main Results (Table 1 & 2)" \
                --backbone qwen3-8b
            ;;
        2)
            run_exp 2 exp2_ablation.py \
                "Ablation Study"
            ;;
        3)
            run_exp 3 exp3_hyperparameter_sensitivity.py \
                "Hyperparameter Sensitivity"
            ;;
        4)
            run_exp 4 exp4_intent_classification.py \
                "Intent Classification Accuracy"
            ;;
        5)
            run_exp 5 exp5_retrieval_quality.py \
                "Retrieval Quality & Faithfulness"
            ;;
        6)
            run_exp 6 exp6_routing_counterfactual.py \
                "Routing Counterfactual"
            ;;
        7)
            run_exp 7 exp7_routing_distribution.py \
                "Routing Distribution"
            ;;
        8)
            run_exp 8 exp8_hop_depth.py \
                "Hop Depth Analysis"
            ;;
        9)
            run_exp 9 exp9_memory_accumulation.py \
                "Memory Accumulation Curve"
            ;;
        *)
            echo "Unknown experiment number: $exp (valid: 1-9)"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "  All experiments completed!"
echo "  Total time: ${TOTAL_ELAPSED}s"
echo "  Results:    $OUTPUT_BASE/"
echo "=============================================="

# 生成汇总 JSON
python3 -c "
import json, os
from pathlib import Path

base = Path('$OUTPUT_BASE')
summary = {'total_elapsed_s': $TOTAL_ELAPSED, 'limit': $LIMIT, 'experiments': {}}

for exp_dir in sorted(base.glob('exp*_*')):
    if exp_dir.is_dir():
        name = exp_dir.name
        result_files = list(exp_dir.glob('*results.json'))
        if result_files:
            with open(result_files[0]) as f:
                data = json.load(f)
            summary['experiments'][name] = {'status': 'ok', 'result_file': str(result_files[0])}
        else:
            summary['experiments'][name] = {'status': 'no_results'}

with open(base / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary written to {base}/summary.json')
" 2>/dev/null || echo "(summary generation skipped)"
