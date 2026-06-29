#!/bin/bash
# Baseline eval matching OPSD paper protocol exactly.
# Uses vLLM with enforce_eager=True, VLLM_USE_V1=0 to bypass flashinfer.
#
# Expected results (Avg@12, nonthinking):
#   Qwen3-1.7B: AIME24=11.9%  AIME25=9.2%  HMMT25=5.0%
#
# Usage:
#   bash eval/run_baseline_eval.sh                        # nonthinking, all 3 datasets
#   bash eval/run_baseline_eval.sh Qwen/Qwen3-1.7B thinking
#   bash eval/run_baseline_eval.sh Qwen/Qwen3-4B nonthinking

MODEL="${1:-Qwen/Qwen3-1.7B}"
MODE="${2:-nonthinking}"
TP="${3:-4}"
OUTPUT_DIR="eval/results"
mkdir -p "$OUTPUT_DIR"

THINKING_FLAG=""
[ "$MODE" = "thinking" ] && THINKING_FLAG="--thinking"

export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1

echo "============================================================"
echo "Baseline eval: $MODEL | $MODE | tp=$TP"
echo "============================================================"

# Run 3 datasets sequentially on the same 4-GPU set
for DATASET in aime24 aime25 hmmt25; do
    echo ""
    echo "--- $DATASET ---"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --val_n 12 \
        $THINKING_FLAG \
        --tp "$TP" \
        --output_dir "$OUTPUT_DIR"
done

echo ""
echo "Summarizing..."
python eval/summarize_results.py "$OUTPUT_DIR" --model "$MODEL" --mode "$MODE"
