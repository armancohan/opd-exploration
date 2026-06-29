#!/bin/bash
# Evaluate Qwen3-1.7B base model on AIME24, AIME25, HMMT25
# Runs three evaluations in parallel using different GPU pairs.
# Expected results (from OPSD paper, thinking mode, Avg@12):
#   AIME24: ~51.5%   AIME25: ~36.7%   HMMT25: ~23.1%
# Non-thinking mode, Avg@12:
#   AIME24: ~11.9%   AIME25: ~9.2%    HMMT25: ~5.0%

MODEL="${1:-Qwen/Qwen3-1.7B}"
MODE="${2:-thinking}"  # "thinking" or "nonthinking"
VAL_N="${3:-12}"
MAX_TOKENS="${4:-16384}"  # Use 38912 to match OPSD paper exactly (slower)
OUTPUT_DIR="eval/results"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Baseline evaluation: $MODEL"
echo "Mode: $MODE  |  Val-N: $VAL_N  |  Max tokens: $MAX_TOKENS"
echo "============================================================"

THINKING_FLAG=""
if [ "$MODE" = "thinking" ]; then
    THINKING_FLAG="--thinking"
fi

# Run all three datasets in parallel on different GPU groups (2 GPUs each)
CUDA_VISIBLE_DEVICES=0,1 python eval/evaluate.py \
    --model "$MODEL" \
    --dataset aime24 \
    --val_n "$VAL_N" \
    $THINKING_FLAG \
    --temperature 1.0 \
    --max_new_tokens "$MAX_TOKENS" \
    --tp 2 \
    --output_dir "$OUTPUT_DIR" &
PID1=$!

CUDA_VISIBLE_DEVICES=2,3 python eval/evaluate.py \
    --model "$MODEL" \
    --dataset aime25 \
    --val_n "$VAL_N" \
    $THINKING_FLAG \
    --temperature 1.0 \
    --max_new_tokens "$MAX_TOKENS" \
    --tp 2 \
    --output_dir "$OUTPUT_DIR" &
PID2=$!

CUDA_VISIBLE_DEVICES=4,5 python eval/evaluate.py \
    --model "$MODEL" \
    --dataset hmmt25 \
    --val_n "$VAL_N" \
    $THINKING_FLAG \
    --temperature 1.0 \
    --max_new_tokens "$MAX_TOKENS" \
    --tp 2 \
    --output_dir "$OUTPUT_DIR" &
PID3=$!

wait $PID1
STATUS1=$?
wait $PID2
STATUS2=$?
wait $PID3
STATUS3=$?

echo ""
echo "============================================================"
echo "BASELINE EVAL COMPLETE"
echo "============================================================"
python eval/summarize_results.py "$OUTPUT_DIR" --model "$MODEL" --mode "$MODE"

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ] || [ $STATUS3 -ne 0 ]; then
    echo "WARNING: one or more evaluations had non-zero exit codes"
    exit 1
fi
