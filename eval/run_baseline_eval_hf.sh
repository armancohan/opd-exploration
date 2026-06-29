#!/bin/bash
# Evaluate Qwen3-1.7B base model using HuggingFace generate (no vLLM).
# Runs AIME24, AIME25, HMMT25 on available GPUs.
# Val_n=12 matches OPSD paper protocol.
#
# Expected results — OPSD paper (Avg@12):
#   Nonthinking: AIME24~11.9%  AIME25~9.2%  HMMT25~5.0%
#   Thinking:    AIME24~51.5%  AIME25~36.7% HMMT25~23.1%

MODEL="${1:-Qwen/Qwen3-1.7B}"
MODE="${2:-nonthinking}"
VAL_N="${3:-12}"
MAX_TOKENS="${4:-2048}"

OUTPUT_DIR="eval/results"
mkdir -p "$OUTPUT_DIR"

THINKING_FLAG=""
if [ "$MODE" = "thinking" ]; then
    THINKING_FLAG="--thinking"
    MAX_TOKENS="${4:-8192}"
fi

echo "============================================================"
echo "Baseline eval (HF): $MODEL | $MODE | val_n=$VAL_N"
echo "============================================================"

# Run all three datasets in parallel on separate GPU groups
# 4 GPUs per job (device_map=auto uses all visible GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval/evaluate_hf.py \
    --model "$MODEL" \
    --dataset aime24 \
    --val_n "$VAL_N" \
    $THINKING_FLAG \
    --max_new_tokens "$MAX_TOKENS" \
    --batch_size 4 \
    --output_dir "$OUTPUT_DIR" &
PID1=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 python eval/evaluate_hf.py \
    --model "$MODEL" \
    --dataset aime25 \
    --val_n "$VAL_N" \
    $THINKING_FLAG \
    --max_new_tokens "$MAX_TOKENS" \
    --batch_size 4 \
    --output_dir "$OUTPUT_DIR" &
PID2=$!

wait $PID1 && wait $PID2

echo ""
echo "Summarizing results..."
python eval/summarize_results.py "$OUTPUT_DIR" --model "$MODEL" --mode "$MODE"
