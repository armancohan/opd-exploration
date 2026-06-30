#!/bin/bash
# Launch base OPSD, OG-OPD, and PSD in parallel on separate GPU sets.
#
# Usage:
#   bash scripts/run_comparison_experiments.sh
#
# Env overrides:
#   WANDB_PROJECT=opsd-experiments   (required for W&B logging)
#   STEPS=300                         (training steps, default 300)
#   N_TRAIN=2000                      (training samples, default 2000)
#   EVAL_STEPS=25                     (how often to eval, default 25)
#   MODEL=Qwen/Qwen3-1.7B
#   GPUS_OPSD=0,1,2,3
#   GPUS_OG=4,5,6,7
#   GPUS_PSD=0,1,2,3     (PSD; can overlap with OPSD if GPUs allow, or set to separate)
#
# To run only a subset, comment out the launch block(s) you don't need.

set -e

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-300}"
N_TRAIN="${N_TRAIN:-2000}"
EVAL_STEPS="${EVAL_STEPS:-25}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
GPUS_OPSD="${GPUS_OPSD:-0,1,2,3}"
GPUS_OG="${GPUS_OG:-4,5,6,7}"
GPUS_PSD="${GPUS_PSD:-0,1,2,3}"
N_GPUS_OPSD=$(echo "$GPUS_OPSD" | tr ',' '\n' | wc -l)
N_GPUS_OG=$(echo "$GPUS_OG" | tr ',' '\n' | wc -l)
N_GPUS_PSD=$(echo "$GPUS_PSD" | tr ',' '\n' | wc -l)

TS=$(date +%Y%m%d_%H%M%S)
OUT_OPSD="outputs/opsd_base_2k_${TS}"
OUT_OG="outputs/og_opsd_2k_${TS}"
OUT_PSD="outputs/psd_2k_${TS}"
LOG_OPSD="outputs/opsd_base_2k_${TS}.log"
LOG_OG="outputs/og_opsd_2k_${TS}.log"
LOG_PSD="outputs/psd_2k_${TS}.log"

cd "$(dirname "$0")/.."
mkdir -p outputs

echo "============================================================"
echo "Launching comparison experiments"
echo "  OPSD    → GPUs $GPUS_OPSD | output: $OUT_OPSD"
echo "  OG-OPD  → GPUs $GPUS_OG   | output: $OUT_OG"
echo "  PSD     → GPUs $GPUS_PSD  | output: $OUT_PSD"
echo "  steps=$STEPS  n_train=$N_TRAIN  eval_steps=$EVAL_STEPS"
echo "  wandb_project=${WANDB_PROJECT:-<not set>}"
echo "============================================================"
echo ""

# Common args shared by all runs
COMMON_ARGS=(
    --model "$MODEL"
    --dataset "siyanzhao/Openthoughts_math_30k_opsd"
    --n_train_samples "$N_TRAIN"
    --max_steps "$STEPS"
    --n_rollouts 4
    --batch_size 1
    --lr 5e-6
    --max_completion_length 1024
    --gradient_accumulation_steps 4
    --beta 0.0
    --temperature 1.1
    --eval_dataset "aime2024"
    --eval_steps "$EVAL_STEPS"
    --max_prompt_len 512
    --max_grad_norm 1.0
)

if [ -n "$WANDB_PROJECT" ]; then
    COMMON_ARGS+=(--wandb_project "$WANDB_PROJECT")
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Base OPSD ---
CUDA_VISIBLE_DEVICES="$GPUS_OPSD" torchrun \
    --nproc_per_node="$N_GPUS_OPSD" \
    --master_port=29500 \
    scripts/train_base_opsd.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "$OUT_OPSD" \
    --wandb_run_name "opsd-base-2k-${TS}" \
    > "$LOG_OPSD" 2>&1 &
PID_OPSD=$!
echo "Base OPSD  started (PID $PID_OPSD) → $LOG_OPSD"

# --- OG-OPD (Outcome-Gated) ---
CUDA_VISIBLE_DEVICES="$GPUS_OG" torchrun \
    --nproc_per_node="$N_GPUS_OG" \
    --master_port=29501 \
    scripts/train_base_opsd.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "$OUT_OG" \
    --wandb_run_name "og-opsd-2k-${TS}" \
    --outcome_gate \
    > "$LOG_OG" 2>&1 &
PID_OG=$!
echo "OG-OPD     started (PID $PID_OG) → $LOG_OG"

# --- PSD (Progressive Self-Distillation) ---
# PSD needs its own master_port; runs on GPUS_PSD (set to a free GPU set).
# If GPUS_PSD overlaps with OPSD/OG, launch sequentially or adjust ports.
CUDA_VISIBLE_DEVICES="$GPUS_PSD" torchrun \
    --nproc_per_node="$N_GPUS_PSD" \
    --master_port=29502 \
    scripts/train_psd.py \
    "${COMMON_ARGS[@]}" \
    --output_dir "$OUT_PSD" \
    --wandb_run_name "psd-2k-${TS}" \
    --buffer_size 5 \
    --buffer_strategy random \
    > "$LOG_PSD" 2>&1 &
PID_PSD=$!
echo "PSD        started (PID $PID_PSD) → $LOG_PSD"

echo ""
echo "All running. Monitor with:"
echo "  tail -f $LOG_OPSD"
echo "  tail -f $LOG_OG"
echo "  tail -f $LOG_PSD"
echo ""
echo "Waiting for all to finish..."

wait $PID_OPSD
STATUS_OPSD=$?
wait $PID_OG
STATUS_OG=$?
wait $PID_PSD
STATUS_PSD=$?

echo ""
echo "============================================================"
[ $STATUS_OPSD -eq 0 ] && echo "  Base OPSD: DONE → $OUT_OPSD" \
                        || echo "  Base OPSD: FAILED (exit $STATUS_OPSD) — see $LOG_OPSD"
[ $STATUS_OG -eq 0 ]   && echo "  OG-OPD:    DONE → $OUT_OG" \
                        || echo "  OG-OPD:    FAILED (exit $STATUS_OG) — see $LOG_OG"
[ $STATUS_PSD -eq 0 ]  && echo "  PSD:       DONE → $OUT_PSD" \
                        || echo "  PSD:       FAILED (exit $STATUS_PSD) — see $LOG_PSD"
echo "============================================================"
