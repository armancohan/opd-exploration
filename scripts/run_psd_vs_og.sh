#!/bin/bash
# Two-variant comparison with the FIXED reward verifier (math_verify backend):
#   PSD     (Progressive Self-Distillation)  -> GPUs 0,1,2,3
#   OG-OPD  (Outcome-Gated OPSD)             -> GPUs 4,5,6,7
#
# Base OPSD is intentionally NOT run here (already run previously). Both arms
# below were effectively no-ops under the broken verifier (PSD's buffer never
# filled; outcome-gating gated everything to ~0 reward) and only become
# meaningful now that the reward signal works.
#
# Usage: bash scripts/run_psd_vs_og.sh

set -e
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-300}"
# Small train set so problems repeat within max_steps — required for PSD's buffer
# to actually be hit (otherwise PSD degenerates to plain OPSD). 128 problems over
# 300 steps ≈ 2.3 epochs: buffer fills in epoch 1, gets used from ~step 128 on.
N_TRAIN="${N_TRAIN:-128}"
EVAL_STEPS="${EVAL_STEPS:-25}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
GPUS_PSD="${GPUS_PSD:-0,1,2,3}"
GPUS_OG="${GPUS_OG:-4,5,6,7}"
N_GPUS_PSD=$(echo "$GPUS_PSD" | tr ',' '\n' | wc -l)
N_GPUS_OG=$(echo "$GPUS_OG" | tr ',' '\n' | wc -l)

TS=$(date +%Y%m%d_%H%M%S)
OUT_PSD="outputs/psd_2k_${TS}"
OUT_OG="outputs/og_opsd_2k_${TS}"
LOG_PSD="outputs/psd_2k_${TS}.log"
LOG_OG="outputs/og_opsd_2k_${TS}.log"
mkdir -p outputs

echo "============================================================"
echo "  PSD     -> GPUs $GPUS_PSD | $OUT_PSD"
echo "  OG-OPD  -> GPUs $GPUS_OG  | $OUT_OG"
echo "  model=$MODEL steps=$STEPS n_train=$N_TRAIN eval_steps=$EVAL_STEPS"
echo "  wandb_project=${WANDB_PROJECT:-<not set>}"
echo "============================================================"

COMMON_ARGS=(
    --model "$MODEL"
    --dataset "siyanzhao/Openthoughts_math_30k_opsd"
    --n_train_samples "$N_TRAIN"
    --max_steps "$STEPS"
    --n_rollouts 4
    --batch_size 1
    --lr 5e-6
    --max_completion_length 3072
    --gradient_accumulation_steps 4
    --beta 0.0
    --temperature 1.1
    --eval_dataset "aime2024,math500"
    --eval_steps "$EVAL_STEPS"
    --max_prompt_len 512
    --max_grad_norm 1.0
)
[ -n "$WANDB_PROJECT" ] && COMMON_ARGS+=(--wandb_project "$WANDB_PROJECT")

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- PSD ---
CUDA_VISIBLE_DEVICES="$GPUS_PSD" torchrun \
    --nproc_per_node="$N_GPUS_PSD" --master_port=29502 \
    scripts/train_psd.py "${COMMON_ARGS[@]}" \
    --output_dir "$OUT_PSD" --wandb_run_name "psd-2k-${TS}" \
    --buffer_size 5 --buffer_strategy random \
    > "$LOG_PSD" 2>&1 &
PID_PSD=$!
echo "PSD started (PID $PID_PSD) -> $LOG_PSD"

# --- OG-OPD ---
CUDA_VISIBLE_DEVICES="$GPUS_OG" torchrun \
    --nproc_per_node="$N_GPUS_OG" --master_port=29501 \
    scripts/train_base_opsd.py "${COMMON_ARGS[@]}" \
    --output_dir "$OUT_OG" --wandb_run_name "og-opsd-2k-${TS}" \
    --outcome_gate \
    > "$LOG_OG" 2>&1 &
PID_OG=$!
echo "OG-OPD started (PID $PID_OG) -> $LOG_OG"

echo ""
echo "Monitor: tail -f $LOG_PSD   |   tail -f $LOG_OG"
wait $PID_PSD; STATUS_PSD=$?
wait $PID_OG;  STATUS_OG=$?
echo "============================================================"
[ $STATUS_PSD -eq 0 ] && echo "  PSD:    DONE -> $OUT_PSD" || echo "  PSD:    FAILED ($STATUS_PSD) — see $LOG_PSD"
[ $STATUS_OG -eq 0 ]  && echo "  OG-OPD: DONE -> $OUT_OG"  || echo "  OG-OPD: FAILED ($STATUS_OG) — see $LOG_OG"
echo "============================================================"
