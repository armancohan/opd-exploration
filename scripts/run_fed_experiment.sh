#!/bin/bash
# Run Functional-Equivalence Distillation experiment on 4 GPUs.
# Usage: ./scripts/run_fed_experiment.sh [--steps N] [--model MODEL]
set -e

GPUS="${GPUS:-4,5,6,7}"
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/fed_opsd_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

echo "=== Functional-Equivalence Distillation Experiment ==="
echo "GPUs: $GPUS (n=$N_GPUS)"
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --nproc_per_node="$N_GPUS" \
    --master_port=29501 \
    scripts/train_fed_opsd.py \
    --model "$MODEL" \
    --dataset "AI-MO/NuminaMath-CoT" \
    --n_train_samples 2000 \
    --max_steps "$STEPS" \
    --n_rollouts 4 \
    --batch_size 2 \
    --lr 5e-6 \
    --max_completion_length 1024 \
    --gradient_accumulation_steps 2 \
    --beta 0.0 \
    --temperature 1.1 \
    --output_dir "$OUTPUT_DIR" \
    --eval_dataset "aime2024" \
    --eval_steps 25 \
    --tau_value 0.3 \
    --lambda_within 0.1 \
    --rho 0.5 \
    --beta_fed 0.5 \
    --n_anchor_positions 2 \
    --n_continuations_per_anchor 4 \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
    ${WANDB_PROJECT:+--wandb_run_name "fed-opd"}

echo "FED experiment complete. Results: $OUTPUT_DIR/results.json"
