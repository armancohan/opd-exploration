#!/bin/bash
# Run Causal-Hinge OPD experiment on 4 GPUs.
# Usage: ./scripts/run_ch_experiment.sh [--steps N] [--model MODEL]
set -e

GPUS="${GPUS:-0,1,2,3}"
N_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
STEPS="${STEPS:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ch_opsd_$(date +%Y%m%d_%H%M%S)}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

echo "=== Causal-Hinge OPD Experiment ==="
echo "GPUs: $GPUS (n=$N_GPUS)"
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="$GPUS" torchrun \
    --nproc_per_node="$N_GPUS" \
    --master_port=29500 \
    scripts/train_ch_opsd.py \
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
    --n_probe_positions 2 \
    --n_candidates 4 \
    --n_probes 2 \
    --max_probe_tokens 150 \
    --tau_benefit 0.0 \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
    ${WANDB_PROJECT:+--wandb_run_name "ch-opd"}

echo "CH-OPD experiment complete. Results: $OUTPUT_DIR/results.json"
