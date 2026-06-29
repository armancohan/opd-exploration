# OPD Exploration

Implementations of **Causal-Hinge OPD** (CH-OPD) and **Functional-Equivalence Distillation** (FED), two extensions of [On-Policy Self-Distillation (OPSD)](https://arxiv.org/abs/2506.xxxx) for improving math reasoning in LLMs.

## Methods

### OPSD (baseline)
The student and teacher are the same model. The student sees only the problem; the teacher sees problem + reference solution. A token-level generalized JSD loss distills the teacher's richer distribution into the student. `beta=0` → reverse KL, `beta=0.5` → symmetric JSD, `beta=1` → forward KL.

### Causal-Hinge OPD (CH-OPD)
Extends OPSD by masking distillation to positions where the teacher's benefit `B_t > tau`. Benefit is estimated via branch probing: generate continuations from each candidate position with and without the reference solution, check correctness via a verifier, and distill only where the solution actually helps.

### Functional-Equivalence Distillation (FED)
Instead of token-level KL, FED groups rollouts into math-strategy classes and applies class-level KL. A frozen reference policy enforces within-class diversity, preventing strategy collapse.

## Baselines (Avg@12, nonthinking, vLLM)

| Model | AIME24 | AIME25 | HMMT25 |
|---|---|---|---|
| Qwen3-1.7B (paper) | 11.9% | 9.2% | 5.0% |
| Qwen3-1.7B (ours) | 12.78% | — | — |

Eval protocol matches the OPSD paper exactly: `temperature=1.0`, `top_p=0.8`, `max_tokens=32768`, `val_n=12`, vLLM with `enforce_eager=True`.

## Setup

```bash
conda create -n opsd python=3.10 && conda activate opsd
pip install torch==2.6.0 transformers accelerate datasets math_verify tqdm pyyaml wandb
pip install vllm==0.8.5 nvidia-nccl-cu12==2.21.5  # 2.21.5 required for CUDA 12.x drivers
```

> **Note on vLLM**: set `VLLM_USE_V1=0` to bypass the V1 engine (requires flashinfer, binary-incompatible with torch 2.6). Set `NCCL_P2P_DISABLE=1` for multi-GPU stability.

## Training Data

Uses `siyanzhao/Openthoughts_math_30k_opsd` — the OPSD paper's training dataset (~29k math olympiad problems with verified solutions from OpenThoughts). Loaded automatically; no manual download needed.

## Running

### Baseline eval

```bash
cd /path/to/opsd-experiments
VLLM_USE_V1=0 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash eval/run_baseline_eval.sh Qwen/Qwen3-1.7B nonthinking 4
```

Results saved to `eval/results/`.

### Training

```bash
# CH-OPD
WANDB_PROJECT=opsd-experiments bash scripts/run_ch_experiment.sh

# FED-OPD
WANDB_PROJECT=opsd-experiments bash scripts/run_fed_experiment.sh

# Both (CH on GPUs 0-3, FED on GPUs 4-7)
bash scripts/run_both_experiments.sh
```

Key hyperparameters (override via CLI or YAML config):

| Param | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen3-1.7B` | Base model |
| `--dataset` | `siyanzhao/Openthoughts_math_30k_opsd` | Training data |
| `--max_steps` | 100 | Training steps |
| `--n_rollouts` | 4 | Rollouts per problem |
| `--beta` | 0.0 | JSD interpolation (0=reverse KL, 1=forward KL) |
| `--lr` | 5e-6 | Learning rate |
| `--max_completion_length` | 1024 | Max generation length during training |

CH-OPD specific: `--n_probe_positions`, `--n_candidates`, `--n_probes`, `--tau_benefit`

FED specific: `--n_anchor_positions`, `--n_continuations_per_anchor`, `--rho`, `--beta_fed`, `--tau_value`, `--lambda_within`

## Structure

```
src/
  opsd_base.py      # Base OPSD trainer, OPSDConfig
  causal_hinge.py   # CH-OPD trainer, CHConfig
  fed.py            # FED trainer, FEDConfig
  data.py           # Dataset loading (training + eval)
scripts/
  train_ch_opsd.py  # CH-OPD training entry point
  train_fed_opsd.py # FED training entry point
  run_ch_experiment.sh
  run_fed_experiment.sh
  run_both_experiments.sh
eval/
  evaluate.py       # vLLM eval (matches OPSD paper protocol)
  run_baseline_eval.sh
  summarize_results.py
```
