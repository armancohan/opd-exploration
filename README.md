# OPD Exploration

Implementations of **Causal-Hinge OPD** (CH-OPD) and **Functional-Equivalence Distillation** (FED), two efficiency-motivated extensions of On-Policy Self-Distillation (OPSD) for math reasoning in LLMs.

## Background: On-Policy Self-Distillation (OPSD)

Standard knowledge distillation transfers knowledge from a larger teacher model to a smaller student. OPSD removes the requirement for a separate teacher: the same model plays both roles, with the teacher receiving a privileged hint — the reference solution — that the student does not see.

**How it works:**
- *Student* prompt: `[problem]` → generate a completion and learn from it
- *Teacher* prompt: `[problem] + [reference solution]` → same weights, richer context
- Loss: token-level generalized JSD between teacher and student output distributions at each position of the student's completion
- The student learns to approximate the distribution the teacher would produce if it had seen the solution

**Why on-policy matters:** Because the student generates its own rollouts at each step, the training distribution always matches the model's current behavior. This avoids the distribution shift that plagues offline distillation (training on fixed teacher outputs that become stale as the student improves).

**Key parameter β:** Controls the KL direction — `β=0` is reverse KL (student mimics teacher's mass), `β=1` is forward KL (teacher mimics student's support), `β=0.5` is symmetric JSD.

---

## Gap 1: Uniform distillation ignores where the hint actually helps → CH-OPD

OPSD distills at *every* token position in the completion, regardless of whether the reference solution is actually changing what the model would do there. This is wasteful and noisy:

- At positions where the student already knows what to do, teacher and student distributions are nearly identical. Distilling there adds training noise without signal.
- At positions where the student is genuinely uncertain or about to go wrong, the solution steers the teacher toward a better path — and distilling *there* is what actually improves the student.

**Causal-Hinge OPD** introduces a benefit signal B_t at each position that measures whether the reference solution causally changes the outcome. Distillation is applied only where B_t > τ ("hinge positions").

*Original benefit estimation (full rollouts mode):* For each candidate position, generate short continuations with and without the solution hint, verify correctness, and estimate B_t as the solution-induced improvement in expected reward. This is principled but expensive — it requires 10–20 extra forward passes per training step.

*Efficient benefit estimation (default):* Replace probe sampling with the token-level KL divergence KL(p_teacher ‖ p_student) at each position. This is a zero-cost proxy computed directly from the logits already produced during the main forward pass. Positions where the teacher's distribution diverges most from the student are exactly where the solution is changing the model's reasoning. This eliminates all extra generation, reducing step time to match base OPSD, while preserving the key hinge-selection idea. The approach is related to but distinct from SelecTKD (Huang et al., 2025), which gates on teacher-side entropy rather than teacher–student divergence.

---

## Gap 2: Token-level KL collapses strategy diversity → FED

OPSD's token-level KL treats all reasoning paths as interchangeable: it pushes every completion token toward the teacher's distribution, regardless of which solution strategy the completion represents. A problem in math often has multiple valid approaches (algebraic manipulation, geometric insight, number-theoretic argument), and a good policy should maintain probability mass across all of them.

In practice, if the teacher consistently uses one strategy type (because the reference solution reflects that strategy), OPSD's token-level loss progressively collapses the student's output distribution toward that single strategy. This reduces pass@k and makes the model brittle.

**Functional-Equivalence Distillation** addresses this by operating at the level of strategy classes rather than individual tokens:

1. *Strategy classification:* Each rollout completion is assigned to a math strategy class (algebra, geometry, number theory, combinatorics, sequences, trigonometry) by keyword matching on the solution text.
2. *Class-level KL:* Instead of distilling at every token, compute a target distribution Q(class) that up-weights high-reward strategy classes and down-weights low-reward ones, then push the student's class-level distribution toward Q.
3. *Within-class diversity:* A frozen reference policy (the model's initial weights at the start of training) penalizes the student for moving too far from the initial distribution *within* each high-value strategy class. This prevents the student from collapsing all probability within a class onto a single token path.

*Original strategy discovery (full rollouts mode):* For each anchor position (selected by KL), sample `n_continuations_per_anchor` completions from scratch. This directly probes the strategy space from that point but requires 16–32 extra generations per step — 4–5× more generation than base OPSD.

*Efficient strategy discovery (default):* Reuse the `n_rollouts` completions already generated during the standard OPSD rollout phase. These completions, which differ because of stochastic decoding, naturally represent different strategies the model currently considers. Their tokens at each anchor position serve as candidate actions; their completion texts determine strategy class; their verifier rewards give V_hat. This eliminates all extra anchor generation, reducing FED's step cost to match base OPSD.

---

## Baselines (Avg@12, nonthinking, vLLM)

| Model | AIME24 | AIME25 | HMMT25 |
|---|---|---|---|
| Qwen3-1.7B (OPSD paper) | 11.9% | 9.2% | 5.0% |
| Qwen3-1.7B (ours, reproduced) | 12.78% | 8.89% | 5.00% |

Eval protocol matches the OPSD paper exactly: `temperature=1.0`, `top_p=0.8`, `max_tokens=32768`, `val_n=12`, vLLM with `enforce_eager=True`, tensor parallel size 4.

---

## Setup

```bash
conda create -n opsd python=3.10 && conda activate opsd
pip install torch==2.6.0 transformers accelerate datasets math_verify tqdm pyyaml wandb
pip install vllm==0.8.5 nvidia-nccl-cu12==2.21.5  # 2.21.5 required for CUDA 12.x drivers
```

> **Note on vLLM**: set `VLLM_USE_V1=0` to bypass the V1 engine (requires flashinfer, binary-incompatible with torch 2.6). Set `NCCL_P2P_DISABLE=1` for multi-GPU stability.

## Training Data

Uses `siyanzhao/Openthoughts_math_30k_opsd` — the OPSD paper's training dataset (~29k math olympiad problems with verified solutions). Loaded automatically; no manual download needed.

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
# CH-OPD (efficient logit-benefit mode by default)
WANDB_PROJECT=opsd-experiments bash scripts/run_ch_experiment.sh

# FED (efficient rollout-strategy mode by default)
WANDB_PROJECT=opsd-experiments bash scripts/run_fed_experiment.sh

# Both (CH on GPUs 0-3, FED on GPUs 4-7)
bash scripts/run_both_experiments.sh
```

To use the original (more expensive) methods for ablation:
```bash
python scripts/train_ch_opsd.py --no-use_logit_benefit   # probe sampling
python scripts/train_fed_opsd.py --no-use_rollout_strategies  # anchor sampling
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
| `--tau_benefit` | 0.0 | CH-OPD hinge threshold (higher = more selective) |

CH-OPD specific: `--n_probe_positions`, `--n_candidates`, `--n_probes`, `--tau_benefit`, `--use_logit_benefit`

FED specific: `--n_anchor_positions`, `--n_continuations_per_anchor`, `--rho`, `--beta_fed`, `--tau_value`, `--lambda_within`, `--use_rollout_strategies`

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
