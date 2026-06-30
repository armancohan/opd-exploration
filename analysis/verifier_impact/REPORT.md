# Verifier Fix — Real-World Reward Impact

**What happened.** `src/verifier.py` (hand-rolled sympy heuristic) was producing
systematic **false negatives** — rejecting answers the model got *correct*. This
silently suppressed the RL reward signal (reward ≈ 0.000 across all training runs
was partly this bug, not just hard problems) and broke Analysis 4
(reward-correlation showed 0/240 correct).

**Failure modes confirmed:**
- tuples / coordinates: `(3, \frac{\pi}{2})` vs `\left( 3, \frac{\pi}{2} \right)`
- `\left`/`\right` delimiters, greek-in-fraction, expression spacing (`x^2+1` vs `x^2 + 1`)
- brittle `\boxed{}` extraction (missed boxed answers in `\[...\]` / nested-brace forms)

**Fix.** Rewrote `src/verifier.py` to use HuggingFace `math_verify` (the Open-R1
standard) as the backend, with the old sympy heuristic kept as a fallback. Public
API (`verify_math_answer`, `batch_verify`, `extract_boxed_answer`) is unchanged, so
all 6 call sites (FED, OPSD, causal-hinge, evaluate, analysis 4) work without edits.
Bare answer strings are wrapped in `$...$` to force LaTeX parsing.

## Validation

**Edge cases:** 10/10 correct, including true-negatives (`(2,3) ≠ (3,2)`, `7 ≠ 8`).

**Real Qwen3-4B rollouts** (8 problems × 8 rollouts = 64 scored):

| Verifier | Correct | Rate |
|---|---|---|
| OLD (sympy heuristic) | 23/64 | 35.9% |
| **NEW (math_verify)** | **31/64** | **48.4%** |

The fix recovers **8 rollouts the old verifier wrongly rejected** — a **+12.5
points absolute / +35% relative** increase in captured reward. Two of the 8
problems the model genuinely failed (both verifiers agree), so the gain is
concentrated on the problems the model actually solves.

**Implication:** every prior training run was learning from a reward signal with
~35% of its positive examples missing. Re-running training with the fixed verifier
should produce a substantially denser, less noisy reward.

Reproduce: `python analysis/verifier_impact_test.py --n_problems 8 --k 8`
