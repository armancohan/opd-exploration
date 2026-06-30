# Analysis 1: Where Does OPSD's Style Divergence Concentrate?

**Question.** Is OPSD's style signal spread uniformly across the rollout, or concentrated at specific positions? In particular, is it front-loaded at the early "strategy-choice" tokens where the model decides *how* to approach the problem?

**Method.** For 30 problems (Qwen3-4B, Openthoughts), we computed per-token JSD between teacher_full, teacher_answer, and student, then bucketed by token position. We also identified the highest-divergence tokens within the first 20 positions.

---

## Result: Divergence peaks at the very first tokens

| Position bucket | OPSD signal (full vs student) | answer-only | style | style % |
|---|---|---|---|---|
| **0–9** | **0.0614** | 0.0007 | 0.0600 | 97.9% |
| 10–19 | 0.0331 | 0.0036 | 0.0287 | 86.6% |
| 20–49 | 0.0381 | 0.0043 | 0.0375 | 98.3% |
| 50–149 | 0.0417 | 0.0030 | 0.0412 | 99.0% |
| 150+ | 0.0337 | 0.0016 | 0.0326 | 96.7% |

**The first 10 tokens carry the highest divergence — ~0.061 nats, roughly 50–80% higher than any later bucket.** This is exactly the strategy-setting zone: the tokens immediately after `<think>` where the model commits to an approach (which variable to introduce, whether to set up coordinates, which lemma to invoke).

The answer-only signal is *lowest* in this same bucket (0.0007), meaning the early-position divergence is almost purely stylistic (97.9%). Knowing the correct answer tells the teacher essentially nothing about how to open the solution; the reference solution's stylistic opening drives the entire signal.

---

## The highest-divergence early tokens are strategy words

Top tokens by mean JSD in the first 20 positions:

| Token | Mean JSD | Interpretation |
|---|---|---|
| ` different` | 0.271 | branching ("a different approach / different cases") |
| ` can` | 0.210 | option framing ("we can use / can be written") |
| ` possible` | 0.189 | enumerating possibilities |
| ` from` | 0.173 | derivation source ("from the equation / from this") |
| `<think>` | 0.162 | the reasoning-mode entry token itself |
| `Okay` | 0.161 | discourse opener |
| ` equal`, ` where`, ` find` | 0.066–0.076 | setup verbs |
| ` let` | 0.046 | variable introduction |

These are precisely the tokens that fork the solution path. The teacher, conditioned on the reference solution, "knows" which fork the reference took and pulls the student toward it — at the exact moment the student is making its own strategic choice.

---

## Implication for method design

This is the strongest actionable signal of the four analyses. It suggests a concrete intervention:

**Position-gated distillation.** Apply OPSD's distillation loss only *after* the approach is committed (e.g., skip the first K≈10–20 tokens), so the student chooses its own strategy and is distilled only on the execution of that strategy. This directly preserves approach diversity (Analysis 3's concern) while keeping distillation pressure on the mechanical steps where the reference is genuinely useful.

This also strengthens the PSD motivation: a self-generated teacher would have taken the *student's own* approach, so the front-loaded divergence would vanish — the teacher and student would open the solution the same way.

**Caveat.** Position-gating alone is a token-weighting variant (the class the project already judged incremental). Its value here is diagnostic — it localizes *where* the style damage happens — and as a cheap ablation baseline against PSD, not as a standalone contribution.
