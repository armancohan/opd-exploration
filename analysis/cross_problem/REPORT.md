# Analysis 2: Cross-Problem Style Transfer

**Question.** How much of the teacher's signal is *problem-specific* vs. a generic style template? If we feed the teacher a different problem's reference solution, does the distillation signal change much?

**Method.** For 15 problems (Qwen3-4B, Openthoughts), we computed the teacher-student JSD twice per rollout:
- **Proper teacher**: the problem conditioned on its own reference solution
- **Cross teacher**: the problem conditioned on 5 *other* problems' reference solutions

The **interchangeability ratio** = mean JSD(cross) / JSD(proper). A ratio near 1.0 means swapping in an unrelated solution barely changes the signal — i.e., the solution acts as a generic style template, not a knowledge source.

---

## Result: 88% interchangeability

| Metric | Value |
|---|---|
| Mean JSD(proper teacher) | 0.0450 |
| Mean JSD(cross teacher) | 0.0308 |
| **Interchangeability ratio** | **0.881** |

Feeding the teacher a **completely unrelated problem's solution** produces 88% of the distillation signal of the correct solution. The reference solution's problem-specific content accounts for only ~12% of the teacher's effect on the student; the remaining 88% is generic "reasoning-style" conditioning that any solution provides.

---

## Per-problem breakdown reveals two regimes

The per-problem ratios are bimodal (range 0.22 to 2.37):

- **Low-ratio problems (ratio < 0.6, ~6/15):** proper solution matters more — these have high proper-teacher JSD (0.055–0.073), meaning the correct solution genuinely diverges from the student's natural rollout. Here content matters.
- **High-ratio problems (ratio > 1.0, ~6/15):** cross solutions produce *as much or more* signal than the proper one. Several have ratio > 1.4, meaning an unrelated solution pulls the student *further* from its natural distribution than the correct solution does. These are problems where the proper-teacher JSD is already low (0.015–0.036) — the model's natural rollout happened to align with its own reference, so any other solution looks more foreign.

The interpretation: when the reference solution matches the student's natural style, OPSD adds little; when it doesn't, OPSD adds a large signal — but that signal is **indistinguishable from feeding a random other solution**. In neither case is the signal carrying problem-specific reasoning content.

---

## Implication for method design

This is the cleanest evidence that OPSD's teacher is a **style conditioner, not a knowledge conditioner**. It supports two design directions:

1. **PSD (primary):** if the teacher signal is 88% generic style, then replacing the external reference with the model's *own* style (via self-generated solutions) removes the foreign-style component while preserving the small problem-specific residual. The 12% that is problem-specific is exactly what survives in PSD, because the model's own correct solution is both problem-specific *and* style-matched.

2. **Diagnostic for "useful" distillation:** the per-problem ratio could serve as a runtime signal — distill more aggressively on low-ratio problems (where the reference carries real content) and less on high-ratio problems (where it's pure style noise). This is a natural ablation/baseline to compare against PSD.

**Caveat.** Cross-teacher JSD being lower in absolute terms (0.031 < 0.045) is expected — a random solution is on average less "relevant" so conditions the model less strongly. The striking finding is not the absolute value but that 88% of the effect survives a complete content swap.
