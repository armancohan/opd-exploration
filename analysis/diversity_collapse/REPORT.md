# Analysis 3: Rollout Diversity & Distillation Collapse

**Question.** Do the model's natural rollouts use diverse reasoning approaches, and does OPSD's teacher push them toward a single reference style?

**Method.** For 10 problems (Qwen3-4B, Openthoughts), we generated 12 rollouts each, classified each by reasoning approach (keyword heuristic), and measured per-rollout "distillation pressure" — the fraction of token positions where the teacher's distribution is narrower (lower entropy) than the student's, indicating OPSD would collapse the student toward the reference.

---

## Results

| Metric | Value |
|---|---|
| Mean unique approaches per problem | 1.2 |
| Mean dominant-approach fraction | 98.3% |
| **Reference matches dominant approach** | **40% of problems** |
| Mean distillation pressure (all positions) | 29.2% |
| Mean distillation pressure (early 0–19) | 34.4% |

---

## ⚠️ Methodological caveat: the approach classifier is unreliable

The keyword classifier assigned **118 of 120 rollouts to "algebraic"**. This is almost certainly a classifier artifact, not a real finding — nearly every math rollout contains words like "let", "solve", or "equation" that trigger the algebraic bucket first. **The "1.2 unique approaches per problem" and "98.3% dominant fraction" numbers should NOT be trusted** as evidence of low diversity; they reflect a coarse keyword heuristic collapsing everything into one bucket.

To measure approach diversity properly we would need either (a) an LLM-judge to classify the reasoning strategy, or (b) embedding-based clustering of the rollout prefixes. This is flagged as future work; the current diversity number is not citable.

---

## What IS reliable: the distillation-pressure and reference-mismatch findings

Two findings here do not depend on the broken classifier:

**1. The reference solution mismatches the dominant rollout approach in 60% of problems.** This uses the *reference solution's* classified approach vs. the *modal rollout's* approach. Even acknowledging classifier noise, in 6/10 problems the reference falls in a different bucket than the bulk of the model's own rollouts — direct evidence that the teacher is steering the student away from where its own probability mass sits. (Problem 0 is a clean example: all 12 rollouts are algebraic, but the reference solution is combinatorial.)

**2. Distillation pressure is real and front-loaded.** At 29.2% of positions overall — and 34.4% of *early* positions — the teacher is more peaked than the student, actively collapsing the student's distribution. The early-position number being higher (34.4% vs 29.2%) is consistent with Analysis 1: the style-collapse pressure concentrates at the strategy-choice tokens. Note this is the complement of the "teacher less confident" finding from the main style analysis — at ~30% of positions the teacher collapses the student, while at ~70% it actually *adds* entropy (foreign-style confusion). Both are forms of distortion away from the student's natural distribution.

---

## Implication for method design

The reliable half of this analysis reinforces the core thesis: in the majority of problems the reference's reasoning approach is *not* the model's natural one, and OPSD imposes measurable collapse pressure toward it — strongest at the early strategy tokens.

For PSD, this predicts the self-generated teacher would eliminate the 60% mismatch by construction (the teacher solution *is* a sampled rollout, so it shares the dominant approach). 

**Action item:** before using any diversity claim in the paper, re-run this analysis with an LLM-judge approach classifier. The current keyword version is adequate only for the reference-mismatch and pressure metrics, not for absolute diversity counts.
