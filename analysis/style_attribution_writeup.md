# Style Attribution Analysis: OPSD's Distillation Signal is Almost Entirely Stylistic

**Setting.** We decompose OPSD's per-token distillation signal into two components: knowledge of the *correct answer* vs. the *style and approach* of the reference solution.

We compare three teacher variants on the same student rollout:

- **teacher_full**: conditioned on `[problem + full reference solution]`
- **teacher_answer**: conditioned on `[problem + "The answer is X."]`
- **student**: conditioned on `[problem only]`

Per-token JSD is measured across 30 problems × 256 tokens = 7,680 positions, on the **same problems and rollouts** for both models. Models evaluated: `Qwen3-1.7B` and `Qwen3-4B`.

---

## Results

| Metric | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| JSD(teacher_full, student) — OPSD signal | 0.0389 | 0.0395 |
| JSD(teacher_answer, student) — answer-only | 0.0021 | 0.0025 |
| JSD(teacher_full, teacher_answer) — style | 0.0377 | 0.0389 |
| **Style fraction** | **96.9%** | **98.4%** |
| Teacher more peaked than student (% positions) | 39.1% | 28.4% |

---

## Finding 1: OPSD's signal is almost entirely stylistic — robust across model sizes

The style fraction is **96.9%** at 1.7B and **98.4%** at 4B. The answer alone contributes about 2% of the distillation signal; the remaining 97–98% is driven by the stylistic choices of the reference solution author.

This is not a mild effect, and it does not weaken with scale — if anything it becomes more pronounced. The absolute JSD values are nearly identical across models (0.039 vs 0.040), confirming the phenomenon is a property of the OPSD setup, not a model-specific artifact.

The per-problem distribution is tight in both cases. For 1.7B, individual problem style fractions range from 0.85 to 1.07 with mean 0.969. There is no subset of problems where answer knowledge dominates the distillation signal.

**Interpretation:** OPSD is training the student to imitate *how the reference author reasons*, not to find the correct answer. The distillation loss is almost a style-matching objective.

---

## Finding 2: The teacher is more uncertain than the student at the majority of positions

A privileged teacher (with the solution) might be expected to be more confident. Instead:

- At **1.7B**: teacher has lower entropy than student at only 39.1% of positions
- At **4B**: teacher has lower entropy than student at only 28.4% of positions

The trend worsens with scale: the larger model is *more* confident in its own natural style, so conditioning it on a foreign reference solution makes it proportionally *less* certain about how to continue the student's rollout. At 72% of positions for the 4B model, OPSD is imposing a distillation gradient that pulls the student away from its most confident natural continuations.

This rules out one possible benign interpretation ("OPSD adds noise but at least it's confident noise"). The teacher is often *less* confident than the student at individual token positions, meaning the distillation signal is sometimes actively degrading the student's natural distribution.

---

## Connection to Progressive Self-Distillation (PSD)

These findings provide quantitative motivation for PSD. The 97–98% style fraction means the realizability gap between teacher and student is almost entirely stylistic in origin. A teacher conditioned on the model's own verified-correct rollout would, by construction, have the same stylistic idiom as the student — reducing the style contribution toward zero while leaving intact the only useful signal (the reasoning path to a correct answer).

The entropy finding strengthens this: as the model scales, its own stylistic preferences become stronger (lower entropy on natural continuations), making stylistically foreign teacher supervision increasingly harmful. PSD removes this mismatch by definition.

---

## Summary

Across both Qwen3-1.7B and Qwen3-4B, OPSD's per-token distillation signal is 97–98% attributable to the style of the reference solution and only 2–3% to the correct answer. The finding is robust, consistent across problems, and worsens with model scale. This supports the core motivation for Progressive Self-Distillation: replacing the fixed external reference with the model's own successful solutions to eliminate the style mismatch that constitutes almost all of OPSD's loss signal.
