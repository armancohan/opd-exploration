# OPSD Style-Bias Analysis Suite — Summary

A set of training-free analyses probing what OPSD's distillation signal actually
teaches. All run on Qwen3-4B (and Qwen3-1.7B for the core decomposition), across
4 datasets, no model training required.

---

## The four analyses

| # | Analysis | Core question | Headline result |
|---|---|---|---|
| 0 | **Style attribution** | Is OPSD's signal answer-content or style? | **95–98% stylistic** across 4 datasets, both model sizes |
| 1 | **Position-resolved** | Where in the rollout does style divergence peak? | **Peaks at tokens 0–9** (0.061 vs ~0.037), the strategy-choice zone |
| 2 | **Cross-problem transfer** | Is the teacher problem-specific or generic? | **88% interchangeable** — a random solution gives 88% of the signal |
| 3 | **Diversity collapse** | Does OPSD push toward one approach? | Reference mismatches model's approach in **60% of problems**; pressure front-loaded (⚠️ diversity count unreliable) |
| 4 | **Reward correlation** | Does style-similarity predict correctness? | *Rerunning* — first pass had 0 reward variance (sparse-reward) |

---

## The consistent story

Every analysis points the same direction: **OPSD's teacher is a style conditioner, not a knowledge conditioner.**

1. The signal is 95–98% stylistic (Analysis 0), robust to dataset and scale.
2. It is **front-loaded** at the exact tokens where the model commits to a reasoning approach (Analysis 1) — `<think>`, "Okay", "different", "can", "possible", "let".
3. It is **88% generic** — swapping in an unrelated problem's solution barely changes it (Analysis 2).
4. The reference's approach **mismatches the model's own preferred approach in 60% of problems** (Analysis 3), and OPSD imposes measurable collapse pressure toward the reference, strongest at early tokens.

Together these say the OPSD teacher-student gap is overwhelmingly a *style mismatch*, localized to the strategy-setting moment, and largely independent of the specific reference content.

---

## Implications for novel OPD methods

**Primary — Progressive Self-Distillation (PSD).** All four analyses motivate replacing the external reference with the model's own verified-correct rollouts:
- removes the foreign-style component (Analyses 0, 2)
- by construction matches the model's dominant approach, eliminating the 60% mismatch (Analysis 3)
- the self-generated teacher opens the solution the same way the student would, killing the front-loaded divergence (Analysis 1)

**Secondary / ablation baselines** (each individually incremental, useful as comparisons against PSD):
- **Position-gated distillation**: skip the first K≈10–20 tokens so the student picks its own strategy (Analysis 1)
- **Content-weighted distillation**: scale loss by per-problem interchangeability ratio — distill more where the reference carries real content (Analysis 2)

---

## Caveats / open items

- **Analysis 3 diversity count** is unreliable (keyword classifier collapsed 118/120 rollouts to "algebraic"). Re-run with an LLM-judge before citing any diversity number. The pressure and reference-mismatch metrics are fine.
- **Analysis 4** needs the rerun to complete with reward variance (longer token budget, easier problems). The first pass had 0/240 correct — itself a data point on how hard the eval set is, but uninformative for correlation.
- All analyses use the *pretrained* model. The style gap may shrink as training proceeds; measuring it across training checkpoints would strengthen the PSD argument (the gap should close faster under PSD than OPSD).

---

## Files

```
analysis/
├── style_attribution.py + _writeup.md     # Analysis 0 (+ qwen3_4b/, math500/, numina_math/, openr1_math/)
├── position_analysis/    REPORT.md, *.png, results.json   # Analysis 1
├── cross_problem/        REPORT.md, *.png, results.json   # Analysis 2
├── diversity_collapse/   REPORT.md, *.png, results.json   # Analysis 3
├── reward_correlation/   REPORT.md, *.png, results.json   # Analysis 4
└── SUMMARY.md            # this file
```
