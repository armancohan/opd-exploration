# Validating the style-fraction metric

`style_fraction = JSD(full, answer) / JSD(full, student)`

Setup: Qwen3-1.7B, 30 Openthoughts problems, 7,680 student-rollout token positions,
one seeded rollout per problem (`--seed 1234`, temperature 0.8, 256 new tokens).
Per-token JSD in nats, chunked over the 151K vocab, evaluated on the student's own
sampled tokens. Script: `analysis/style_attribution_validate.py`.

## JSD computation (what is actually measured)

At every completion position *t* we take the model's next-token distribution under
three (now four) conditionings and compute the **token-level** Jensen–Shannon
divergence, then average over all tokens and problems:

```
JSD(P||Q) = ½ KL(P||M) + ½ KL(Q||M),   M = ½(P+Q),   in nats
```

- It is **token-level**, not sequence-level: we never compare full-sequence
  likelihoods, only per-position next-token distributions on the *same* token
  prefix. This is exactly the quantity OPSD's loss operates on.
- All three conditionings score the **same** student rollout, so any divergence is
  attributable to the conditioning context, not to different text.
- Computed in float32, chunked (64 tokens × 151,936 vocab) to avoid OOM.

## The three checks

| quantity | value (nats) | share of total |
|---|---|---|
| `JSD(full, student)`  — total OPSD signal     | 0.0398 | 100% |
| `JSD(answer, student)` — weak answer-only      | 0.0024 | 6.1% |
| `JSD(minimal, student)` — strong answer-only   | 0.0034 | 8.5% |
| `JSD(full, answer)`  — style numerator (weak)  | 0.0390 | **98.0%** |
| `JSD(full, minimal)` — style numerator (strong)| 0.0375 | **94.4%** |

**1. Direct report + is the baseline a no-op?**
`JSD(answer, student) = 0.0024` nats — small, but strictly **positive**. The
"answer" prompt genuinely moves the distribution; it is a *weak* conditioner, not a
*broken* one. So the high style fraction is not an artifact of a degenerate baseline
that does nothing.

**2. Stronger conditioner (does weak conditioning inflate the number?).**
We added a baseline that is *structurally identical* to the full-solution prompt —
same `"Here is a reference solution:"` framing — but whose body is only the boxed
answer (`"The final answer is \boxed{X}."`), no reasoning. Conditioning the model
this much more strongly on the outcome lifts the answer's share from 6.1% to only
8.5%; the **style fraction stays at 94.4%** (vs 98.0% with the weak prompt). The
finding survives — it is not an artifact of an under-powered baseline.

**3. JSD is not a metric — does the "fraction" framing distort it?**
`sqrt(JSD)` is a true metric (Jensen–Shannon distance); raw JSD is not. Recomputing
the fraction in metric space gives **96.1–98.5%**, matching the raw-JSD 94.4–98.0%.
And the three divergences are nearly additive at the mean level — the gap
`JSD(full,answer)+JSD(answer,student) − JSD(full,student)` is only **+4.1%** of the
total (weak) / +2.9% (strong). So the three conditionings sit nearly colinear with
the answer baseline *between* student and full: the ratio behaves like a real
decomposition here, not just a ratio of unrelated distances.

## Concrete token-level examples

The divergence sits on **stylistic / discourse tokens**, not on answer content.
(JSD in nats; `␣` = leading space.)

**Ex. A — discourse transition** (problem: chore-assignment counting, answer 540).
After finishing a sub-result the student opens the next sentence; the answer barely
matters but the reference's style does.

```
context: "...ways to assign these six tasks to the three boys, with each
          boy getting at least one task."
student next-token top-5 : This / Wait / Hmm / But / I
answer  next-token top-5 : This / Wait / Hmm / But / At      (≈ student)
full    next-token top-5 : The / Now / Hmm / Let / Alright   (≠ student)
JSD(full,student)=0.685   JSD(answer,student)=0.038   JSD(full,answer)=0.685
```

**Ex. B — hedge vs. assert** (problem: power-series identity, answer √e).
The student hedges; the reference is assertive and structured.

```
context: "...x⁶/(2²·4²·6²) + ...  I need to figure out what these series represent."
student next-token top-5 : Maybe / They / It / Perhaps / These
answer  next-token top-5 : Maybe / They / It / Perhaps / These   (= student)
full    next-token top-5 : The / Then / \n\n / Let / From         (≠ student)
JSD(full,student)=0.693   JSD(answer,student)=0.001   JSD(full,answer)=0.693
```

**Ex. C — an honest counterexample where the answer DOES help** (problem:
derangements of 6, answer 265). Here knowing the number nudges the model
(JSD(answer,student)=0.114, far above the ~0.001 typical), yet the reference's style
still dominates (0.334 > 0.114).

```
context: "...derangements! So, the problem is asking for the number of
          derangements of 6 elements."
student next-token top-5 : But / I / The / Now / Wait
answer  next-token top-5 : But / The / I / Now / Wait
full    next-token top-5 : The / Now / But / In / Looking
JSD(full,student)=0.673   JSD(answer,student)=0.114   JSD(full,answer)=0.334
```

## Bottom line

The style fraction is **94–98%** under every stress test — direct measurement,
a structurally-matched strong conditioner, and a metric-space recomputation. Knowing
the correct answer explains only ~6–9% of OPSD's per-token signal; the rest is the
reference solution's reasoning style (discourse markers, assertiveness, phrasing,
notation), as the token-level examples make concrete.
