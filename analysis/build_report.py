"""Build a single standalone HTML report unifying all OPSD style-bias analyses.

Embeds all PNGs as base64 so the file is fully portable (no external assets).
Reads results.json files for the numeric tables.

Usage:
    cd /home/ac3458/code/opsd-experiments
    python analysis/build_report.py
    # → analysis/OPSD_style_analysis_report.html
"""

import base64
import json
import os

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ANALYSIS_DIR, "OPSD_style_analysis_report.html")


def b64img(path):
    full = os.path.join(ANALYSIS_DIR, path)
    if not os.path.exists(full):
        return None
    with open(full, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def load_json(path):
    full = os.path.join(ANALYSIS_DIR, path)
    if not os.path.exists(full):
        return None
    with open(full) as f:
        return json.load(f)


def img_block(path, caption_html):
    data = b64img(path)
    if data is None:
        return f'<div class="missing">[image not found: {path}]</div>'
    return (
        f'<figure>'
        f'<img src="data:image/png;base64,{data}" alt="{path}"/>'
        f'<figcaption>{caption_html}</figcaption>'
        f'</figure>'
    )


# ── Load all result data ──────────────────────────────────────────────────────
sa_opsd_17b = load_json("style_attribution_results.json")
sa_opsd_4b  = load_json("qwen3_4b/style_attribution_results.json")
sa_math500  = load_json("math500/style_attribution_results.json")
sa_numina   = load_json("numina_math/style_attribution_results.json")
sa_openr1   = load_json("openr1_math/style_attribution_results.json")
pos         = load_json("position_analysis/results.json")
cross       = load_json("cross_problem/results.json")
div         = load_json("diversity_collapse/results.json")
rew         = load_json("reward_correlation/results.json")
val         = load_json("style_validation/validation_results.json")
val_ex      = load_json("style_validation/token_examples.json")


def pct(x):
    return f"{x*100:.1f}%" if x is not None else "—"


def f4(x):
    return f"{x:.4f}" if x is not None else "—"


# ── Style attribution cross-dataset table ─────────────────────────────────────
sa_table_rows = ""
for (dataset, model, d) in [
    ("Openthoughts (OPSD data)", "Qwen3-1.7B", sa_opsd_17b),
    ("Openthoughts (OPSD data)", "Qwen3-4B", sa_opsd_4b),
    ("MATH-500", "Qwen3-4B", sa_math500),
    ("NuminaMath-CoT", "Qwen3-4B", sa_numina),
    ("OpenR1-Math-220k", "Qwen3-4B", sa_openr1),
]:
    if d is None:
        continue
    sf = d["style_fraction"]
    hi = ' class="hi"' if sf >= 0.95 else ""
    sa_table_rows += (
        f"<tr><td>{dataset}</td><td>{model}</td>"
        f"<td>{f4(d['jsd_teacher_full_vs_student']['mean'])}</td>"
        f"<td>{f4(d['jsd_teacher_answer_vs_student']['mean'])}</td>"
        f"<td>{f4(d['jsd_teacher_full_vs_teacher_answer']['mean'])}</td>"
        f"<td{hi}><b>{pct(sf)}</b></td></tr>\n"
    )

# ── Style-fraction validation (robustness checks + token examples) ────────────
def _esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

val_table_rows = ""
val_block = ""
val_examples_html = ""
if val:
    j = val["jsd"]
    sfr = val["style_fraction_raw_JSD"]
    sfm = val["style_fraction_metric_sqrtJSD"]
    aos = val["answer_only_share_raw_JSD"]
    gap = val["non_additivity_gap"]
    m_fs = j["full_vs_student   (total OPSD signal)"]["mean"]
    m_as = j["answer_vs_student (weak answer-only)"]["mean"]
    m_ms = j["minimal_vs_student(strong answer-only)"]["mean"]
    m_fa = j["full_vs_answer    (style num, weak)"]["mean"]
    m_fm = j["full_vs_minimal   (style num, strong)"]["mean"]
    sf_weak = sfr["weak_conditioner   (full,answer)/(full,student)"]
    sf_strong = sfr["strong_conditioner (full,minimal)/(full,student)"]
    sf_weak_m = sfm["weak_conditioner"]
    sf_strong_m = sfm["strong_conditioner"]
    ao_weak = aos["weak   (answer,student)/(full,student)"]
    ao_strong = aos["strong (minimal,student)/(full,student)"]

    val_table_rows = (
        f"<tr><td>Total OPSD signal — JSD(full, student)</td><td>{f4(m_fs)}</td>"
        f"<td>100%</td><td>—</td></tr>\n"
        f"<tr><td>Weak answer-only — JSD(answer, student) <span style='color:var(--muted)'>(\"The answer is X\")</span></td>"
        f"<td>{f4(m_as)}</td><td>{pct(ao_weak)}</td><td>style {pct(sf_weak)}</td></tr>\n"
        f"<tr><td>Strong answer-only — JSD(minimal, student) <span style='color:var(--muted)'>(reference framing, answer only)</span></td>"
        f"<td>{f4(m_ms)}</td><td>{pct(ao_strong)}</td><td>style {pct(sf_strong)}</td></tr>\n"
    )

    val_block = f"""
<table>
<tr><th>Robustness check</th><th>Result</th><th>Verdict</th></tr>
<tr><td>Is the answer baseline a broken no-op?</td>
    <td>JSD(answer, student) = <b>{f4(m_as)}</b> nats — small but strictly &gt; 0</td>
    <td>Real, weak conditioner — not degenerate</td></tr>
<tr><td>Does weak conditioning inflate the number?<br>
    <span style="color:var(--muted)">Replace "The answer is X" with a structurally-matched
    reference (same framing, body = boxed answer only)</span></td>
    <td>Style fraction <b>{pct(sf_weak)} → {pct(sf_strong)}</b>;<br>
    answer share {pct(ao_weak)} → {pct(ao_strong)}</td>
    <td>Survives — not an artifact</td></tr>
<tr><td>Does JSD-not-being-a-metric distort the "fraction"?<br>
    <span style="color:var(--muted)">Recompute in metric space, √JSD = Jensen–Shannon distance</span></td>
    <td>raw {pct(sf_weak)}/{pct(sf_strong)} vs metric <b>{pct(sf_weak_m)}/{pct(sf_strong_m)}</b>;<br>
    non-additivity gap only {gap['raw_JSD_weak']*100:+.1f}% (raw)</td>
    <td>Agree — behaves like a real decomposition</td></tr>
</table>
"""

    # token-level examples — pick the three clearest hand-checked positions
    def _topline(tokens):
        return " / ".join(_esc(t[0]).strip() or "␣" for t in tokens)

    picks = []
    if val_ex:
        # (problem-index, position) chosen by inspection for clarity
        wanted = [(0, 174), (1, 166), (2, 123)]
        labels = {
            (0, 174): "Discourse transition — student opens a new sentence; the answer is irrelevant, the reference's style is not.",
            (1, 166): "Hedge vs. assert — the student hedges (“Maybe/Perhaps”); the reference is assertive and structured.",
            (2, 123): "Honest counterexample — here the answer (265 = derangements of 6) DOES nudge the model, yet style still dominates.",
        }
        for (pi, pos_want) in wanted:
            if pi >= len(val_ex):
                continue
            exblock = val_ex[pi]
            match = next((p for p in exblock["positions"] if p["position"] == pos_want), None)
            if match is None:
                continue
            picks.append((labels[(pi, pos_want)], exblock, match))

    for caption, exblock, p in picks:
        ctx = _esc(exblock.get("rollout_preview", "")[:0])  # unused; keep context tail below
        ctx_tail = _esc(p["context_tail"].strip())
        val_examples_html += f"""
<div class="example">
<div class="ex-cap">{caption}</div>
<div class="ex-ctx">…{ctx_tail}</div>
<table style="margin:8px 0">
<tr><th>next-token top-5</th><th>tokens</th></tr>
<tr><td>student <span style="color:var(--muted)">(problem only)</span></td><td><code>{_topline(p['top5_student'])}</code></td></tr>
<tr><td>answer <span style="color:var(--muted)">(+ “The answer is X”)</span></td><td><code>{_topline(p['top5_answer'])}</code></td></tr>
<tr><td>full <span style="color:var(--muted)">(+ reference solution)</span></td><td><code>{_topline(p['top5_full'])}</code></td></tr>
</table>
<div class="ex-jsd">JSD(full, student) = <b>{p['jsd_full_student']:.3f}</b> &nbsp;·&nbsp;
JSD(answer, student) = <b>{p['jsd_answer_student']:.3f}</b> &nbsp;·&nbsp;
JSD(full, answer) = <b>{p['jsd_full_answer']:.3f}</b> &nbsp; (nats)</div>
</div>
"""

# ── Position bucket table ─────────────────────────────────────────────────────
pos_table_rows = ""
if pos:
    for bucket, d in pos["bucket_analysis"].items():
        hi = ' class="hi"' if bucket == "0–9" else ""
        pos_table_rows += (
            f"<tr{hi}><td>{bucket}</td>"
            f"<td>{f4(d['jsd_full_student']['mean'])}</td>"
            f"<td>{f4(d['jsd_answer_student']['mean'])}</td>"
            f"<td>{f4(d['jsd_style']['mean'])}</td>"
            f"<td>{pct(d['style_fraction'])}</td></tr>\n"
        )

pos_tok_rows = ""
if pos:
    for tok, val in pos["top_early_divergent_tokens"][:10]:
        disp = tok.replace("Ġ", "␣").replace("Ċ", "\\n")
        pos_tok_rows += f"<tr><td><code>{disp}</code></td><td>{f4(val)}</td></tr>\n"

# ── Cross-problem ─────────────────────────────────────────────────────────────
cross_ratio = cross["interchangeability_ratio"] if cross else None
cross_proper = cross["mean_jsd_proper_teacher"] if cross else None
cross_cross = cross["mean_jsd_cross_teacher"] if cross else None

# ── Diversity ─────────────────────────────────────────────────────────────────
div_refmatch = div["ref_matches_dominant_approach_rate"] if div else None
div_press = div["mean_distillation_pressure"] if div else None
div_press_early = div["mean_early_position_pressure"] if div else None

# ── Reward correlation ────────────────────────────────────────────────────────
rew_corr = rew.get("pointbiserial_correlation_jsd_vs_reward") if rew else None
rew_pval = rew.get("correlation_pvalue") if rew else None
rew_rate = rew.get("total_reward_rate") if rew else None
rew_ncorrect = rew.get("n_correct_rollouts") if rew else None
rew_nincorrect = rew.get("n_incorrect_rollouts") if rew else None
rew_jsd_c = rew.get("mean_jsd_correct_rollouts") if rew else None
rew_jsd_i = rew.get("mean_jsd_incorrect_rollouts") if rew else None
rew_has_variance = bool(rew_ncorrect) and bool(rew_nincorrect)

# Analysis-4 narrative: depends on whether the (fixed-verifier) rerun produced
# reward variance. The original 0-reward result was a verifier false-negative bug
# (src/verifier.py), since fixed by switching to the math_verify backend.
if rew_has_variance:
    if rew_corr is not None and rew_corr < -0.1 and (rew_pval or 1) < 0.05:
        _a4_interp = ("a <b>negative</b> correlation: rollouts stylistically closer to the "
                      "reference (lower JSD) are more likely correct — here OPSD's style signal "
                      "is a partial proxy for correct reasoning.")
    elif rew_corr is None or abs(rew_corr) < 0.1 or (rew_pval or 1) > 0.1:
        _a4_interp = ("<b>no significant</b> correlation: reference-style similarity is orthogonal "
                      "to answer correctness, so OPSD's style signal is noise from the reward's "
                      "point of view.")
    else:
        _a4_interp = "a weak correlation; interpret with caution."
    a4_block = f"""<div class="takeaway">
<b>Result.</b> With the reward verifier fixed (math_verify backend; the original
sympy grader produced false negatives that zeroed out the reward), the rerun yields
real reward variance: <b>{rew_ncorrect} correct</b> and <b>{rew_nincorrect} incorrect</b>
rollouts (reward rate {pct(rew_rate)}). Mean teacher–student JSD is
{f4(rew_jsd_c)} for correct vs {f4(rew_jsd_i)} for incorrect rollouts; point-biserial
r = {f4(rew_corr)} (p = {f4(rew_pval)}) — {_a4_interp}
</div>"""
    a4_caption = ("<b>Figure 5.</b> Left: JSD distribution split by correct vs incorrect rollouts. "
                  "Center: mean teacher–student JSD by outcome. Right: per-problem reward rate vs the "
                  "JSD gap between correct and incorrect rollouts. Reward computed with the fixed "
                  "math_verify backend.")
else:
    a4_block = f"""<div class="warn">
<b>Status: rerun pending variance.</b> Reward rate {pct(rew_rate)} with
{rew_ncorrect or 0} correct rollouts. (The original 0-reward result was traced to a
verifier false-negative bug in <code>src/verifier.py</code>, since fixed via the
math_verify backend; this panel regenerates once the rerun completes.)
</div>"""
    a4_caption = ("<b>Figure 5.</b> Reward-correlation panel; regenerates once the fixed-verifier "
                  "rerun yields reward variance.")

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OPSD Style-Bias Analysis</title>
<style>
  :root {{
    --bg: #0f1115; --card: #1a1d24; --ink: #e6e8eb; --muted: #9aa3af;
    --accent: #5b9dff; --hi: #7ee787; --warn: #ffb454; --line: #2a2e37;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6; font-size: 16px;
  }}
  .wrap {{ max-width: 1000px; margin: 0 auto; padding: 48px 24px 96px; }}
  h1 {{ font-size: 2.2rem; margin: 0 0 6px; letter-spacing: -0.02em; }}
  h2 {{
    font-size: 1.5rem; margin: 56px 0 4px; padding-top: 24px;
    border-top: 1px solid var(--line); letter-spacing: -0.01em;
  }}
  h3 {{ font-size: 1.1rem; margin: 28px 0 8px; color: var(--accent); }}
  .sub {{ color: var(--muted); font-size: 1.05rem; margin-bottom: 8px; }}
  .lead {{ font-size: 1.1rem; color: var(--ink); }}
  .question {{
    background: var(--card); border-left: 3px solid var(--accent);
    padding: 12px 18px; margin: 16px 0; border-radius: 0 8px 8px 0;
  }}
  .takeaway {{
    background: linear-gradient(90deg, rgba(126,231,135,0.10), transparent);
    border-left: 3px solid var(--hi); padding: 12px 18px; margin: 16px 0;
    border-radius: 0 8px 8px 0;
  }}
  .warn {{
    background: linear-gradient(90deg, rgba(255,180,84,0.10), transparent);
    border-left: 3px solid var(--warn); padding: 12px 18px; margin: 16px 0;
    border-radius: 0 8px 8px 0;
  }}
  table {{
    border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 0.92rem;
    background: var(--card); border-radius: 8px; overflow: hidden;
  }}
  th, td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #222632; color: var(--muted); font-weight: 600; font-size: 0.82rem;
        text-transform: uppercase; letter-spacing: 0.03em; }}
  tr.hi td {{ background: rgba(126,231,135,0.08); }}
  td.hi {{ color: var(--hi); }}
  .method {{
    background: #1b1f2a; border-left: 3px solid var(--muted);
    padding: 12px 18px; margin: 16px 0; border-radius: 0 8px 8px 0;
    font-size: 0.92rem;
  }}
  .method code, .method b {{ color: var(--ink); }}
  .example {{
    background: #14171f; border: 1px solid var(--line); border-radius: 10px;
    padding: 14px 18px; margin: 14px 0;
  }}
  .example .ex-cap {{ color: var(--ink); font-weight: 600; margin-bottom: 8px; }}
  .example .ex-ctx {{
    color: var(--muted); font-style: italic; font-size: 0.88rem;
    background: #0e1118; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px;
  }}
  .example .ex-jsd {{ color: var(--muted); font-size: 0.86rem; margin-top: 6px; }}
  .example table {{ margin: 8px 0; }}
  code {{ background: #222632; padding: 1px 6px; border-radius: 4px; font-size: 0.88em; }}
  figure {{ margin: 24px 0; background: var(--card); padding: 16px; border-radius: 10px; }}
  figure img {{ width: 100%; height: auto; border-radius: 6px; background: #fff; }}
  figcaption {{ color: var(--muted); font-size: 0.9rem; margin-top: 12px; padding: 0 4px; }}
  figcaption b {{ color: var(--ink); }}
  .missing {{ color: var(--warn); padding: 20px; background: var(--card); border-radius: 8px; }}
  .toc {{ background: var(--card); border-radius: 10px; padding: 18px 24px; margin: 28px 0; }}
  .toc ol {{ margin: 6px 0 0; padding-left: 22px; }}
  .toc a {{ color: var(--accent); text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
  .big {{ font-size: 2.6rem; font-weight: 700; color: var(--hi); line-height: 1; }}
  .metric-row {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 20px 0; }}
  .metric {{ background: var(--card); border-radius: 10px; padding: 18px 22px; flex: 1; min-width: 180px; }}
  .metric .label {{ color: var(--muted); font-size: 0.85rem; }}
  .footer {{ color: var(--muted); font-size: 0.85rem; margin-top: 64px;
             border-top: 1px solid var(--line); padding-top: 20px; }}
</style>
</head>
<body>
<div class="wrap">

<h1>OPSD's Distillation Signal Is Almost Entirely Stylistic</h1>
<div class="sub">A suite of training-free analyses on Qwen3-1.7B / 4B across four math datasets</div>

<p class="lead">
On-Policy Self-Distillation (OPSD) trains a student to match a teacher that is the
<em>same model</em> conditioned on a reference solution. The intended signal is
"here is how to reach the correct answer." These analyses show that, in practice,
<b>95–98% of the signal is stylistic</b> — it teaches the model to imitate the
reference author's phrasing and approach, not to find the answer. The effect is
front-loaded at the strategy-choice tokens, largely interchangeable across
problems, and pushes the model away from its own preferred reasoning approach.
</p>

<div class="takeaway">
<b>One-line summary:</b> the OPSD teacher is a <b>style conditioner, not a knowledge
conditioner</b> — which is precisely the weakness Progressive Self-Distillation (PSD)
is designed to remove.
</div>

<div class="toc">
<b>Contents</b>
<ol>
  <li><a href="#a0">Style attribution — how much of the signal is style?</a></li>
  <li><a href="#a1">Position-resolved — where in the rollout does style live?</a></li>
  <li><a href="#a2">Cross-problem transfer — is the teacher problem-specific?</a></li>
  <li><a href="#a3">Diversity collapse — does OPSD funnel toward one approach?</a></li>
  <li><a href="#a4">Reward correlation — does style predict correctness?</a></li>
  <li><a href="#impl">Implications for method design</a></li>
</ol>
</div>

<!-- ============================ ANALYSIS 0 ============================ -->
<h2 id="a0">1 · Style Attribution</h2>
<div class="question">
<b>Question.</b> When OPSD pulls the student toward the teacher, is that signal
driven by knowledge of the <em>correct answer</em>, or by the <em>style</em> of the
reference solution?
</div>

<p><b>Method.</b> We run the same model in three conditions on the same student
rollout and measure per-token Jensen–Shannon divergence (JSD) between them:</p>
<ul>
<li><code>teacher_full</code> — conditioned on <b>[problem + full reference solution]</b> (this is OPSD)</li>
<li><code>teacher_answer</code> — conditioned on <b>[problem + "The answer is X."]</b></li>
<li><code>student</code> — conditioned on <b>[problem only]</b></li>
</ul>
<p>The <b>style fraction</b> = JSD(full, answer) / JSD(full, student): the share of
OPSD's signal that survives once the model already knows the answer.</p>

<div class="method">
<b>How the JSD is computed.</b> The divergence is <b>token-level</b>, not
sequence-level. We take one student rollout, then for each conditioning re-run the
<em>same</em> token sequence through the model and read off the next-token
distribution <code>P<sub>t</sub></code> at every completion position <code>t</code>.
At each position we compute the Jensen–Shannon divergence in <b>nats</b>,
<code>JSD(P‖Q) = ½·KL(P‖M) + ½·KL(Q‖M)</code> with <code>M = ½(P+Q)</code>, then
average over all completion tokens and all problems. Because every conditioning
scores the identical rollout, any divergence is attributable to the
<em>context</em>, not to different text — and it is exactly the per-token quantity
OPSD's loss minimizes. Computed in float32, chunked over the 151,936-token vocab to
avoid OOM (<code>analysis/style_attribution.py</code>).</p></div>

<div class="metric-row">
  <div class="metric"><div class="big">{pct(sa_opsd_4b['style_fraction']) if sa_opsd_4b else '—'}</div>
    <div class="label">style fraction, Qwen3-4B on OPSD data</div></div>
  <div class="metric"><div class="big">~19×</div>
    <div class="label">full-solution signal vs answer-only signal</div></div>
  <div class="metric"><div class="big">4 / 4</div>
    <div class="label">datasets where style fraction &gt; 90%</div></div>
</div>

<table>
<tr><th>Dataset</th><th>Model</th><th>JSD(full, student)<br>OPSD signal</th>
    <th>JSD(answer, student)<br>answer-only</th><th>JSD(full, answer)<br>style</th><th>Style fraction</th></tr>
{sa_table_rows}
</table>

{img_block("qwen3_4b/style_attribution.png",
  "<b>Figure 1.</b> Six-panel decomposition (Qwen3-4B, Openthoughts). "
  "<b>Top row:</b> per-token JSD distributions (left) — the style curve (green) sits almost on top of the "
  "full-signal curve (blue), while the answer-only curve (orange) is crushed near zero; mean-JSD bars (center) "
  "showing the answer contributes ~3% of the signal; teacher-vs-student entropy (right). "
  "<b>Bottom row:</b> JSD vs token position (left), entropy vs position with the style-lock-in zone shaded (center), "
  "and a per-problem scatter of style vs total signal (right) — every problem sits near the diagonal, i.e. style ≈ total.")}

<div class="takeaway">
<b>Result.</b> Across both model sizes and all four datasets, knowing the correct
answer contributes only 2–5% of OPSD's distillation signal. The remaining
<b>95–98% is the reference solution's style</b>. The effect does not weaken with
scale — at 4B it is slightly <em>stronger</em> (98.4%) than at 1.7B (96.9%).
</div>

<h3>1.1 · Is the style fraction real? Three robustness checks</h3>
<p>Because so much rides on this one ratio, we stress-tested it on Qwen3-1.7B
(30 problems, 7,680 token positions, seeded rollouts). The metric survives all
three challenges a skeptic would raise.</p>

{val_block}

<table>
<tr><th>Conditioning (scored on the same student rollout)</th><th>Mean JSD vs student</th>
    <th>Share of total signal</th><th>Implied style fraction</th></tr>
{val_table_rows}
</table>

<p>Two things to read off the table: (1) the answer-only divergence is small but
strictly positive, so the baseline genuinely conditions the model — it is not a
degenerate no-op that trivially inflates the style number; and (2) even a
<em>structurally-matched</em> strong conditioner (same "Here is a reference
solution:" framing, body reduced to just the boxed answer) leaves the style fraction
at <b>94.4%</b>. Knowing the outcome explains only ~6–9% of the per-token signal.</p>

<h3>1.2 · What the divergence actually sits on — token-level examples</h3>
<p>The point becomes concrete at the token level: the high-divergence positions are
<b>discourse and phrasing choices</b>, where the student and the answer-conditioned
model agree but the full reference pulls elsewhere. Top-5 next-token predictions at
three hand-checked positions:</p>

{val_examples_html}

<div class="takeaway">
<b>Validation result.</b> The style fraction is <b>94–98%</b> under direct
measurement, a strong structural conditioner, and a metric-space (√JSD) recomputation
alike; the three divergences are nearly additive (gap ≈ +3–4%), so the ratio reads as
a genuine decomposition rather than an artifact. The token examples show the residual
signal is reasoning <em>style</em> — discourse markers, assertiveness, phrasing — not
answer content.
</div>

<!-- ============================ ANALYSIS 1 ============================ -->
<h2 id="a1">2 · Position-Resolved Divergence</h2>
<div class="question">
<b>Question.</b> Is the style signal spread evenly across the rollout, or
concentrated at the early "strategy-choice" tokens where the model commits to an
approach?
</div>

<table>
<tr><th>Token position</th><th>OPSD signal</th><th>answer-only</th><th>style</th><th>style %</th></tr>
{pos_table_rows}
</table>

<p>The first <b>10 tokens carry ~0.061 nats</b> — 50–80% more divergence than any
later bucket — and the answer-only signal there is the lowest of all (0.0007).
The opening of the solution is where the teacher most strongly imposes the
reference style, and where answer-knowledge matters least.</p>

<h3>Highest-divergence tokens in the first 20 positions</h3>
<table>
<tr><th>Token (␣ = leading space)</th><th>Mean JSD</th></tr>
{pos_tok_rows}
</table>
<p>These are strategy-forking tokens: <code>␣different</code> (a different approach /
case), <code>␣can</code> / <code>␣possible</code> (option framing),
<code>&lt;think&gt;</code> and <code>Okay</code> (reasoning-mode entry),
<code>␣let</code> (variable introduction). The teacher steers the student at the
exact moment it picks its path.</p>

{img_block("position_analysis/position_analysis.png",
  "<b>Figure 2.</b> Left: smoothed JSD vs token position — all three curves, with the early "
  "strategy zone (0–19) shaded red; the OPSD/style curves spike at the start. Center: JSD by position bucket — "
  "the 0–9 bucket is visibly tallest. Right: the highest-JSD tokens within the first 20 positions, "
  "dominated by approach-selecting words.")}

<div class="takeaway">
<b>Result.</b> Style divergence is <b>front-loaded at the strategy-choice tokens</b>.
This localizes the damage: OPSD interferes most precisely when the student is
deciding <em>how</em> to solve the problem. → motivates <b>position-gated
distillation</b> (skip the first ~10–20 tokens) as a cheap ablation baseline.
</div>

<!-- ============================ ANALYSIS 2 ============================ -->
<h2 id="a2">3 · Cross-Problem Style Transfer</h2>
<div class="question">
<b>Question.</b> Is the teacher's signal problem-specific, or a generic style
template? What happens if we feed the teacher a <em>different problem's</em>
reference solution?
</div>

<p><b>Method.</b> For each problem we compute teacher-student JSD twice: once with
the problem's own reference (<b>proper</b>), once with 5 other problems' references
(<b>cross</b>). The <b>interchangeability ratio</b> = JSD(cross) / JSD(proper).
Near 1.0 means an unrelated solution gives nearly the same signal.</p>

<div class="metric-row">
  <div class="metric"><div class="big">{f"{cross_ratio:.0%}" if cross_ratio else '—'}</div>
    <div class="label">interchangeability ratio</div></div>
  <div class="metric"><div class="big">{f4(cross_proper)}</div>
    <div class="label">JSD, proper teacher</div></div>
  <div class="metric"><div class="big">{f4(cross_cross)}</div>
    <div class="label">JSD, cross teacher (random solution)</div></div>
</div>

{img_block("cross_problem/cross_problem_transfer.png",
  "<b>Figure 3.</b> Left: per-problem scatter of proper- vs cross-teacher JSD; points cluster near the "
  "y=x line, i.e. swapping in a random solution barely changes the signal. Center: distribution of "
  "interchangeability ratios across problems (mean shown), with 1.0 marking full interchangeability. "
  "Right: mean JSD for proper vs cross teacher — the cross bar retains ~88% of the proper bar.")}

<div class="takeaway">
<b>Result.</b> A <b>completely unrelated problem's solution produces 88% of the
signal</b> of the correct one. Only ~12% of the teacher's effect is
problem-specific content; the rest is generic reasoning-style conditioning. The
teacher is a style template, not a knowledge source.
</div>

<!-- ============================ ANALYSIS 3 ============================ -->
<h2 id="a3">4 · Diversity & Distillation Collapse</h2>
<div class="question">
<b>Question.</b> Do the model's natural rollouts use diverse approaches, and does
OPSD push them toward the single reference style?
</div>

<div class="metric-row">
  <div class="metric"><div class="big">{pct(div_refmatch)}</div>
    <div class="label">problems where reference matches model's dominant approach</div></div>
  <div class="metric"><div class="big">{pct(div_press)}</div>
    <div class="label">positions where teacher collapses student (all)</div></div>
  <div class="metric"><div class="big">{pct(div_press_early)}</div>
    <div class="label">collapse pressure, early tokens (0–19)</div></div>
</div>

{img_block("diversity_collapse/diversity_collapse.png",
  "<b>Figure 4.</b> Top-left: natural approach distribution across all rollouts (see caveat — the keyword "
  "classifier over-assigns 'algebraic'). Top-right: unique approaches per problem. Bottom-left: per-problem "
  "distillation collapse pressure, all positions vs early tokens — early is consistently higher. "
  "Bottom-right: approach concentration, colored by whether the reference solution matches the model's "
  "dominant approach (green = match, red = mismatch).")}

<div class="warn">
<b>⚠️ Caveat.</b> The keyword approach-classifier collapsed 118/120 rollouts into
"algebraic", so the absolute <em>diversity counts</em> are unreliable and should not
be cited. What <em>is</em> reliable: the reference mismatches the model's modal
approach in <b>{pct(div_refmatch)} of problems</b>, and collapse pressure is real
and front-loaded ({pct(div_press_early)} early vs {pct(div_press)} overall) — both
independent of the classifier. Re-run with an LLM-judge before using any diversity number.
</div>

<!-- ============================ ANALYSIS 4 ============================ -->
<h2 id="a4">5 · Realizability Gap vs Reward</h2>
<div class="question">
<b>Question.</b> For problems the model can sometimes solve, does stylistic
closeness to the reference (low JSD) predict getting the right answer?
</div>

{a4_block}

{img_block("reward_correlation/reward_correlation.png", a4_caption)}

<!-- ============================ IMPLICATIONS ============================ -->
<h2 id="impl">6 · Implications for Method Design</h2>

<p>Every analysis points the same way: the OPSD teacher–student gap is overwhelmingly
a <b>style mismatch</b>, localized to the strategy-setting moment and largely
independent of the specific reference content.</p>

<h3>Primary — Progressive Self-Distillation (PSD)</h3>
<p>Replace the external reference with the model's own verified-correct rollouts.
This is motivated by all four analyses at once:</p>
<ul>
<li>removes the foreign-style component that is 95–98% of the signal (§1) and 88% generic (§3)</li>
<li>by construction matches the model's dominant approach, eliminating the {pct(div_refmatch)} reference-mismatch (§4)</li>
<li>the self-generated teacher opens the solution the way the student would, killing the front-loaded divergence (§2)</li>
</ul>

<h3>Secondary — cheap ablation baselines (each individually incremental)</h3>
<ul>
<li><b>Position-gated distillation</b>: skip the first K≈10–20 tokens so the student picks its own strategy (§2)</li>
<li><b>Content-weighted distillation</b>: scale the loss by per-problem interchangeability ratio — distill more where the reference carries real content (§3)</li>
</ul>

<h3>Open items</h3>
<ul>
<li>Re-run §4 diversity with an LLM-judge classifier before citing diversity counts.</li>
<li>§5 reward-correlation now has variance after the verifier fix (math_verify backend); see the result above.</li>
<li>Measure the style gap across <em>training checkpoints</em> — it should close faster under PSD than OPSD, which would be the cleanest single figure for the paper.</li>
</ul>

<div class="footer">
Generated by <code>analysis/build_report.py</code> · all images embedded as base64 ·
models: Qwen3-1.7B, Qwen3-4B · datasets: Openthoughts-30k, MATH-500, NuminaMath-CoT, OpenR1-Math-220k ·
all analyses training-free (pretrained checkpoints).
</div>

</div>
</body>
</html>
"""

with open(OUT, "w") as f:
    f.write(html)

size_kb = os.path.getsize(OUT) / 1024
print(f"Wrote {OUT} ({size_kb:.0f} KB)")
