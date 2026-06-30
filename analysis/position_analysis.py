"""Analysis 1: Where in the rollout does the style divergence concentrate?

Tests whether OPSD's style signal is front-loaded at early 'strategy-choice'
tokens (positions 0-19) or spread uniformly across the sequence.

If high-JSD positions cluster at the start — where the model chooses its
reasoning approach — OPSD is most harmful precisely when the student is
making its key branching decisions.
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_train_dataset, TEACHER_PROMPT_TEMPLATE, STUDENT_PROMPT_TEMPLATE
from analysis.utils import (load_model_and_tokenizer, apply_chat_template,
                             get_completion_logits, generate_rollout,
                             chunked_jsd, chunked_entropy, extract_answer, smooth)

BUCKETS = [(0, 10), (10, 20), (20, 50), (50, 150), (150, 9999)]
BUCKET_LABELS = ["0–9", "10–19", "20–49", "50–149", "150+"]


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    data = load_train_dataset(args.dataset, n_samples=args.n_problems, split=args.split)
    print(f"Loaded {len(data)} problems from {args.dataset}")

    all_jsd_full = []   # per-token JSD(full, student) arrays
    all_jsd_ans  = []
    all_jsd_sty  = []
    all_tokens   = []   # decoded token strings at each position

    for idx, item in enumerate(data):
        print(f"  {idx+1}/{len(data)}...", end=" ", flush=True)
        problem, solution = item["problem"], item["solution"]
        answer = extract_answer(solution)

        student_prompt = apply_chat_template(tokenizer, STUDENT_PROMPT_TEMPLATE.format(problem=problem))
        comp_ids, _, _ = generate_rollout(model, tokenizer, student_prompt,
                                          args.max_new_tokens, args.max_prompt_len, device)
        if comp_ids.shape[0] == 0:
            print("SKIP"); continue

        tf_text = apply_chat_template(tokenizer, TEACHER_PROMPT_TEMPLATE.format(problem=problem, solution=solution))
        ta_text = apply_chat_template(tokenizer, f"Problem: {problem}\n\nThe answer is: {answer}\n\nReason step by step and put your final answer within \\boxed{{}}.")

        lf = get_completion_logits(model, tokenizer, tf_text, comp_ids, args.max_prompt_len, device)
        la = get_completion_logits(model, tokenizer, ta_text, comp_ids, args.max_prompt_len, device)
        ls = get_completion_logits(model, tokenizer, student_prompt, comp_ids, args.max_prompt_len, device)
        T = min(lf.shape[0], la.shape[0], ls.shape[0])

        jsd_fs = chunked_jsd(lf[:T], ls[:T]).cpu().numpy()
        jsd_as = chunked_jsd(la[:T], ls[:T]).cpu().numpy()
        jsd_fa = chunked_jsd(lf[:T], la[:T]).cpu().numpy()

        toks = tokenizer.convert_ids_to_tokens(comp_ids[:T].tolist())
        all_jsd_full.append(jsd_fs)
        all_jsd_ans.append(jsd_as)
        all_jsd_sty.append(jsd_fa)
        all_tokens.append(toks)
        print(f"OK ({T} tokens)")

    if not all_jsd_full:
        print("No results."); return

    # ── Bucket analysis ──────────────────────────────────────────────────────
    def bucket_means(arrays):
        results = []
        for lo, hi in BUCKETS:
            vals = np.concatenate([a[lo:min(hi, len(a))] for a in arrays if len(a) > lo])
            results.append((float(np.mean(vals)), float(np.std(vals))) if len(vals) else (0, 0))
        return results

    bm_full = bucket_means(all_jsd_full)
    bm_ans  = bucket_means(all_jsd_ans)
    bm_sty  = bucket_means(all_jsd_sty)

    # ── Top divergent tokens in first 20 positions ───────────────────────────
    early_tok_jsd = {}  # token_str -> list of JSD values
    for jsd_arr, toks in zip(all_jsd_full, all_tokens):
        for pos in range(min(20, len(jsd_arr))):
            tok = toks[pos].replace("▁", " ").strip()
            early_tok_jsd.setdefault(tok, []).append(float(jsd_arr[pos]))
    tok_mean_jsd = {t: np.mean(v) for t, v in early_tok_jsd.items() if len(v) >= 2}
    top_tokens = sorted(tok_mean_jsd.items(), key=lambda x: -x[1])[:20]

    # ── Position-resolved curves (align to shortest) ─────────────────────────
    min_T = min(len(a) for a in all_jsd_full)
    pos_full = np.stack([a[:min_T] for a in all_jsd_full]).mean(0)
    pos_ans  = np.stack([a[:min_T] for a in all_jsd_ans]).mean(0)
    pos_sty  = np.stack([a[:min_T] for a in all_jsd_sty]).mean(0)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "n_problems": len(all_jsd_full),
        "bucket_analysis": {
            lbl: {
                "jsd_full_student": {"mean": bm_full[i][0], "std": bm_full[i][1]},
                "jsd_answer_student": {"mean": bm_ans[i][0], "std": bm_ans[i][1]},
                "jsd_style": {"mean": bm_sty[i][0], "std": bm_sty[i][1]},
                "style_fraction": bm_sty[i][0] / max(bm_full[i][0], 1e-9),
            }
            for i, lbl in enumerate(BUCKET_LABELS)
        },
        "top_early_divergent_tokens": top_tokens[:20],
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Analysis 1: Position-Resolved Divergence", fontsize=13, fontweight="bold")

    # Panel 1: position curves
    ax = axes[0]
    sm_pos = np.arange(len(smooth(pos_full)))
    ax.plot(sm_pos, smooth(pos_full), color="steelblue", lw=2, label="JSD(full, student) — OPSD signal")
    ax.plot(sm_pos, smooth(pos_ans),  color="orange",    lw=2, label="JSD(answer, student) — answer-only")
    ax.plot(sm_pos, smooth(pos_sty),  color="green",     lw=2, label="JSD(full, answer) — style")
    ax.axvspan(0, 20, alpha=0.08, color="red", label="Early strategy zone (0–19)")
    ax.set_xlabel("Token position"); ax.set_ylabel("Mean JSD (nats)")
    ax.set_title("JSD vs Token Position")
    ax.legend(fontsize=8)

    # Panel 2: bucket bar chart
    ax = axes[1]
    x = np.arange(len(BUCKET_LABELS))
    w = 0.28
    ax.bar(x - w, [bm_full[i][0] for i in range(len(BUCKET_LABELS))],
           width=w, label="OPSD signal", color="steelblue", alpha=0.85)
    ax.bar(x,     [bm_ans[i][0]  for i in range(len(BUCKET_LABELS))],
           width=w, label="answer-only", color="orange", alpha=0.85)
    ax.bar(x + w, [bm_sty[i][0]  for i in range(len(BUCKET_LABELS))],
           width=w, label="style", color="green", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(BUCKET_LABELS)
    ax.set_xlabel("Token position bucket"); ax.set_ylabel("Mean JSD (nats)")
    ax.set_title("JSD by Position Bucket")
    ax.legend(fontsize=8)

    # Panel 3: top divergent early tokens
    ax = axes[2]
    labels_t = [t for t, _ in top_tokens[:15]]
    vals_t   = [v for _, v in top_tokens[:15]]
    bars = ax.barh(range(len(labels_t)), vals_t, color="tomato", alpha=0.8)
    ax.set_yticks(range(len(labels_t))); ax.set_yticklabels(labels_t, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean JSD (nats)")
    ax.set_title("Highest-JSD Tokens\nin First 20 Positions")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "position_analysis.png"), dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output_dir}/")

    # ── Text summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("       Analysis 1: Position-Resolved Divergence")
    print("=" * 62)
    print(f"{'Bucket':<10} {'OPSD signal':>14} {'answer-only':>14} {'style':>12} {'style%':>8}")
    print("-" * 62)
    for i, lbl in enumerate(BUCKET_LABELS):
        sf = bm_sty[i][0] / max(bm_full[i][0], 1e-9)
        print(f"{lbl:<10} {bm_full[i][0]:>14.4f} {bm_ans[i][0]:>14.4f} {bm_sty[i][0]:>12.4f} {sf:>7.1%}")
    print("=" * 62)
    print("\nTop divergent tokens in first 20 positions:")
    for tok, val in top_tokens[:10]:
        print(f"  '{tok}': {val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    p.add_argument("--split", default="train")
    p.add_argument("--n_problems", type=int, default=30)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--output_dir", default="analysis/position_analysis")
    run(p.parse_args())
