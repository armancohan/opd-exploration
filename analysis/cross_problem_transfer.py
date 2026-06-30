"""Analysis 2: Cross-problem style transfer test.

For each problem i, we use a *different* problem j's reference solution as the
teacher context, and compare:
  JSD(teacher_proper_i, student_i)   — correct teacher for problem i
  JSD(teacher_cross_j,  student_i)   — wrong problem's solution as teacher

If cross-teacher JSD ≈ proper-teacher JSD, solutions are near-interchangeable
style templates — pure style with no problem-specific content.

Interchangeability ratio = mean_j JSD(cross_j) / JSD(proper)
A ratio near 1.0 means the reference solution's specific content is irrelevant;
the style effect is what dominates.
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
                             chunked_jsd, extract_answer, smooth)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    data = load_train_dataset(args.dataset, n_samples=args.n_problems, split=args.split)
    N = len(data)
    print(f"Loaded {N} problems")

    # Step 1: generate rollouts for all problems
    rollouts = []
    for idx, item in enumerate(data):
        print(f"  Generating rollout {idx+1}/{N}...", end=" ", flush=True)
        student_prompt = apply_chat_template(
            tokenizer, STUDENT_PROMPT_TEMPLATE.format(problem=item["problem"]))
        comp_ids, comp_text, _ = generate_rollout(
            model, tokenizer, student_prompt,
            args.max_new_tokens, args.max_prompt_len, device)
        rollouts.append({"comp_ids": comp_ids, "comp_text": comp_text})
        print(f"OK ({comp_ids.shape[0]} tokens)")

    # Step 2: for each problem, compute proper-teacher and cross-teacher JSD
    proper_jsds = []
    cross_jsds  = []   # list of lists: cross_jsds[i] = [JSD with j's solution for j≠i]
    interchange_ratios = []

    for i, item_i in enumerate(data):
        print(f"  Problem {i+1}/{N} teacher analysis...", end=" ", flush=True)
        comp_ids = rollouts[i]["comp_ids"]
        if comp_ids.shape[0] == 0:
            print("SKIP"); continue

        student_prompt = apply_chat_template(
            tokenizer, STUDENT_PROMPT_TEMPLATE.format(problem=item_i["problem"]))
        ls = get_completion_logits(model, tokenizer, student_prompt,
                                   comp_ids, args.max_prompt_len, device)

        # Proper teacher
        tf_proper = apply_chat_template(
            tokenizer, TEACHER_PROMPT_TEMPLATE.format(
                problem=item_i["problem"], solution=item_i["solution"]))
        lf_proper = get_completion_logits(model, tokenizer, tf_proper,
                                          comp_ids, args.max_prompt_len, device)
        T = min(lf_proper.shape[0], ls.shape[0])
        jsd_proper = float(chunked_jsd(lf_proper[:T], ls[:T]).mean().item())
        proper_jsds.append(jsd_proper)

        # Cross teachers: use a sample of other problems' solutions
        cross_sample = [j for j in range(N) if j != i][:args.n_cross]
        cross_vals = []
        for j in cross_sample:
            tf_cross = apply_chat_template(
                tokenizer, TEACHER_PROMPT_TEMPLATE.format(
                    problem=item_i["problem"], solution=data[j]["solution"]))
            lf_cross = get_completion_logits(model, tokenizer, tf_cross,
                                             comp_ids, args.max_prompt_len, device)
            T2 = min(lf_cross.shape[0], ls.shape[0])
            cross_vals.append(float(chunked_jsd(lf_cross[:T2], ls[:T2]).mean().item()))

        mean_cross = float(np.mean(cross_vals))
        cross_jsds.append(cross_vals)
        ratio = mean_cross / max(jsd_proper, 1e-9)
        interchange_ratios.append(ratio)
        print(f"proper={jsd_proper:.4f}  cross_mean={mean_cross:.4f}  ratio={ratio:.3f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    mean_proper = float(np.mean(proper_jsds))
    mean_cross_flat = float(np.mean([v for row in cross_jsds for v in row]))
    mean_ratio = float(np.mean(interchange_ratios))

    result = {
        "n_problems": len(proper_jsds),
        "n_cross_per_problem": args.n_cross,
        "mean_jsd_proper_teacher": mean_proper,
        "mean_jsd_cross_teacher":  mean_cross_flat,
        "interchangeability_ratio": mean_ratio,
        "per_problem_ratios": interchange_ratios,
        "per_problem_proper": proper_jsds,
        "per_problem_cross_mean": [float(np.mean(r)) for r in cross_jsds],
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Analysis 2: Cross-Problem Style Transfer", fontsize=13, fontweight="bold")

    cross_means = [float(np.mean(r)) for r in cross_jsds]

    # Panel 1: proper vs cross scatter
    ax = axes[0]
    ax.scatter(proper_jsds, cross_means, color="steelblue", s=60, alpha=0.8, edgecolors="gray", lw=0.4)
    lim = max(max(proper_jsds), max(cross_means)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y=x (identical)")
    ax.set_xlabel("JSD with proper teacher (own solution)")
    ax.set_ylabel("Mean JSD with cross teachers (other solutions)")
    ax.set_title(f"Proper vs Cross-Problem Teacher\nInterchangeability ratio = {mean_ratio:.3f}")
    ax.legend(fontsize=9)

    # Panel 2: distribution of interchangeability ratios
    ax = axes[1]
    ax.hist(interchange_ratios, bins=15, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="red", lw=1.5, ls="--", label="ratio=1 (fully interchangeable)")
    ax.axvline(mean_ratio, color="orange", lw=1.5, ls="-", label=f"mean={mean_ratio:.3f}")
    ax.set_xlabel("Interchangeability ratio\n(cross / proper teacher JSD)")
    ax.set_ylabel("Count (problems)")
    ax.set_title("Per-Problem Interchangeability\n(1.0 = solutions fully interchangeable)")
    ax.legend(fontsize=8)

    # Panel 3: proper vs cross bar chart
    ax = axes[2]
    ax.bar(["Proper teacher\n(own solution)", "Cross teacher\n(other problem's solution)"],
           [mean_proper, mean_cross_flat],
           color=["steelblue", "tomato"], alpha=0.85)
    ax.set_ylabel("Mean JSD (nats)")
    ax.set_title(f"Mean Teacher JSD\nCross/proper = {mean_ratio:.3f}")
    for i, (lbl, val) in enumerate([("", mean_proper), ("", mean_cross_flat)]):
        ax.text(i, val + 0.0005, f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "cross_problem_transfer.png"), dpi=150, bbox_inches="tight")

    print("\n" + "=" * 58)
    print("     Analysis 2: Cross-Problem Style Transfer")
    print("=" * 58)
    print(f"Problems:                    {len(proper_jsds)}")
    print(f"Cross-teachers per problem:  {args.n_cross}")
    print(f"Mean JSD(proper teacher):    {mean_proper:.4f}")
    print(f"Mean JSD(cross teacher):     {mean_cross_flat:.4f}")
    print(f"Interchangeability ratio:    {mean_ratio:.3f}")
    print()
    if mean_ratio > 0.80:
        print("=> HIGH interchangeability: solutions from different problems")
        print("   produce nearly the same teacher signal. The reference")
        print("   solution is acting as a generic style template, not a")
        print("   problem-specific knowledge source.")
    elif mean_ratio > 0.50:
        print("=> MODERATE interchangeability: cross-problem solutions give")
        print("   somewhat different signal, but style still dominates.")
    else:
        print("=> LOW interchangeability: solutions carry problem-specific")
        print("   information beyond style.")
    print("=" * 58)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    p.add_argument("--split", default="train")
    p.add_argument("--n_problems", type=int, default=15)
    p.add_argument("--n_cross", type=int, default=5,
                   help="Number of cross-problem solutions to test per problem")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--output_dir", default="analysis/cross_problem")
    run(p.parse_args())
