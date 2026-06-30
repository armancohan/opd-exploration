"""Analysis 4: Teacher-student realizability gap vs. reward correlation.

For problems the model can sometimes solve correctly, does stylistic
similarity to the reference (low JSD) correlate with getting the right answer?

If correct rollouts have LOWER JSD with the teacher (= more stylistically
similar to the reference), OPSD's style signal accidentally helps — the
reference style is a proxy for correct reasoning.

If there is NO correlation (or HIGHER JSD for correct rollouts), then style
similarity is orthogonal to answer correctness, confirming OPSD's distillation
signal is noise from the reward perspective.

Uses MATH-500 (level 1-3 problems) where the model can sometimes succeed.
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from src.data import load_train_dataset, TEACHER_PROMPT_TEMPLATE, STUDENT_PROMPT_TEMPLATE
from src.verifier import batch_verify
from analysis.utils import (load_model_and_tokenizer, apply_chat_template,
                             get_completion_logits, generate_rollout,
                             chunked_jsd, chunked_entropy, extract_answer, smooth)


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Load MATH-500 (has 'level' field — filter to easier problems for more rewards)
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    if "level" in ds.column_names:
        ds = ds.filter(lambda x: int(x.get("level", 5)) <= args.max_level)
        print(f"Filtered to level ≤ {args.max_level}: {len(ds)} problems")
    ds = ds.shuffle(seed=42).select(range(min(args.n_problems, len(ds))))

    data = [{"problem": x["problem"], "solution": x["solution"]} for x in ds]
    print(f"Using {len(data)} problems")

    all_jsds    = []   # mean JSD per rollout
    all_rewards = []   # 0 or 1 per rollout
    all_correct_jsds   = []
    all_incorrect_jsds = []
    problem_results = []

    for idx, item in enumerate(data):
        print(f"  Problem {idx+1}/{len(data)}...", end=" ", flush=True)
        problem, solution = item["problem"], item["solution"]

        student_prompt = apply_chat_template(
            tokenizer, STUDENT_PROMPT_TEMPLATE.format(problem=problem))
        tf_prompt = apply_chat_template(
            tokenizer, TEACHER_PROMPT_TEMPLATE.format(problem=problem, solution=solution))

        rollout_texts, rollout_ids_list = [], []
        for _ in range(args.k_rollouts):
            comp_ids, comp_text, _ = generate_rollout(
                model, tokenizer, student_prompt,
                args.max_new_tokens, args.max_prompt_len, device)
            rollout_texts.append(comp_text)
            rollout_ids_list.append(comp_ids)

        rewards = batch_verify(rollout_texts, [solution] * args.k_rollouts)
        n_correct = sum(1 for r in rewards if r > 0.5)
        print(f"reward={np.mean(rewards):.2f} ({n_correct}/{args.k_rollouts} correct)")

        # Compute JSD for each rollout
        rollout_jsds = []
        for comp_ids, reward in zip(rollout_ids_list, rewards):
            if comp_ids.shape[0] == 0:
                rollout_jsds.append(None); continue
            lf = get_completion_logits(model, tokenizer, tf_prompt,
                                       comp_ids, args.max_prompt_len, device)
            ls = get_completion_logits(model, tokenizer, student_prompt,
                                       comp_ids, args.max_prompt_len, device)
            T = min(lf.shape[0], ls.shape[0])
            mean_jsd = float(chunked_jsd(lf[:T], ls[:T]).mean().item())
            rollout_jsds.append(mean_jsd)

            all_jsds.append(mean_jsd)
            all_rewards.append(float(reward))
            if reward > 0.5:
                all_correct_jsds.append(mean_jsd)
            else:
                all_incorrect_jsds.append(mean_jsd)

        problem_results.append({
            "problem_idx": idx,
            "n_correct": n_correct,
            "mean_reward": float(np.mean(rewards)),
            "rollout_jsds": [j for j in rollout_jsds if j is not None],
            "mean_jsd_correct":   float(np.mean([j for j, r in zip(rollout_jsds, rewards) if j is not None and r > 0.5])) if n_correct > 0 else None,
            "mean_jsd_incorrect": float(np.mean([j for j, r in zip(rollout_jsds, rewards) if j is not None and r <= 0.5])) if n_correct < args.k_rollouts else None,
        })

    # ── Statistics ────────────────────────────────────────────────────────────
    total_reward_rate = float(np.mean(all_rewards))
    mean_jsd_correct   = float(np.mean(all_correct_jsds))  if all_correct_jsds   else None
    mean_jsd_incorrect = float(np.mean(all_incorrect_jsds)) if all_incorrect_jsds else None

    # Point-biserial correlation: JSD vs binary reward
    if len(set(all_rewards)) > 1:
        corr, pval = stats.pointbiserialr(all_rewards, all_jsds)
    else:
        corr, pval = 0.0, 1.0

    result = {
        "n_problems": len(data),
        "k_rollouts": args.k_rollouts,
        "total_reward_rate": total_reward_rate,
        "n_correct_rollouts": len(all_correct_jsds),
        "n_incorrect_rollouts": len(all_incorrect_jsds),
        "mean_jsd_correct_rollouts":   mean_jsd_correct,
        "mean_jsd_incorrect_rollouts": mean_jsd_incorrect,
        "jsd_correct_minus_incorrect": (mean_jsd_correct - mean_jsd_incorrect) if (mean_jsd_correct and mean_jsd_incorrect) else None,
        "pointbiserial_correlation_jsd_vs_reward": corr,
        "correlation_pvalue": pval,
        "per_problem": problem_results,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Analysis 4: Realizability Gap vs Reward", fontsize=13, fontweight="bold")

    # Panel 1: JSD distribution by correct/incorrect
    ax = axes[0]
    if all_correct_jsds:
        ax.hist(all_correct_jsds,   bins=25, alpha=0.65, color="green",
                label=f"Correct rollouts (n={len(all_correct_jsds)})", density=True)
    if all_incorrect_jsds:
        ax.hist(all_incorrect_jsds, bins=25, alpha=0.65, color="tomato",
                label=f"Incorrect rollouts (n={len(all_incorrect_jsds)})", density=True)
    ax.set_xlabel("Mean JSD(teacher_full, student_rollout)")
    ax.set_ylabel("Density")
    ax.set_title("JSD Distribution by Correctness\n(lower JSD = closer to reference style)")
    ax.legend(fontsize=9)

    # Panel 2: mean JSD bar
    ax = axes[1]
    vals, lbls, cols = [], [], []
    if mean_jsd_correct is not None:
        vals.append(mean_jsd_correct); lbls.append("Correct"); cols.append("green")
    if mean_jsd_incorrect is not None:
        vals.append(mean_jsd_incorrect); lbls.append("Incorrect"); cols.append("tomato")
    if vals:
        bars = ax.bar(lbls, vals, color=cols, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02, f"{v:.4f}",
                    ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Mean JSD (nats)")
    ax.set_title(f"Mean Teacher-Student JSD\nby Rollout Outcome\n(r={corr:.3f}, p={pval:.3f})")

    # Panel 3: per-problem reward rate vs mean JSD gap
    ax = axes[2]
    rewards_pp = [r["mean_reward"] for r in problem_results]
    jsd_gaps = []
    for r in problem_results:
        if r["mean_jsd_correct"] is not None and r["mean_jsd_incorrect"] is not None:
            jsd_gaps.append((r["mean_reward"], r["mean_jsd_correct"] - r["mean_jsd_incorrect"]))
    if jsd_gaps:
        xs, ys = zip(*jsd_gaps)
        ax.scatter(xs, ys, color="steelblue", s=70, alpha=0.8, edgecolors="gray", lw=0.4)
        ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.6)
        ax.set_xlabel("Problem reward rate")
        ax.set_ylabel("JSD(correct) - JSD(incorrect)\n(negative = correct rollouts closer to ref style)")
        ax.set_title("Per-Problem: Does Correct Style\nMatch Reference?")
    else:
        ax.text(0.5, 0.5, "No problems with both\ncorrect and incorrect rollouts",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "reward_correlation.png"), dpi=150, bbox_inches="tight")

    print("\n" + "=" * 62)
    print("     Analysis 4: Realizability Gap vs Reward")
    print("=" * 62)
    print(f"Problems: {len(data)}, rollouts/problem: {args.k_rollouts}")
    print(f"Overall reward rate: {total_reward_rate:.3f}")
    print(f"Correct rollouts:   {len(all_correct_jsds)}")
    print(f"Incorrect rollouts: {len(all_incorrect_jsds)}")
    print()
    if mean_jsd_correct is not None:
        print(f"Mean JSD (correct rollouts):   {mean_jsd_correct:.4f}")
    if mean_jsd_incorrect is not None:
        print(f"Mean JSD (incorrect rollouts): {mean_jsd_incorrect:.4f}")
    print(f"Point-biserial r(JSD, reward): {corr:.4f}  (p={pval:.4f})")
    print()
    if corr < -0.1 and pval < 0.05:
        print("=> NEGATIVE correlation: closer to reference style → more likely correct.")
        print("   OPSD's style signal is a proxy for correct reasoning on this dataset.")
    elif abs(corr) < 0.1 or pval > 0.1:
        print("=> NO significant correlation: reference style similarity is orthogonal")
        print("   to answer correctness. OPSD's signal is stylistic noise w.r.t. reward.")
    else:
        print(f"=> Correlation r={corr:.3f} (p={pval:.3f}): interpret with caution.")
    print("=" * 62)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    p.add_argument("--split", default="test")
    p.add_argument("--n_problems", type=int, default=30)
    p.add_argument("--k_rollouts", type=int, default=8)
    p.add_argument("--max_level", type=int, default=3,
                   help="Max MATH difficulty level (1-5); lower = easier = more rewards")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--output_dir", default="analysis/reward_correlation")
    run(p.parse_args())
