"""Analysis 3: Rollout diversity collapse under OPSD distillation.

For each problem, generates K rollouts and measures natural diversity
(approach variety, token entropy). Then computes the distillation pressure
OPSD would apply at each token position — specifically, where the teacher
distribution is narrower than the student's, imposing a collapse.

Key metrics:
  - Natural diversity: distinct approach keywords across K rollouts
  - Distillation pressure: fraction of positions where teacher entropy < student
  - Approach convergence: would OPSD funnel all K approaches toward one style?
"""

import argparse, json, os, sys, re
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_train_dataset, TEACHER_PROMPT_TEMPLATE, STUDENT_PROMPT_TEMPLATE
from analysis.utils import (load_model_and_tokenizer, apply_chat_template,
                             get_completion_logits, generate_rollout,
                             chunked_jsd, chunked_entropy, extract_answer, smooth)

# Keywords that signal a reasoning approach choice (appear early in rollouts)
APPROACH_KEYWORDS = {
    "algebraic":    ["let", "equation", "solve", "variable", "substitut", "algebra"],
    "coordinate":   ["coordinat", "x-axis", "y-axis", "origin", "slope", "intersect"],
    "geometric":    ["triangle", "angle", "circle", "parallel", "perpendicular", "similar"],
    "combinatorial":["combinat", "permut", "count", "choose", "arrange", "case"],
    "induction":    ["induct", "base case", "inductive", "hypothesis", "assume"],
    "number_theory":["modulo", "divisib", "prime", "gcd", "factor", "congruent"],
}


def classify_approach(text: str) -> str:
    text_lower = text.lower()[:300]  # only first 300 chars = approach-setting zone
    for approach, keywords in APPROACH_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return approach
    return "other"


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    data = load_train_dataset(args.dataset, n_samples=args.n_problems, split=args.split)
    print(f"Loaded {len(data)} problems")

    problem_results = []

    for idx, item in enumerate(data):
        print(f"\nProblem {idx+1}/{len(data)}")
        problem, solution = item["problem"], item["solution"]
        answer = extract_answer(solution)

        student_prompt = apply_chat_template(
            tokenizer, STUDENT_PROMPT_TEMPLATE.format(problem=problem))
        tf_prompt = apply_chat_template(
            tokenizer, TEACHER_PROMPT_TEMPLATE.format(problem=problem, solution=solution))

        # Generate K rollouts
        rollout_texts, rollout_ids_list = [], []
        for k in range(args.k_rollouts):
            print(f"  rollout {k+1}/{args.k_rollouts}...", end=" ", flush=True)
            comp_ids, comp_text, _ = generate_rollout(
                model, tokenizer, student_prompt,
                args.max_new_tokens, args.max_prompt_len, device)
            rollout_texts.append(comp_text)
            rollout_ids_list.append(comp_ids)
            print(f"{classify_approach(comp_text)}", end="  ")
        print()

        # Approach diversity
        approaches = [classify_approach(t) for t in rollout_texts]
        approach_counts = Counter(approaches)
        n_unique_approaches = len(set(approaches))
        dominant_approach = approach_counts.most_common(1)[0][0]
        dominant_frac = approach_counts.most_common(1)[0][1] / args.k_rollouts

        # Reference solution's approach
        ref_approach = classify_approach(solution)

        # Per-rollout: measure distillation pressure (where teacher < student entropy)
        rollout_pressure = []
        for comp_ids in rollout_ids_list:
            if comp_ids.shape[0] == 0:
                continue
            lf = get_completion_logits(model, tokenizer, tf_prompt,
                                       comp_ids, args.max_prompt_len, device)
            ls = get_completion_logits(model, tokenizer, student_prompt,
                                       comp_ids, args.max_prompt_len, device)
            T = min(lf.shape[0], ls.shape[0])
            ent_tf = chunked_entropy(lf[:T]).cpu().numpy()
            ent_st = chunked_entropy(ls[:T]).cpu().numpy()
            jsd    = chunked_jsd(lf[:T], ls[:T]).cpu().numpy()

            pressure_frac = float(np.mean(ent_tf < ent_st))
            mean_jsd = float(np.mean(jsd))
            # Early-position pressure (pos 0-19): strategy zone
            early_pressure = float(np.mean(ent_tf[:20] < ent_st[:20])) if T >= 20 else pressure_frac
            rollout_pressure.append({
                "approach": classify_approach(tokenizer.decode(comp_ids, skip_special_tokens=True)),
                "pressure_frac": pressure_frac,
                "early_pressure_frac": early_pressure,
                "mean_jsd": mean_jsd,
            })

        problem_results.append({
            "problem_idx": idx,
            "n_unique_approaches": n_unique_approaches,
            "approach_counts": dict(approach_counts),
            "dominant_approach": dominant_approach,
            "dominant_frac": dominant_frac,
            "ref_approach": ref_approach,
            "rollout_pressure": rollout_pressure,
            "mean_pressure": float(np.mean([r["pressure_frac"] for r in rollout_pressure])),
            "mean_early_pressure": float(np.mean([r["early_pressure_frac"] for r in rollout_pressure])),
            "mean_jsd": float(np.mean([r["mean_jsd"] for r in rollout_pressure])),
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    # How often does the reference solution's approach match the dominant rollout approach?
    ref_matches_dominant = [
        1 if r["ref_approach"] == r["dominant_approach"] else 0
        for r in problem_results
    ]

    # Approach distribution across all rollouts
    all_approaches = []
    for r in problem_results:
        for approach, count in r["approach_counts"].items():
            all_approaches.extend([approach] * count)
    approach_dist = Counter(all_approaches)

    mean_unique = float(np.mean([r["n_unique_approaches"] for r in problem_results]))
    mean_dom_frac = float(np.mean([r["dominant_frac"] for r in problem_results]))
    mean_pressure = float(np.mean([r["mean_pressure"] for r in problem_results]))
    mean_early_pressure = float(np.mean([r["mean_early_pressure"] for r in problem_results]))
    ref_match_rate = float(np.mean(ref_matches_dominant))

    result = {
        "n_problems": len(problem_results),
        "k_rollouts": args.k_rollouts,
        "mean_unique_approaches_per_problem": mean_unique,
        "mean_dominant_fraction": mean_dom_frac,
        "mean_distillation_pressure": mean_pressure,
        "mean_early_position_pressure": mean_early_pressure,
        "ref_matches_dominant_approach_rate": ref_match_rate,
        "overall_approach_distribution": dict(approach_dist),
        "per_problem": problem_results,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analysis 3: Rollout Diversity & Distillation Collapse", fontsize=13, fontweight="bold")

    # Panel 1: approach distribution pie
    ax = axes[0, 0]
    labels_ap = list(approach_dist.keys())
    sizes_ap  = [approach_dist[k] for k in labels_ap]
    ax.pie(sizes_ap, labels=labels_ap, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"Natural Approach Distribution\n({args.k_rollouts} rollouts × {len(problem_results)} problems)")

    # Panel 2: unique approaches per problem
    ax = axes[0, 1]
    unique_counts = [r["n_unique_approaches"] for r in problem_results]
    ax.bar(range(len(unique_counts)), unique_counts, color="steelblue", alpha=0.8)
    ax.axhline(mean_unique, color="red", lw=1.5, ls="--", label=f"mean={mean_unique:.1f}")
    ax.set_xlabel("Problem index"); ax.set_ylabel("Unique approaches")
    ax.set_title(f"Approach Diversity per Problem\n(out of {len(APPROACH_KEYWORDS)} categories + 'other')")
    ax.legend()

    # Panel 3: distillation pressure per problem
    ax = axes[1, 0]
    pressures = [r["mean_pressure"] for r in problem_results]
    early_ps  = [r["mean_early_pressure"] for r in problem_results]
    x = np.arange(len(pressures))
    ax.bar(x - 0.2, pressures,  width=0.4, label="All positions",  color="tomato", alpha=0.8)
    ax.bar(x + 0.2, early_ps,   width=0.4, label="Early (0–19)",   color="darkred", alpha=0.8)
    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.5, label="50% baseline")
    ax.set_xlabel("Problem index")
    ax.set_ylabel("Fraction of positions where\nteacher entropy < student")
    ax.set_title("Distillation Collapse Pressure\n(higher = more style imposition)")
    ax.legend(fontsize=8)

    # Panel 4: ref approach match vs diversity
    ax = axes[1, 1]
    dom_fracs = [r["dominant_frac"] for r in problem_results]
    ref_match = [r["ref_approach"] == r["dominant_approach"] for r in problem_results]
    colors_m = ["green" if m else "tomato" for m in ref_match]
    ax.scatter(range(len(dom_fracs)), dom_fracs, c=colors_m, s=80, edgecolors="gray", lw=0.5)
    ax.axhline(1/args.k_rollouts, color="gray", ls="--", alpha=0.5, label="uniform baseline")
    ax.set_xlabel("Problem index")
    ax.set_ylabel("Fraction of rollouts using dominant approach")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="green", label="ref matches dominant"),
                       Patch(color="tomato", label="ref mismatches dominant")], fontsize=8)
    ax.set_title(f"Approach Concentration\n(ref matches dominant in {ref_match_rate:.0%} of problems)")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "diversity_collapse.png"), dpi=150, bbox_inches="tight")

    print("\n" + "=" * 60)
    print("     Analysis 3: Diversity & Distillation Collapse")
    print("=" * 60)
    print(f"Problems: {len(problem_results)}, rollouts per problem: {args.k_rollouts}")
    print(f"Mean unique approaches per problem:  {mean_unique:.1f}")
    print(f"Mean dominant-approach fraction:     {mean_dom_frac:.1%}")
    print(f"Reference matches dominant approach: {ref_match_rate:.1%} of problems")
    print(f"Mean distillation pressure (all):    {mean_pressure:.1%}")
    print(f"Mean distillation pressure (early):  {mean_early_pressure:.1%}")
    print(f"\nApproach distribution: {dict(approach_dist)}")
    if ref_match_rate < 0.5:
        print(f"\n=> Reference approach mismatches the model's preferred approach")
        print(f"   in {1-ref_match_rate:.0%} of problems. OPSD is systematically")
        print(f"   pushing the model away from its dominant reasoning style.")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--dataset", default="siyanzhao/Openthoughts_math_30k_opsd")
    p.add_argument("--split", default="train")
    p.add_argument("--n_problems", type=int, default=10)
    p.add_argument("--k_rollouts", type=int, default=12)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--max_prompt_len", type=int, default=1024)
    p.add_argument("--output_dir", default="analysis/diversity_collapse")
    run(p.parse_args())
