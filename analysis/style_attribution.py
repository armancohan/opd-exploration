"""Style Attribution Analysis for OPSD.

Tests the hypothesis that OPSD teacher distributions are driven significantly by
solution *style* (algebraic notation, phrasing, structure) beyond what is needed
to simply know the correct answer.

Three teacher variants are compared on the same student rollout prefix:
  - teacher_full:   [problem + full reference solution] → rollout
  - teacher_answer: [problem + "The answer is X"]      → rollout
  - student:        [problem only]                     → rollout

Key metrics:
  JSD(teacher_full, student)   : actual OPSD distillation signal
  JSD(teacher_answer, student) : signal from knowing the answer alone
  JSD(teacher_full, teacher_answer) : style contribution beyond the answer

If the style contribution is a large fraction of the total signal, OPSD is
training the student to mimic solution style, not just find the correct answer.

Also computes a diversity/entropy analysis: does teacher_full consistently have
lower entropy than the student? Lower teacher entropy → OPSD collapses the
student's token distribution (style lock-in).

Run with:
  cd /home/ac3458/code/opsd-experiments
  CUDA_VISIBLE_DEVICES=0 python analysis/style_attribution.py --n_problems 30
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import (
    load_train_dataset,
    TEACHER_PROMPT_TEMPLATE,
    STUDENT_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_answer(solution: str) -> str:
    """Pull the boxed answer from a solution string, or fall back to last 50 chars."""
    m = re.search(r"\\boxed\{([^}]+)\}", solution)
    if m:
        return m.group(1).strip()
    return solution[-50:].strip()


def apply_chat_template(tokenizer, text: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return text


def tokenize(tokenizer, text: str, max_length: int = 1024, device: torch.device = None):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    if device is not None:
        return enc.input_ids.to(device), enc.attention_mask.to(device)
    return enc.input_ids, enc.attention_mask


def get_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Forward pass → logits [1, T, V]."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits  # [1, T, V]


# ---------------------------------------------------------------------------
# Chunked JSD (avoids OOM with 152K vocab)
# ---------------------------------------------------------------------------

def chunked_jsd_per_token(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    chunk_size: int = 64,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute per-token JSD(P || Q) where P=softmax(logits_a), Q=softmax(logits_b).

    Returns a 1D tensor of shape [T] with JSD values in nats.
    logits_a, logits_b: [T, V]
    """
    T, V = logits_a.shape
    jsd_vals = torch.zeros(T, device=logits_a.device, dtype=torch.float32)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        la = logits_a[start:end].float() / temperature
        lb = logits_b[start:end].float() / temperature

        log_pa = F.log_softmax(la, dim=-1)
        log_pb = F.log_softmax(lb, dim=-1)

        # M = 0.5*(P + Q) in log space via logsumexp
        log_m = torch.logaddexp(log_pa - 0.693147, log_pb - 0.693147)

        kl_pm = (log_pa.exp() * (log_pa - log_m)).sum(dim=-1)
        kl_qm = (log_pb.exp() * (log_pb - log_m)).sum(dim=-1)
        jsd_vals[start:end] = 0.5 * (kl_pm + kl_qm)

    return jsd_vals  # [T]


def chunked_entropy_per_token(
    logits: torch.Tensor,
    chunk_size: int = 64,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token entropy in nats. logits: [T, V] → [T]"""
    T, V = logits.shape
    ent = torch.zeros(T, device=logits.device, dtype=torch.float32)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        l = logits[start:end].float() / temperature
        log_p = F.log_softmax(l, dim=-1)
        ent[start:end] = -(log_p.exp() * log_p).sum(dim=-1)
    return ent


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_problem(
    model,
    tokenizer,
    problem: str,
    solution: str,
    max_new_tokens: int,
    max_prompt_len: int,
    device: torch.device,
    temperature: float = 0.8,
) -> dict | None:
    """Run the three-way attribution analysis for one problem."""

    answer = extract_answer(solution)

    # Build prompt texts
    teacher_full_text = apply_chat_template(
        tokenizer, TEACHER_PROMPT_TEMPLATE.format(problem=problem, solution=solution)
    )
    teacher_answer_text = apply_chat_template(
        tokenizer,
        f"Problem: {problem}\n\nThe answer is: {answer}\n\n"
        "Reason step by step and put your final answer within \\boxed{}.",
    )
    student_text = apply_chat_template(
        tokenizer, STUDENT_PROMPT_TEMPLATE.format(problem=problem)
    )

    # Tokenize prompts
    student_ids, student_mask = tokenize(tokenizer, student_text, max_prompt_len, device)
    prompt_len = student_ids.shape[1]

    # Generate ONE student rollout
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=student_ids,
            attention_mask=student_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True,
        )
    completion_ids = gen_out[0, prompt_len:]  # [T_comp]

    if completion_ids.shape[0] == 0:
        return None

    completion_ids_cpu = completion_ids.cpu()

    # Build full sequences: [prompt + completion]
    def build_full_seq(prompt_text: str, comp_ids: torch.Tensor):
        p_ids, p_mask = tokenize(tokenizer, prompt_text, max_prompt_len, device)
        p_len = p_ids.shape[1]
        full = torch.cat([p_ids[0], comp_ids.to(device)]).unsqueeze(0)  # [1, P+C]
        full_mask = torch.ones(1, full.shape[1], dtype=torch.long, device=device)
        return full, full_mask, p_len

    full_tf, mask_tf, p_len_tf = build_full_seq(teacher_full_text, completion_ids_cpu)
    full_ta, mask_ta, p_len_ta = build_full_seq(teacher_answer_text, completion_ids_cpu)
    full_st, mask_st, p_len_st = build_full_seq(student_text, completion_ids_cpu)

    # Forward passes — only over the completion tokens (no grad)
    def get_completion_logits(full_ids, full_mask, p_len):
        logits = get_logits(model, full_ids, full_mask)  # [1, L, V]
        # logits[t] predicts token t+1; completion starts at p_len
        # so logits for completion tokens are at positions [p_len-1 : p_len-1+T_comp]
        T_comp = completion_ids_cpu.shape[0]
        comp_logits = logits[0, p_len - 1: p_len - 1 + T_comp, :]  # [T_comp, V]
        return comp_logits

    logits_tf = get_completion_logits(full_tf, mask_tf, p_len_tf)
    logits_ta = get_completion_logits(full_ta, mask_ta, p_len_ta)
    logits_st = get_completion_logits(full_st, mask_st, p_len_st)

    T = min(logits_tf.shape[0], logits_ta.shape[0], logits_st.shape[0])
    if T == 0:
        return None

    logits_tf = logits_tf[:T]
    logits_ta = logits_ta[:T]
    logits_st = logits_st[:T]

    # JSD computations — all in float32 to avoid underflow
    jsd_tf_st = chunked_jsd_per_token(logits_tf, logits_st)    # OPSD signal
    jsd_ta_st = chunked_jsd_per_token(logits_ta, logits_st)    # answer-only signal
    jsd_tf_ta = chunked_jsd_per_token(logits_tf, logits_ta)    # style contribution

    # Entropy analysis
    ent_tf = chunked_entropy_per_token(logits_tf)
    ent_st = chunked_entropy_per_token(logits_st)

    return {
        "jsd_tf_st": jsd_tf_st.cpu().numpy(),   # [T]
        "jsd_ta_st": jsd_ta_st.cpu().numpy(),
        "jsd_tf_ta": jsd_tf_ta.cpu().numpy(),
        "ent_tf":    ent_tf.cpu().numpy(),
        "ent_st":    ent_st.cpu().numpy(),
        "n_tokens":  T,
    }


def main():
    parser = argparse.ArgumentParser(description="OPSD Style Attribution Analysis")
    parser.add_argument("--n_problems", type=int, default=30, help="Number of problems to analyze")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Student rollout length")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output_dir", type=str, default="analysis")
    parser.add_argument("--dataset", type=str, default="siyanzhao/Openthoughts_math_30k_opsd")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use (e.g. 'train', 'test')")
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model.eval()

    print(f"Loading dataset: {args.dataset} (split={args.dataset_split})")
    data = load_train_dataset(args.dataset, n_samples=args.n_problems, split=args.dataset_split)
    print(f"Loaded {len(data)} problems (requested {args.n_problems})")

    all_results = []
    for idx, item in enumerate(data):
        print(f"  Problem {idx+1}/{len(data)}...", end=" ", flush=True)
        try:
            result = analyze_problem(
                model=model,
                tokenizer=tokenizer,
                problem=item["problem"],
                solution=item["solution"],
                max_new_tokens=args.max_new_tokens,
                max_prompt_len=args.max_prompt_len,
                device=device,
                temperature=args.temperature,
            )
            if result is not None:
                all_results.append(result)
                print(f"OK ({result['n_tokens']} tokens)")
            else:
                print("SKIPPED (empty completion)")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_results:
        print("No results collected — exiting.")
        return

    # Aggregate flat arrays across all problems
    def concat_metric(key):
        return np.concatenate([r[key] for r in all_results])

    jsd_tf_st_all = concat_metric("jsd_tf_st")
    jsd_ta_st_all = concat_metric("jsd_ta_st")
    jsd_tf_ta_all = concat_metric("jsd_tf_ta")
    ent_tf_all    = concat_metric("ent_tf")
    ent_st_all    = concat_metric("ent_st")

    mean_tf_st = float(np.mean(jsd_tf_st_all))
    std_tf_st  = float(np.std(jsd_tf_st_all))
    mean_ta_st = float(np.mean(jsd_ta_st_all))
    std_ta_st  = float(np.std(jsd_ta_st_all))
    mean_tf_ta = float(np.mean(jsd_tf_ta_all))
    std_tf_ta  = float(np.std(jsd_tf_ta_all))

    style_fraction = mean_tf_ta / max(mean_tf_st, 1e-9)
    teacher_lower_entropy_frac = float(np.mean(ent_tf_all < ent_st_all))

    # Per-problem style fractions (for scatter/histogram)
    per_problem_style_frac = []
    for r in all_results:
        m_full = float(np.mean(r["jsd_tf_st"]))
        m_style = float(np.mean(r["jsd_tf_ta"]))
        per_problem_style_frac.append(m_style / max(m_full, 1e-9))

    # Position-resolved curves: align all sequences to the same length by
    # truncating to the shortest, then average per position.
    min_T = min(r["n_tokens"] for r in all_results)
    pos_tf_st = np.stack([r["jsd_tf_st"][:min_T] for r in all_results]).mean(axis=0)
    pos_ta_st = np.stack([r["jsd_ta_st"][:min_T] for r in all_results]).mean(axis=0)
    pos_tf_ta = np.stack([r["jsd_tf_ta"][:min_T] for r in all_results]).mean(axis=0)
    pos_ent_tf = np.stack([r["ent_tf"][:min_T] for r in all_results]).mean(axis=0)
    pos_ent_st = np.stack([r["ent_st"][:min_T] for r in all_results]).mean(axis=0)
    positions = np.arange(min_T)

    # -----------------------------------------------------------------------
    # Save raw results
    # -----------------------------------------------------------------------
    output_json = {
        "n_problems": len(all_results),
        "n_tokens_total": int(len(jsd_tf_st_all)),
        "jsd_teacher_full_vs_student":      {"mean": mean_tf_st, "std": std_tf_st},
        "jsd_teacher_answer_vs_student":    {"mean": mean_ta_st, "std": std_ta_st},
        "jsd_teacher_full_vs_teacher_answer": {"mean": mean_tf_ta, "std": std_tf_ta},
        "style_fraction": style_fraction,
        "teacher_lower_entropy_fraction": teacher_lower_entropy_frac,
        "per_problem_style_fractions": per_problem_style_frac,
    }
    json_path = os.path.join(args.output_dir, "style_attribution_results.json")
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved results to {json_path}")

    # -----------------------------------------------------------------------
    # Plot — 2x3 grid
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("OPSD Style Attribution Analysis", fontsize=14, fontweight="bold")

    C_FULL   = "steelblue"
    C_ANS    = "orange"
    C_STYLE  = "green"
    C_ENT_T  = "tomato"
    C_ENT_S  = "steelblue"

    # ---- Row 0, Col 0: JSD distribution histograms ----
    ax = axes[0, 0]
    vmax = max(jsd_tf_st_all.max(), jsd_ta_st_all.max(), jsd_tf_ta_all.max())
    bins = np.linspace(0, vmax, 60)
    ax.hist(jsd_tf_st_all, bins=bins, alpha=0.55, label="JSD(full, student) — OPSD signal", color=C_FULL)
    ax.hist(jsd_ta_st_all, bins=bins, alpha=0.55, label="JSD(answer, student) — answer-only", color=C_ANS)
    ax.hist(jsd_tf_ta_all, bins=bins, alpha=0.55, label="JSD(full, answer) — style contribution", color=C_STYLE)
    ax.set_xlabel("JSD (nats)")
    ax.set_ylabel("Token count")
    ax.set_title("Per-Token JSD Distributions")
    ax.legend(fontsize=8)

    # ---- Row 0, Col 1: Mean JSD bar chart ----
    ax = axes[0, 1]
    bar_labels = ["JSD(full,\nstudent)\nOPSD signal", "JSD(answer,\nstudent)\nanswer-only", "JSD(full,\nanswer)\nstyle contrib"]
    means = [mean_tf_st, mean_ta_st, mean_tf_ta]
    stds  = [std_tf_st,  std_ta_st,  std_tf_ta]
    colors = [C_FULL, C_ANS, C_STYLE]
    bars = ax.bar(bar_labels, means, yerr=stds, color=colors, alpha=0.85, capsize=5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + max(stds) * 0.05,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mean JSD (nats)")
    ax.set_title(f"Mean JSD per Condition\nStyle fraction = {style_fraction:.1%}")

    # ---- Row 0, Col 2: Entropy distributions ----
    ax = axes[0, 2]
    ax.hist(ent_st_all,  bins=60, alpha=0.6, label="Student entropy", color=C_ENT_S)
    ax.hist(ent_tf_all,  bins=60, alpha=0.6, label="Teacher (full) entropy", color=C_ENT_T)
    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Token count")
    ax.set_title(
        f"Entropy: Teacher vs Student\n"
        f"Teacher more peaked at {teacher_lower_entropy_frac:.1%} of positions"
    )
    ax.legend(fontsize=9)

    # ---- Row 1, Col 0: Position-resolved JSD curves ----
    ax = axes[1, 0]
    # Smooth with a short rolling window for readability
    def smooth(arr, w=8):
        return np.convolve(arr, np.ones(w) / w, mode="valid")
    sm_pos = np.arange(len(smooth(pos_tf_st)))
    ax.plot(sm_pos, smooth(pos_tf_st), color=C_FULL,  lw=1.8, label="JSD(full, student)")
    ax.plot(sm_pos, smooth(pos_ta_st), color=C_ANS,   lw=1.8, label="JSD(answer, student)")
    ax.plot(sm_pos, smooth(pos_tf_ta), color=C_STYLE,  lw=1.8, label="JSD(full, answer) = style")
    ax.axvline(x=0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Token position in rollout")
    ax.set_ylabel("Mean JSD (nats)")
    ax.set_title("JSD vs Token Position\n(smoothed, avg across problems)")
    ax.legend(fontsize=8)

    # ---- Row 1, Col 1: Position-resolved entropy ----
    ax = axes[1, 1]
    ax.plot(np.arange(len(smooth(pos_ent_st))), smooth(pos_ent_st), color=C_ENT_S, lw=1.8, label="Student entropy")
    ax.plot(np.arange(len(smooth(pos_ent_tf))), smooth(pos_ent_tf), color=C_ENT_T, lw=1.8, label="Teacher entropy")
    ax.fill_between(
        np.arange(len(smooth(pos_ent_st))),
        smooth(pos_ent_tf), smooth(pos_ent_st),
        where=smooth(pos_ent_tf) < smooth(pos_ent_st),
        alpha=0.15, color=C_ENT_T, label="Style lock-in zone"
    )
    ax.set_xlabel("Token position in rollout")
    ax.set_ylabel("Mean entropy (nats)")
    ax.set_title("Entropy vs Token Position\n(teacher lower = style collapse pressure)")
    ax.legend(fontsize=8)

    # ---- Row 1, Col 2: Per-problem style fraction scatter ----
    ax = axes[1, 2]
    per_prob_full  = [float(np.mean(r["jsd_tf_st"])) for r in all_results]
    per_prob_style = [float(np.mean(r["jsd_tf_ta"])) for r in all_results]
    sc = ax.scatter(per_prob_full, per_prob_style, c=per_problem_style_frac,
                    cmap="RdYlGn_r", vmin=0, vmax=1, s=60, edgecolors="gray", lw=0.4)
    # y=x line: style contribution equals total signal
    lim_max = max(max(per_prob_full), max(per_prob_style)) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, alpha=0.4, label="style = total signal")
    plt.colorbar(sc, ax=ax, label="Style fraction")
    ax.set_xlabel("Mean JSD(full, student) — OPSD signal")
    ax.set_ylabel("Mean JSD(full, answer) — style contribution")
    ax.set_title("Per-Problem: Style vs Total Signal\n(color = style fraction; diagonal = 100%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "style_attribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    # -----------------------------------------------------------------------
    # Text summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("           Style Attribution Analysis")
    print("=" * 60)
    print(f"Problems analyzed:  {len(all_results)}")
    print(f"Total token positions: {len(jsd_tf_st_all):,}")
    print()
    print(f"Mean JSD(teacher_full, student):    {mean_tf_st:.4f} ± {std_tf_st:.4f}  [OPSD signal]")
    print(f"Mean JSD(teacher_answer, student):  {mean_ta_st:.4f} ± {std_ta_st:.4f}  [answer-only signal]")
    print(f"Mean JSD(teacher_full, teacher_answer): {mean_tf_ta:.4f} ± {std_tf_ta:.4f}  [style contribution]")
    print()
    print(f"Style fraction (style / total signal): {style_fraction:.1%}")
    print()
    print(f"Interpretation: {style_fraction:.1%} of OPSD's distillation pressure at the")
    print(f"per-token level comes from solution style beyond the answer content.")
    print()
    print(f"Diversity pressure: Teacher (full-context) entropy is LOWER than student")
    print(f"entropy at {teacher_lower_entropy_frac:.1%} of positions.")
    if teacher_lower_entropy_frac > 0.55:
        print("=> Strong style lock-in pressure: OPSD is collapsing the student's")
        print("   distribution toward the reference solution's stylistic choices.")
    elif teacher_lower_entropy_frac > 0.45:
        print("=> Moderate style pressure: the teacher is often more peaked but")
        print("   the effect is not overwhelming.")
    else:
        print("=> Weak style pressure: the teacher is frequently LESS confident")
        print("   than the student (counter-intuitive, investigate further).")
    print("=" * 60)


if __name__ == "__main__":
    main()
