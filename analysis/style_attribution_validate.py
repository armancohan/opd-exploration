"""Validation of the OPSD style-attribution metric.

Answers three questions a skeptical reader will ask about

    style_fraction = JSD(full, answer) / JSD(full, student)

(1) DIRECT REPORT + NON-ADDITIVITY.
    Report JSD(answer, student) on its own, and check how non-additive the three
    JSDs are: JSD(full,answer) + JSD(answer,student)  vs  JSD(full,student).
    Because sqrt(JSD) is a true metric (Jensen-Shannon distance) but JSD itself
    is not, we report the "fraction" in BOTH raw-JSD space and metric (sqrt) space
    so the reader can see how much the squared-distance framing distorts it.

(2) STRONGER ANSWER-CONDITIONER.
    The original answer baseline is an ad-hoc prompt ("The answer is: X"), which
    may UNDER-condition the model and thereby inflate the style fraction. We add a
    second baseline that is structurally identical to the full-solution prompt
    (same "Here is a reference solution:" framing) but whose solution body is just
    the boxed answer with no reasoning. If the style fraction survives this
    stronger conditioner, the claim is robust; if it collapses, the original
    number was an artifact of weak conditioning.

      teacher_full    : [problem + full reference solution]          (template)
      teacher_minimal : [problem + "The final answer is \\boxed{X}."] (SAME template)
      teacher_answer  : [problem + "The answer is: X" + reason...]   (original, weak)
      student         : [problem only]

(3) CONCRETE EXAMPLES.
    For a few problems we dump the highest-divergence rollout positions with the
    decoded context, the student's sampled token, and the top-5 next-token
    predictions under each conditioning — so a reader can SEE that the divergence
    sits on stylistic/discourse tokens rather than on the answer.

Run (uses the most-free GPU so it can sit alongside training):
  cd /home/ac3458/code/opsd-experiments
  CUDA_VISIBLE_DEVICES=3 python analysis/style_attribution_validate.py \
      --n_problems 30 --out analysis/style_validation
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import (
    load_train_dataset,
    TEACHER_PROMPT_TEMPLATE,
    STUDENT_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers (shared with style_attribution.py)
# ---------------------------------------------------------------------------

def extract_answer(solution: str) -> str:
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


def tokenize(tokenizer, text, max_length, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def get_logits(model, input_ids, attention_mask):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits  # [1, T, V]


def chunked_jsd_per_token(logits_a, logits_b, chunk_size=64, temperature=1.0):
    """Per-token JSD(softmax(a) || softmax(b)) in nats. logits: [T, V] -> [T]."""
    T, V = logits_a.shape
    jsd = torch.zeros(T, device=logits_a.device, dtype=torch.float32)
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        la = logits_a[s:e].float() / temperature
        lb = logits_b[s:e].float() / temperature
        log_pa = F.log_softmax(la, dim=-1)
        log_pb = F.log_softmax(lb, dim=-1)
        log_m = torch.logaddexp(log_pa - 0.693147, log_pb - 0.693147)
        kl_pm = (log_pa.exp() * (log_pa - log_m)).sum(dim=-1)
        kl_qm = (log_pb.exp() * (log_pb - log_m)).sum(dim=-1)
        jsd[s:e] = 0.5 * (kl_pm + kl_qm)
    return jsd


# ---------------------------------------------------------------------------
# Per-problem analysis
# ---------------------------------------------------------------------------

def analyze_problem(model, tokenizer, problem, solution, max_new_tokens,
                    max_prompt_len, device, temperature, capture_examples=False,
                    top_k=5):
    answer = extract_answer(solution)

    # ---- four conditionings ----
    teacher_full_text = apply_chat_template(
        tokenizer, TEACHER_PROMPT_TEMPLATE.format(problem=problem, solution=solution))
    # STRONG conditioner: identical template, reasoning stripped to just the answer
    teacher_minimal_text = apply_chat_template(
        tokenizer, TEACHER_PROMPT_TEMPLATE.format(
            problem=problem, solution=f"The final answer is \\boxed{{{answer}}}."))
    # WEAK conditioner: the original ad-hoc answer prompt (reproduces prior numbers)
    teacher_answer_text = apply_chat_template(
        tokenizer,
        f"Problem: {problem}\n\nThe answer is: {answer}\n\n"
        "Reason step by step and put your final answer within \\boxed{}.")
    student_text = apply_chat_template(
        tokenizer, STUDENT_PROMPT_TEMPLATE.format(problem=problem))

    # ---- one student rollout ----
    student_ids, student_mask = tokenize(tokenizer, student_text, max_prompt_len, device)
    prompt_len = student_ids.shape[1]
    with torch.no_grad():
        gen = model.generate(
            input_ids=student_ids, attention_mask=student_mask,
            max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=True, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True)
    completion_ids = gen[0, prompt_len:].cpu()
    if completion_ids.shape[0] == 0:
        return None

    def comp_logits(prompt_text):
        p_ids, _ = tokenize(tokenizer, prompt_text, max_prompt_len, device)
        p_len = p_ids.shape[1]
        full = torch.cat([p_ids[0], completion_ids.to(device)]).unsqueeze(0)
        mask = torch.ones(1, full.shape[1], dtype=torch.long, device=device)
        logits = get_logits(model, full, mask)
        Tc = completion_ids.shape[0]
        return logits[0, p_len - 1: p_len - 1 + Tc, :]

    lf = comp_logits(teacher_full_text)
    lm = comp_logits(teacher_minimal_text)
    la = comp_logits(teacher_answer_text)
    ls = comp_logits(student_text)

    T = min(lf.shape[0], lm.shape[0], la.shape[0], ls.shape[0])
    lf, lm, la, ls = lf[:T], lm[:T], la[:T], ls[:T]

    out = {
        "jsd_full_student":    chunked_jsd_per_token(lf, ls).cpu().numpy(),  # total OPSD signal
        "jsd_answer_student":  chunked_jsd_per_token(la, ls).cpu().numpy(),  # weak answer-only
        "jsd_minimal_student": chunked_jsd_per_token(lm, ls).cpu().numpy(),  # strong answer-only
        "jsd_full_answer":     chunked_jsd_per_token(lf, la).cpu().numpy(),  # style num (weak)
        "jsd_full_minimal":    chunked_jsd_per_token(lf, lm).cpu().numpy(),  # style num (strong)
        "n_tokens": T,
    }

    # ---- concrete token-level examples ----
    if capture_examples:
        jsd_fs = out["jsd_full_student"]
        order = np.argsort(-jsd_fs)  # highest-divergence positions first
        examples = []
        ce = completion_ids[:T]
        for pos in order[:6]:
            pos = int(pos)
            ctx_ids = ce[max(0, pos - 30):pos]
            context = tokenizer.decode(ctx_ids, skip_special_tokens=True)
            sampled = tokenizer.decode(ce[pos:pos + 1], skip_special_tokens=False)

            def topk(logits_row):
                p = F.softmax(logits_row.float(), dim=-1)
                pv, pi = torch.topk(p, top_k)
                return [(tokenizer.decode([int(i)], skip_special_tokens=False), round(float(v), 3))
                        for v, i in zip(pv, pi)]

            examples.append({
                "position": pos,
                "context_tail": context[-200:],
                "student_sampled_token": sampled,
                "jsd_full_student": round(float(out["jsd_full_student"][pos]), 4),
                "jsd_answer_student": round(float(out["jsd_answer_student"][pos]), 4),
                "jsd_full_answer": round(float(out["jsd_full_answer"][pos]), 4),
                "top5_student":  topk(ls[pos]),
                "top5_answer":   topk(la[pos]),
                "top5_full":     topk(lf[pos]),
            })
        out["examples"] = {
            "problem": problem[:300],
            "answer": answer,
            "rollout_preview": tokenizer.decode(ce, skip_special_tokens=True)[:600],
            "positions": examples,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_problems", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    ap.add_argument("--dataset", type=str, default="siyanzhao/Openthoughts_math_30k_opsd")
    ap.add_argument("--dataset_split", type=str, default="train")
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--n_examples", type=int, default=3, help="problems to dump token examples for")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default="analysis/style_validation")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager")
    model.eval()

    data = load_train_dataset(args.dataset, n_samples=args.n_problems, split=args.dataset_split)
    print(f"Loaded {len(data)} problems")

    results, example_dumps = [], []
    for i, item in enumerate(data):
        print(f"  {i+1}/{len(data)}", end=" ", flush=True)
        try:
            r = analyze_problem(
                model, tok, item["problem"], item["solution"],
                args.max_new_tokens, args.max_prompt_len, device, args.temperature,
                capture_examples=(i < args.n_examples))
            if r is None:
                print("skip"); continue
            if "examples" in r:
                example_dumps.append(r.pop("examples"))
            results.append(r)
            print(f"ok ({r['n_tokens']}t)")
        except Exception as e:
            print(f"ERR {e}")

    if not results:
        print("no results"); return

    def cat(k):
        return np.concatenate([r[k] for r in results])

    fs = cat("jsd_full_student")     # total signal
    as_ = cat("jsd_answer_student")  # weak answer-only
    ms = cat("jsd_minimal_student")  # strong answer-only
    fa = cat("jsd_full_answer")      # style num (weak)
    fm = cat("jsd_full_minimal")     # style num (strong)

    def stats(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x)), "median": float(np.median(x))}

    m_fs, m_as, m_ms, m_fa, m_fm = [float(np.mean(x)) for x in (fs, as_, ms, fa, fm)]

    # ---- metric-space (sqrt JSD = Jensen-Shannon distance) ----
    # clamp tiny-negative underflow to 0 before sqrt
    def msqrt(x):
        return float(np.mean(np.sqrt(np.maximum(x, 0.0))))
    md_fs, md_as, md_ms, md_fa, md_fm = [msqrt(x) for x in (fs, as_, ms, fa, fm)]

    # ---- non-additivity gaps, computed on AGGREGATE means ----
    # (per-token ratios blow up on near-zero denominators, so we use means).
    # raw JSD is NOT a metric: gap may be either sign. metric (sqrt) obeys triangle: gap>=0.
    raw_gap = (m_fa + m_as - m_fs) / max(m_fs, 1e-9)
    raw_gap_min = (m_fm + m_ms - m_fs) / max(m_fs, 1e-9)
    metric_gap = (md_fa + md_as - md_fs) / max(md_fs, 1e-9)

    summary = {
        "n_problems": len(results),
        "n_tokens": int(len(fs)),
        "jsd": {
            "full_vs_student   (total OPSD signal)": stats(fs),
            "answer_vs_student (weak answer-only)":  stats(as_),
            "minimal_vs_student(strong answer-only)":stats(ms),
            "full_vs_answer    (style num, weak)":   stats(fa),
            "full_vs_minimal   (style num, strong)": stats(fm),
        },
        "style_fraction_raw_JSD": {
            "weak_conditioner   (full,answer)/(full,student)":  m_fa / max(m_fs, 1e-9),
            "strong_conditioner (full,minimal)/(full,student)": m_fm / max(m_fs, 1e-9),
        },
        "style_fraction_metric_sqrtJSD": {
            "weak_conditioner":   md_fa / max(md_fs, 1e-9),
            "strong_conditioner": md_fm / max(md_fs, 1e-9),
        },
        "answer_only_share_raw_JSD": {
            "weak   (answer,student)/(full,student)":  m_as / max(m_fs, 1e-9),
            "strong (minimal,student)/(full,student)": m_ms / max(m_fs, 1e-9),
        },
        "non_additivity_gap": {
            "_note": "(JSD(f,a)+JSD(a,s)-JSD(f,s))/JSD(f,s); raw JSD can be <0, metric sqrt must be >=0",
            "raw_JSD_weak":    raw_gap,
            "raw_JSD_strong":  raw_gap_min,
            "metric_sqrt_weak": metric_gap,
        },
    }

    with open(os.path.join(args.out, "validation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "token_examples.json"), "w") as f:
        json.dump(example_dumps, f, indent=2)

    print("\n" + "=" * 70)
    print("  Style-attribution validation")
    print("=" * 70)
    print(f"problems={len(results)}  tokens={len(fs):,}\n")
    print(f"  JSD(full,    student)  = {m_fs:.4f}   [total OPSD signal]")
    print(f"  JSD(answer,  student)  = {m_as:.4f}   [weak answer-only]")
    print(f"  JSD(minimal, student)  = {m_ms:.4f}   [strong answer-only]")
    print(f"  JSD(full,    answer )  = {m_fa:.4f}   [style numerator, weak]")
    print(f"  JSD(full,    minimal)  = {m_fm:.4f}   [style numerator, strong]\n")
    print(f"  style fraction (raw JSD):    weak={m_fa/max(m_fs,1e-9):.1%}   strong={m_fm/max(m_fs,1e-9):.1%}")
    print(f"  style fraction (sqrt metric):weak={md_fa/max(md_fs,1e-9):.1%}   strong={md_fm/max(md_fs,1e-9):.1%}")
    print(f"  answer-only share (raw):     weak={m_as/max(m_fs,1e-9):.1%}   strong={m_ms/max(m_fs,1e-9):.1%}\n")
    print(f"  non-additivity gap raw(weak)={raw_gap:+.1%}  raw(strong)={raw_gap_min:+.1%}  metric={metric_gap:+.1%}")
    print("=" * 70)
    print(f"saved -> {args.out}/validation_results.json , token_examples.json")


if __name__ == "__main__":
    main()
