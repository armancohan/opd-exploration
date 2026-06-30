"""Measure real-world reward-rate impact of the verifier fix.

Generates batched rollouts on real problems and scores them with both the new
(math_verify) backend and the old sympy heuristic, so we can quantify how many
correct answers the old verifier was rejecting as false negatives.
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data import load_train_dataset, STUDENT_PROMPT_TEMPLATE as PROMPT_TEMPLATE
from src.verifier import verify_math_answer, extract_boxed_answer
import src.verifier as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n_problems", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=3072)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    ds = load_train_dataset(n_samples=args.n_problems)

    n_new = n_old = n_total = 0
    disagree = []
    for d in ds:
        problem, solution = d["problem"], d["solution"]
        prompt = PROMPT_TEMPLATE.format(problem=problem)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok([text] * args.k, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.95, pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        comp = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for c in comp:
            n_total += 1
            new_ok = verify_math_answer(c, solution)
            old_ok = V._verify_heuristic(c, solution)
            n_new += new_ok
            n_old += old_ok
            if new_ok and not old_ok:
                disagree.append((extract_boxed_answer(c), extract_boxed_answer(solution)))
        print(f"  problem done | running: new={n_new}/{n_total} old={n_old}/{n_total}", flush=True)

    print("\n==== RESULTS ====")
    print(f"rollouts scored: {n_total}")
    print(f"NEW (math_verify) correct: {n_new}/{n_total} = {100*n_new/n_total:.1f}%")
    print(f"OLD (heuristic)  correct: {n_old}/{n_total} = {100*n_old/n_total:.1f}%")
    print(f"recovered by fix (new-correct, old-rejected): {len(disagree)}")
    for pred, gt in disagree[:15]:
        print(f"   pred={pred!r:30}  gt={gt!r}")


if __name__ == "__main__":
    main()
