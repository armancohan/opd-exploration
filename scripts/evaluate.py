"""Evaluation script for trained models."""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_aime_dataset, load_math500_dataset, STUDENT_PROMPT_TEMPLATE
from src.verifier import batch_verify


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="aime2024", choices=["aime2024", "aime2025", "math500"])
    p.add_argument("--n_rollouts", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--gpu_id", type=int, default=0)
    return p.parse_args()


def pass_at_k(n: int, c: int, k: int) -> float:
    """Estimate pass@k given n total rollouts with c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    ).to(device)
    model.eval()

    if args.dataset == "aime2024":
        problems = load_aime_dataset([2024])
    elif args.dataset == "aime2025":
        problems = load_aime_dataset([2025])
    else:
        problems = load_math500_dataset()

    print(f"Evaluating {len(problems)} problems with {args.n_rollouts} rollouts each...")

    per_problem = []

    for i, item in enumerate(problems):
        prompt_text = STUDENT_PROMPT_TEMPLATE.format(problem=item["problem"])
        try:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(device)
        prompt_ids_rep = prompt_ids.repeat(args.n_rollouts, 1)
        attn = torch.ones_like(prompt_ids_rep)

        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids_rep,
                attention_mask=attn,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        prompt_len = prompt_ids.shape[1]
        completions = [
            tokenizer.decode(generated[j, prompt_len:], skip_special_tokens=True)
            for j in range(args.n_rollouts)
        ]
        rewards = batch_verify(completions, [item["solution"]] * args.n_rollouts)
        n_correct = sum(rewards)

        problem_result = {
            "problem": item["problem"][:100] + "...",
            "n_rollouts": args.n_rollouts,
            "n_correct": n_correct,
            "pass@1": pass_at_k(args.n_rollouts, n_correct, 1),
            "pass@4": pass_at_k(args.n_rollouts, n_correct, 4),
            "pass@8": pass_at_k(args.n_rollouts, n_correct, 8),
        }
        per_problem.append(problem_result)

        if (i + 1) % 5 == 0:
            current_pass1 = sum(p["pass@1"] for p in per_problem) / len(per_problem)
            print(f"  [{i+1}/{len(problems)}] running pass@1={current_pass1:.3f}")

    # Aggregate
    pass1 = sum(p["pass@1"] for p in per_problem) / len(per_problem)
    pass4 = sum(p["pass@4"] for p in per_problem) / len(per_problem)
    pass8 = sum(p["pass@8"] for p in per_problem) / len(per_problem)

    summary = {
        "model": args.model_path,
        "dataset": args.dataset,
        "n_problems": len(per_problem),
        "n_rollouts": args.n_rollouts,
        "pass@1": pass1,
        "pass@4": pass4,
        "pass@8": pass8,
        "per_problem": per_problem,
    }

    print(f"\n=== Results: {args.dataset} ===")
    print(f"pass@1: {pass1:.3f}")
    print(f"pass@4: {pass4:.3f}")
    print(f"pass@8: {pass8:.3f}")

    output_file = args.output_file or f"eval_{os.path.basename(args.model_path)}_{args.dataset}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
