"""
Transformers-based evaluation script (no vLLM required).
Slower than vLLM but works without CUDA driver constraints.
Uses tensor parallelism via accelerate for multi-GPU speed.

Reference baselines (OPSD paper, Avg@12):
  Qwen3-1.7B thinking:    AIME24=51.5%  AIME25=36.7%  HMMT25=23.1%
  Qwen3-1.7B nonthinking: AIME24=11.9%  AIME25=9.2%   HMMT25=5.0%
  Qwen3-4B nonthinking:   AIME24=23.1%  AIME25=21.4%  HMMT25=10.8%
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from math_verify import parse, verify
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_CONFIGS = {
    "aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
    "aime25": ("yentinglin/aime_2025", "train", "problem", "answer"),
    "hmmt25": ("MathArena/hmmt_feb_2025", "train", "problem", "answer"),
    "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "solution"),
    "amc23": ("math-ai/amc23", "test", "question", "answer"),
}


def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i, depth, close = idx, 0, None
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                close = i
                break
        i += 1
    if close is None:
        return None
    inner = text[idx:close + 1]
    return inner[7:-1].strip() if inner.startswith("\\boxed{") else None


def grade(pred: str | None, gt: str) -> bool:
    if pred is None:
        return False
    try:
        ps = f"${pred}$" if "$" not in pred else pred
        gs = f"${gt}$" if "$" not in gt else gt
        return verify(parse(gs, fallback_mode="no_fallback"),
                      parse(ps, fallback_mode="no_fallback"),
                      timeout_seconds=5)
    except Exception:
        p, g = pred.replace("$", "").strip(), gt.replace("$", "").strip()
        return p == g


def load_problems(name: str) -> list[dict]:
    hf, split, pk, ak = DATASET_CONFIGS[name]
    trust = name in ("aime25", "hmmt25")
    ds = load_dataset(hf, split=split, trust_remote_code=trust)
    problems = []
    for i, ex in enumerate(ds):
        ans = ex[ak]
        if name == "math500":
            ans = extract_boxed(ans) or ans
        problems.append({"id": ex.get("id", i), "problem": ex[pk], "answer": str(ans)})
    return problems


def run_eval_hf(
    model_path: str,
    dataset_name: str,
    val_n: int = 12,
    enable_thinking: bool = False,
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    batch_size: int = 8,
    output_file: str | None = None,
) -> dict:
    problems = load_problems(dataset_name)
    print(f"Loaded {len(problems)} problems from {dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        use_cache=True,
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    def make_prompt(problem: str) -> str:
        msg = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    results = []
    total_correct = 0
    pass_at_n = 0

    for prob in tqdm(problems, desc=f"Evaluating {dataset_name}"):
        prompt = make_prompt(prob["problem"])
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc.input_ids.to(model.device)
        attn_mask = enc.attention_mask.to(model.device)

        # Expand to val_n
        input_ids = input_ids.expand(val_n, -1)
        attn_mask = attn_mask.expand(val_n, -1)

        generations = []
        with torch.no_grad():
            # Generate in sub-batches to avoid OOM
            for i in range(0, val_n, batch_size):
                batch_ids = input_ids[i:i + batch_size]
                batch_mask = attn_mask[i:i + batch_size]
                out = model.generate(
                    batch_ids,
                    attention_mask=batch_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )
                comp = out[:, batch_ids.shape[1]:]
                texts = tokenizer.batch_decode(comp, skip_special_tokens=True)
                for text in texts:
                    pred = extract_boxed(text)
                    generations.append({
                        "text": text[:500],  # truncate for storage
                        "predicted": pred,
                        "correct": grade(pred, prob["answer"]),
                    })

        n_correct = sum(g["correct"] for g in generations)
        has_correct = n_correct > 0
        total_correct += n_correct
        if has_correct:
            pass_at_n += 1

        results.append({
            "id": prob["id"],
            "problem": prob["problem"],
            "answer": prob["answer"],
            "generations": generations,
            "n_correct": n_correct,
            "pass_at_n": has_correct,
        })

    n_problems = len(problems)
    avg = total_correct / (n_problems * val_n) * 100
    passk = pass_at_n / n_problems * 100

    summary = {
        "model": model_path,
        "dataset": dataset_name,
        "enable_thinking": enable_thinking,
        "val_n": val_n,
        "n_problems": n_problems,
        "avg_at_n": round(avg, 2),
        "pass_at_n_pct": round(passk, 2),
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name.upper()} | {'thinking' if enable_thinking else 'nonthinking'}")
    print(f"Avg@{val_n}:  {avg:.2f}%")
    print(f"Pass@{val_n}: {passk:.2f}%")
    print(f"{'='*60}\n")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="aime24", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--val_n", type=int, default=12)
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Use 16384+ for thinking mode")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="eval/results")
    args = parser.parse_args()

    model_name = Path(args.model).name
    mode = "thinking" if args.thinking else "nonthinking"
    outfile = f"{args.output_dir}/{args.dataset}_{model_name}_{mode}_n{args.val_n}_hf.json"

    run_eval_hf(
        model_path=args.model,
        dataset_name=args.dataset,
        val_n=args.val_n,
        enable_thinking=args.thinking,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        output_file=outfile,
    )


if __name__ == "__main__":
    main()
