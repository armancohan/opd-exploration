"""
vLLM-based evaluation matching the OPSD paper protocol exactly.

Reference: Zhao et al. 2026 (OPSD), evaluate_math.py
Key parameters:
  nonthinking: temperature=1.0, top_p=0.8, max_tokens=32768
  thinking:    temperature=0.6, top_p=0.95, max_tokens=40960
  val_n=12, tensor_parallel_size=4, enforce_eager=True

Baselines (Avg@12, nonthinking):
  Qwen3-1.7B:  AIME24=11.9%  AIME25=9.2%   HMMT25=5.0%
  Qwen3-4B:    AIME24=23.1%  AIME25=21.4%  HMMT25=10.8%

Launch:
  VLLM_USE_V1=0 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    python eval/evaluate.py --model Qwen/Qwen3-1.7B --dataset aime24
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify
from tqdm import tqdm
from vllm import LLM, SamplingParams


DATASET_CONFIGS = {
    "aime24":  ("HuggingFaceH4/aime_2024",    "train", "problem", "answer",   False),
    "aime25":  ("yentinglin/aime_2025",        "train", "problem", "answer",   True),
    "hmmt25":  ("MathArena/hmmt_feb_2025",     "train", "problem", "answer",   True),
    "math500": ("HuggingFaceH4/MATH-500",      "test",  "problem", "solution", False),
    "amc23":   ("math-ai/amc23",               "test",  "question", "answer",  False),
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
    inner = text[idx: close + 1]
    return inner[7:-1].strip() if inner.startswith("\\boxed{") else None


def grade(pred: str | None, gt: str) -> bool:
    if pred is None:
        return False
    try:
        ps = f"${pred}$" if "$" not in pred else pred
        gs = f"${gt}$"   if "$" not in gt   else gt
        return verify(parse(gs, fallback_mode="no_fallback"),
                      parse(ps, fallback_mode="no_fallback"),
                      timeout_seconds=5)
    except Exception:
        return pred.replace("$", "").strip() == gt.replace("$", "").strip()


def load_problems(name: str) -> list[dict]:
    hf, split, pk, ak, trust = DATASET_CONFIGS[name]
    ds = load_dataset(hf, split=split, trust_remote_code=trust)
    problems = []
    for i, ex in enumerate(ds):
        ans = ex[ak]
        if name == "math500":
            ans = extract_boxed(ans) or ans
        problems.append({"id": ex.get("id", i), "problem": ex[pk], "answer": str(ans)})
    return problems


def run_eval(
    model_path: str,
    dataset_name: str,
    val_n: int = 12,
    enable_thinking: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.9,
    output_file: str | None = None,
) -> dict:
    # Match OPSD paper defaults exactly
    if enable_thinking:
        temperature = temperature or 0.6
        top_p       = top_p       or 0.95
        max_tokens  = max_tokens  or 40960
    else:
        temperature = temperature or 1.0
        top_p       = top_p       or 0.8
        max_tokens  = max_tokens  or 32768

    problems = load_problems(dataset_name)
    print(f"Loaded {len(problems)} problems from {dataset_name}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def make_prompt(problem: str) -> str:
        msg = f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_tokens + 512,  # prompt headroom
        gpu_memory_utilization=gpu_memory_utilization,
        distributed_executor_backend="mp",
        enforce_eager=True,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=-1,
        min_p=0.0,
        presence_penalty=0.0,
        max_tokens=max_tokens,
        n=val_n,
    )

    prompts = [make_prompt(p["problem"]) for p in problems]
    print(f"Generating {val_n} samples × {len(problems)} problems "
          f"(max_tokens={max_tokens}, tp={tensor_parallel_size})...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    total_correct = 0
    pass_at_n = 0

    for prob, out in tqdm(zip(problems, outputs), total=len(problems)):
        generations = []
        for o in out.outputs:
            text = o.text
            # Strip <think>...</think> for answer extraction in thinking mode
            if enable_thinking and "</think>" in text:
                answer_text = text[text.rfind("</think>") + len("</think>"):]
            else:
                answer_text = text
            pred = extract_boxed(answer_text)
            correct = grade(pred, prob["answer"])
            generations.append({"text": text[:2000], "predicted": pred, "correct": correct})

        n_correct = sum(g["correct"] for g in generations)
        total_correct += n_correct
        if n_correct > 0:
            pass_at_n += 1

        results.append({
            "id": prob["id"],
            "problem": prob["problem"],
            "answer": prob["answer"],
            "generations": generations,
            "n_correct": n_correct,
            "pass_at_n": n_correct > 0,
        })

    n_problems = len(problems)
    avg   = total_correct / (n_problems * val_n) * 100
    passk = pass_at_n / n_problems * 100

    summary = {
        "model": model_path,
        "dataset": dataset_name,
        "enable_thinking": enable_thinking,
        "val_n": val_n,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
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
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--tp", type=int, default=4, dest="tensor_parallel_size",
                        help="tensor parallel size (default: 4)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--output_dir", default="eval/results")
    args = parser.parse_args()

    model_name = Path(args.model).name
    mode = "thinking" if args.thinking else "nonthinking"
    outfile = f"{args.output_dir}/{args.dataset}_{model_name}_{mode}_n{args.val_n}.json"

    run_eval(
        model_path=args.model,
        dataset_name=args.dataset,
        val_n=args.val_n,
        enable_thinking=args.thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        output_file=outfile,
    )


if __name__ == "__main__":
    main()
