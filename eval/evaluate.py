"""
Evaluation script for math reasoning models.
Matches OPSD paper evaluation protocol exactly.
Uses vLLM for fast inference + math_verify for grading.

Reference baseline (Qwen3-1.7B, thinking mode, Avg@12):
  AIME 2024: 51.5%  AIME 2025: 36.7%  HMMT25: 23.1%
Reference baseline (Qwen3-1.7B, non-thinking, Avg@12):
  AIME 2024: 11.9%  AIME 2025: 9.2%   HMMT25: 5.0%
Reference baseline (Qwen3-4B, non-thinking, Avg@12):
  AIME 2024: 23.1%  AIME 2025: 21.4%  HMMT25: 10.8%
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from math_verify import parse, verify
from transformers import AutoTokenizer
from tqdm import tqdm


DATASET_CONFIGS = {
    "aime24": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "problem_key": "problem",
        "answer_key": "answer",
        "id_key": "id",
    },
    "aime25": {
        "hf_name": "yentinglin/aime_2025",
        "split": "train",
        "problem_key": "problem",
        "answer_key": "answer",
        "id_key": "problem_idx",
        "trust_remote_code": True,
    },
    "hmmt25": {
        "hf_name": "MathArena/hmmt_feb_2025",
        "split": "train",
        "problem_key": "problem",
        "answer_key": "answer",
        "id_key": "problem_idx",
        "trust_remote_code": True,
    },
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "solution",  # needs boxed extraction
        "id_key": None,
    },
    "amc23": {
        "hf_name": "math-ai/amc23",
        "split": "test",
        "problem_key": "question",
        "answer_key": "answer",
        "id_key": None,
    },
}


def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    depth = 0
    close_idx = None
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
        i += 1
    if close_idx is None:
        return None
    inner = text[idx:close_idx + 1]
    if inner.startswith("\\boxed{") and inner.endswith("}"):
        return inner[7:-1].strip()
    return None


def grade(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    try:
        pred_str = f"${predicted}$" if "$" not in predicted else predicted
        gt_str = f"${ground_truth}$" if "$" not in ground_truth else ground_truth
        return verify(parse(gt_str, fallback_mode="no_fallback"),
                      parse(pred_str, fallback_mode="no_fallback"),
                      timeout_seconds=5)
    except Exception:
        p = predicted.replace("$", "").replace(" ", "").lower()
        g = ground_truth.replace("$", "").replace(" ", "").lower()
        return p == g


def load_eval_dataset(name: str) -> list[dict]:
    cfg = DATASET_CONFIGS[name]
    kwargs = {"trust_remote_code": True} if cfg.get("trust_remote_code") else {}
    ds = load_dataset(cfg["hf_name"], split=cfg["split"], **kwargs)
    problems = []
    for i, ex in enumerate(ds):
        problem = ex[cfg["problem_key"]]
        raw_answer = ex[cfg["answer_key"]]
        # MATH-500 stores full solution; extract boxed answer
        if name == "math500":
            answer = extract_boxed_answer(raw_answer) or raw_answer
        else:
            answer = str(raw_answer)
        qid = ex.get(cfg["id_key"], i) if cfg["id_key"] else i
        problems.append({"id": qid, "problem": problem, "answer": answer})
    return problems


def build_prompts(problems: list[dict], tokenizer, enable_thinking: bool) -> list[str]:
    prompts = []
    for p in problems:
        msg = f"{p['problem']}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompts.append(text)
    return prompts


def run_eval(
    model_path: str,
    dataset_name: str,
    val_n: int = 12,
    enable_thinking: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_new_tokens: int = 16384,
    tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.9,
    lora_path: str | None = None,
    output_file: str | None = None,
) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    problems = load_eval_dataset(dataset_name)
    print(f"Loaded {len(problems)} problems from {dataset_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    max_model_len = 40960 if enable_thinking else 32768

    llm_kwargs = dict(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=max_model_len,
        distributed_executor_backend="mp",
        enforce_eager=False,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)

    prompts = build_prompts(problems, tokenizer, enable_thinking)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=val_n,
    )

    lora_request = None
    if lora_path:
        lora_request = LoRARequest("adapter", 1, lora_path)

    print(f"Generating {val_n} solutions per problem...")
    if lora_request:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=True)
    else:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    results = []
    total_correct = 0
    pass_at_n = 0

    for problem, output in tqdm(zip(problems, outputs), total=len(problems), desc="Grading"):
        gt = problem["answer"]
        generations = []
        for out in output.outputs:
            text = out.text
            pred = extract_boxed_answer(text)
            correct = grade(pred, gt)
            generations.append({
                "text": text,
                "predicted": pred,
                "correct": correct,
            })
        n_correct = sum(g["correct"] for g in generations)
        has_any_correct = n_correct > 0
        total_correct += n_correct
        if has_any_correct:
            pass_at_n += 1

        results.append({
            "id": problem["id"],
            "problem": problem["problem"],
            "answer": gt,
            "generations": generations,
            "n_correct": n_correct,
            "pass_at_n": has_any_correct,
        })

    n_problems = len(problems)
    avg_at_n = total_correct / (n_problems * val_n) * 100
    pass_pct = pass_at_n / n_problems * 100

    summary = {
        "model": model_path,
        "lora": lora_path,
        "dataset": dataset_name,
        "enable_thinking": enable_thinking,
        "val_n": val_n,
        "temperature": temperature,
        "n_problems": n_problems,
        "avg_at_n": round(avg_at_n, 2),
        "pass_at_n_pct": round(pass_pct, 2),
        "total_correct": total_correct,
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name.upper()} | {'thinking' if enable_thinking else 'no-thinking'}")
    print(f"Avg@{val_n}:  {avg_at_n:.2f}%  (target for Qwen3-1.7B base: see README)")
    print(f"Pass@{val_n}: {pass_pct:.2f}%")
    print(f"{'='*60}\n")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--dataset", default="aime24",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--val_n", type=int, default=12)
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=16384,
                        help="16384 for quick eval, 38912 to match OPSD paper exactly")
    parser.add_argument("--tp", type=int, default=4, dest="tensor_parallel_size")
    parser.add_argument("--gpu_util", type=float, default=0.9)
    parser.add_argument("--output_dir", default="eval/results")
    args = parser.parse_args()

    model_name = Path(args.model).name
    mode = "thinking" if args.thinking else "nonthinking"
    lora_tag = f"_lora-{Path(args.lora_path).name}" if args.lora_path else ""
    outfile = (
        f"{args.output_dir}/{args.dataset}_{model_name}{lora_tag}_{mode}_n{args.val_n}.json"
    )

    run_eval(
        model_path=args.model,
        dataset_name=args.dataset,
        val_n=args.val_n,
        enable_thinking=args.thinking,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_util,
        lora_path=args.lora_path,
        output_file=outfile,
    )


if __name__ == "__main__":
    main()
