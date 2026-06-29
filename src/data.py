"""Dataset loading and collation utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


STUDENT_PROMPT_TEMPLATE = (
    "Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)

TEACHER_PROMPT_TEMPLATE = (
    "Problem: {problem}\n\n"
    "Here is a reference solution:\n{solution}\n\n"
    "Using this as guidance, reason step by step and put your final answer within \\boxed{{}}."
)


def load_train_dataset(
    name: str = "AI-MO/NuminaMath-CoT",
    n_samples: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """Load training problems with solutions."""
    ds = load_dataset(name, split="train")
    ds = ds.shuffle(seed=seed)
    if n_samples > 0:
        ds = ds.select(range(min(n_samples, len(ds))))

    # Normalize column names
    problems = []
    for item in ds:
        problem = item.get("problem") or item.get("question") or item.get("input", "")
        solution = item.get("solution") or item.get("answer") or item.get("output", "")
        if problem and solution:
            problems.append({"problem": problem, "solution": solution})
    return problems


def load_aime_dataset(years: list[int] = None) -> list[dict]:
    """Load AIME problems."""
    if years is None:
        years = [2024]
    problems = []
    for year in years:
        try:
            ds = load_dataset(f"Maxwell-Jia/AIME_{year}", split="train")
        except Exception:
            try:
                ds = load_dataset("lighteval/AIME_2024", split="train")
            except Exception:
                continue
        for item in ds:
            problem = item.get("problem") or item.get("question", "")
            solution = str(item.get("answer") or item.get("solution", ""))
            if problem:
                problems.append({"problem": problem, "solution": solution, "year": year})
    return problems


def load_math500_dataset() -> list[dict]:
    """Load MATH-500 benchmark."""
    try:
        ds = load_dataset("lighteval/MATH", split="test", trust_remote_code=True)
        problems = []
        for item in ds:
            problems.append({
                "problem": item.get("problem", ""),
                "solution": item.get("solution", ""),
            })
        return problems[:500]
    except Exception:
        ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
        problems = []
        for item in ds:
            problems.append({
                "problem": item.get("problem", ""),
                "solution": item.get("solution", ""),
            })
        random.shuffle(problems)
        return problems[:500]


@dataclass
class MathDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_prompt_len: int = 512

    def __call__(self, features: list[dict]) -> dict[str, Any]:
        problems = [f["problem"] for f in features]
        solutions = [f["solution"] for f in features]

        student_texts = [STUDENT_PROMPT_TEMPLATE.format(problem=p) for p in problems]
        teacher_texts = [TEACHER_PROMPT_TEMPLATE.format(problem=p, solution=s) for p, s in zip(problems, solutions)]

        # Apply chat template if tokenizer supports it
        def apply_template(msgs):
            try:
                return self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                return msgs[0]["content"]

        student_prompts = [
            apply_template([{"role": "user", "content": t}]) for t in student_texts
        ]
        teacher_prompts = [
            apply_template([{"role": "user", "content": t}]) for t in teacher_texts
        ]

        student_enc = self.tokenizer(
            student_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
        )
        teacher_enc = self.tokenizer(
            teacher_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_prompt_len,
        )

        return {
            "student_input_ids": student_enc["input_ids"],
            "student_attention_mask": student_enc["attention_mask"],
            "teacher_input_ids": teacher_enc["input_ids"],
            "teacher_attention_mask": teacher_enc["attention_mask"],
            "solutions": solutions,
            "problems": problems,
        }
