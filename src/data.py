"""Dataset loading and collation utilities."""

from __future__ import annotations

import random
import warnings
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
    name: str = "siyanzhao/Openthoughts_math_30k_opsd",
    n_samples: int = 0,
    seed: int = 42,
) -> list[dict]:
    """Load training problems with solutions.

    For siyanzhao/Openthoughts_math_30k_opsd (OPSD paper dataset), filters to
    correct==True examples only. n_samples=0 means use the full dataset.
    """
    ds = load_dataset(name, split="train")
    ds = ds.shuffle(seed=seed)

    problems = []
    for item in ds:
        # Filter out incorrect solutions when the field is present
        if "correct" in item and str(item["correct"]).lower() != "true":
            continue
        problem = item.get("problem") or item.get("question") or item.get("input", "")
        solution = item.get("solution") or item.get("answer") or item.get("output", "")
        if problem and solution:
            problems.append({"problem": problem, "solution": solution})

    if n_samples > 0:
        problems = problems[:n_samples]
    return problems


def load_aime_dataset(years: list[int] = None) -> list[dict]:
    """Load AIME problems."""
    if years is None:
        years = [2024]

    # Primary sources — known to work reliably
    _PRIMARY = {
        2024: ("HuggingFaceH4/aime_2024", "train", True),
        2025: ("yentinglin/aime_2025",    "train", True),
    }
    _FALLBACK = {
        2024: (f"Maxwell-Jia/AIME_2024", "train", False),
        2025: (f"Maxwell-Jia/AIME_2025", "train", False),
    }

    problems = []
    for year in years:
        ds = None
        for repo, split, trust in [_PRIMARY.get(year, (None, None, None)),
                                    _FALLBACK.get(year, (None, None, None))]:
            if repo is None:
                continue
            try:
                ds = load_dataset(repo, split=split, trust_remote_code=trust)
                break
            except Exception as e:
                warnings.warn(f"AIME {year}: {repo} failed ({e}); trying next source")

        if ds is None:
            warnings.warn(f"All sources for AIME {year} failed; skipping")
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
    except Exception as e:
        warnings.warn(
            f"lighteval/MATH not available ({e}); falling back to hendrycks/competition_math. "
            "This is a random 500-sample subset and is NOT the canonical MATH-500 benchmark — "
            "results will not be comparable to published numbers."
        )
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
            except Exception as e:
                warnings.warn(f"apply_chat_template failed in collator: {e}; using raw content")
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
