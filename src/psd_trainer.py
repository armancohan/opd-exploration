"""Progressive Self-Distillation (PSD) trainer.

PSD extends OPSD by maintaining a per-problem buffer of verified-correct
self-generated rollouts. The teacher context transitions from the external
reference solution (early training, when the buffer is empty) to the model's
own successful solutions (later training), narrowing the teacher-student
realizability gap over time.

Distinction from SSOPD (arXiv 2605.17497): SSOPD uses the current-batch
shortest correct rollout as teacher context, discarded at step end.
PSD accumulates solutions across steps in a persistent per-problem buffer,
enabling problem-specific teacher evolution rather than batch-local selection.
"""

from __future__ import annotations

import json
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .opsd_base import OPSDTrainer, OPSDConfig, _compute_jsd
from .data import TEACHER_PROMPT_TEMPLATE


@dataclass
class PSDConfig(OPSDConfig):
    # Buffer capacity per problem (number of correct rollouts to keep)
    buffer_size: int = 5
    # How to select from the buffer: "random" | "latest" | "shortest"
    buffer_strategy: str = "random"
    # Path to persist/restore buffer across runs (None = no persistence)
    buffer_path: str | None = None
    # Log buffer fill rate every N steps
    buffer_log_steps: int = 10


class PSDTrainer(OPSDTrainer):
    """Progressive Self-Distillation trainer.

    Identical to base OPSD except the teacher context for each problem is
    drawn from a persistent buffer of the model's own previously-correct
    rollouts. Falls back to the reference solution when the buffer is empty.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: Optimizer,
        accelerator: Accelerator,
        config: PSDConfig,
    ):
        super().__init__(model, tokenizer, optimizer, accelerator, config)
        self.config: PSDConfig = config
        # problem_text -> list of correct completion texts
        self.buffer: dict[str, list[str]] = defaultdict(list)
        self._buffer_hits = 0
        self._buffer_queries = 0

        if config.buffer_path and os.path.exists(config.buffer_path):
            self._load_buffer(config.buffer_path)

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def _get_teacher_solution(self, problem: str, reference_solution: str) -> tuple[str, bool]:
        """Return (solution_text, from_buffer).

        Uses a buffered self-generated solution when available;
        falls back to the reference solution otherwise.
        """
        self._buffer_queries += 1
        solutions = self.buffer.get(problem)
        if not solutions:
            return reference_solution, False

        self._buffer_hits += 1
        if self.config.buffer_strategy == "latest":
            return solutions[-1], True
        elif self.config.buffer_strategy == "shortest":
            return min(solutions, key=len), True
        else:
            return random.choice(solutions), True

    def _update_buffer(self, rollouts: list[dict]) -> int:
        """Add correct rollouts to the per-problem buffer. Returns number added."""
        added = 0
        for r in rollouts:
            if r["reward"] > 0.5:
                problem = r["problem"]
                self.buffer[problem].append(r["completion_text"])
                if len(self.buffer[problem]) > self.config.buffer_size:
                    self.buffer[problem] = self.buffer[problem][-self.config.buffer_size:]
                added += 1
        return added

    def _buffer_fill_rate(self) -> float:
        """Fraction of buffer slots that contain at least one solution."""
        if not self.buffer:
            return 0.0
        return sum(1 for v in self.buffer.values() if v) / max(len(self.buffer), 1)

    def _save_buffer(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(self.buffer), f)

    def _load_buffer(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.buffer = defaultdict(list, data)
        n_problems = len(self.buffer)
        n_solutions = sum(len(v) for v in self.buffer.values())
        if self.accelerator.is_main_process:
            print(f"PSD: loaded buffer with {n_solutions} solutions across {n_problems} problems from {path}")

    # ------------------------------------------------------------------
    # Override _build_teacher_inputs to use buffer solutions
    # ------------------------------------------------------------------

    def _build_teacher_inputs(self, rollouts: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build teacher sequences using buffer solutions where available."""
        device = self.accelerator.device
        teacher_seqs = []
        labels_list = []

        for r in rollouts:
            teacher_solution, from_buffer = self._get_teacher_solution(
                r["problem"], r["solution"]
            )

            teacher_prompt = TEACHER_PROMPT_TEMPLATE.format(
                problem=r["problem"], solution=teacher_solution
            )
            try:
                teacher_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": teacher_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                warnings.warn(f"apply_chat_template failed for teacher prompt: {e}; using raw template")

            teacher_prompt_ids = self.tokenizer(
                teacher_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_prompt_len,
            ).input_ids[0]

            full_seq = torch.cat([teacher_prompt_ids, r["completion_ids"].cpu()])
            teacher_seqs.append(full_seq)

            lab = full_seq.clone()
            lab[:len(teacher_prompt_ids)] = -100
            labels_list.append(lab)

        max_len = max(s.shape[0] for s in teacher_seqs)
        teacher_ids_padded = torch.zeros(len(teacher_seqs), max_len, dtype=torch.long)
        attention_masks = torch.zeros(len(teacher_seqs), max_len, dtype=torch.long)
        labels_padded = torch.full((len(teacher_seqs), max_len), -100, dtype=torch.long)

        for i, (seq, lab) in enumerate(zip(teacher_seqs, labels_list)):
            teacher_ids_padded[i, :len(seq)] = seq
            attention_masks[i, :len(seq)] = 1
            labels_padded[i, :len(lab)] = lab

        return teacher_ids_padded.to(device), attention_masks.to(device), labels_padded.to(device)

    # ------------------------------------------------------------------
    # Override train_step to update buffer after each step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        metrics = super().train_step(batch)

        # After parent step (which called generate_rollouts internally), we don't
        # have direct access to rollouts — re-use the fact that generate_rollouts
        # stored rewards. Instead, we hook into the base class by overriding
        # generate_rollouts to capture rollouts for buffer updates.
        # (See _last_rollouts set in generate_rollouts below.)
        if hasattr(self, "_last_rollouts") and self._last_rollouts:
            added = self._update_buffer(self._last_rollouts)
            metrics["buffer_added"] = added
            self._last_rollouts = None

        fill = self._buffer_fill_rate()
        hit_rate = self._buffer_hits / max(self._buffer_queries, 1)
        metrics["buffer_fill"] = fill
        metrics["buffer_hit_rate"] = hit_rate

        if (
            self.accelerator.is_main_process
            and self.step % self.config.buffer_log_steps == 0
        ):
            n_problems = len(self.buffer)
            n_solutions = sum(len(v) for v in self.buffer.values())
            print(
                f"  [PSD buffer] {n_solutions} solutions across {n_problems} problems "
                f"| hit_rate={hit_rate:.1%} | fill={fill:.1%}"
            )

        if self.config.buffer_path and self.step % 50 == 0 and self.accelerator.is_main_process:
            self._save_buffer(self.config.buffer_path)

        return metrics

    def generate_rollouts(self, problems, solutions, n_rollouts=None, max_new_tokens=None):
        rollouts = super().generate_rollouts(problems, solutions, n_rollouts, max_new_tokens)
        self._last_rollouts = rollouts
        return rollouts
