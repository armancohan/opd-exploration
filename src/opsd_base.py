"""Base OPSD trainer: self-distillation with privileged teacher context."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .data import MathDataCollator, TEACHER_PROMPT_TEMPLATE
from .verifier import batch_verify


@dataclass
class OPSDConfig:
    lr: float = 5e-6
    n_rollouts: int = 4
    batch_size: int = 2
    max_completion_length: int = 1024
    eval_max_new_tokens: int | None = None  # defaults to max_completion_length if None
    beta: float = 0.0          # 0=reverse KL, 1=forward KL, 0.5=JSD
    temperature: float = 1.1
    top_p: float = 0.95
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 2
    max_steps: int = 100
    eval_steps: int = 25
    max_prompt_len: int = 512
    output_dir: str = "outputs/base_opsd"
    wandb_project: str | None = None


def _compute_jsd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None,
    beta: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generalized JSD loss. beta=0 → reverse KL(student||teacher)."""
    sl = student_logits / temperature
    tl = teacher_logits / temperature

    log_ps = F.log_softmax(sl, dim=-1)
    log_pt = F.log_softmax(tl, dim=-1)

    if beta == 0.0:
        # Reverse KL: E_teacher[log p_T - log p_S]
        loss = F.kl_div(log_ps, log_pt, reduction="none", log_target=True)
    elif beta == 1.0:
        loss = F.kl_div(log_pt, log_ps, reduction="none", log_target=True)
    else:
        b = torch.tensor(beta, dtype=log_ps.dtype, device=log_ps.device)
        log_mix = torch.logsumexp(
            torch.stack([log_ps + torch.log1p(-b), log_pt + torch.log(b)]), dim=0
        )
        loss = b * F.kl_div(log_mix, log_pt, reduction="none", log_target=True) + (1 - b) * F.kl_div(
            log_mix, log_ps, reduction="none", log_target=True
        )

    # Sum over vocab dim
    loss = loss.sum(dim=-1)  # [B, T]

    if labels is not None:
        mask = labels != -100
        loss = loss[mask].mean()
    else:
        loss = loss.mean()
    return loss


class OPSDTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        optimizer: Optimizer,
        accelerator: Accelerator,
        config: OPSDConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.config = config
        self.collator = MathDataCollator(tokenizer, max_prompt_len=config.max_prompt_len)
        self.step = 0

    def generate_rollouts(
        self,
        problems: list[str],
        solutions: list[str],
        n_rollouts: int | None = None,
        max_new_tokens: int | None = None,
    ) -> list[dict]:
        """Generate student rollouts and score them."""
        if n_rollouts is None:
            n_rollouts = self.config.n_rollouts
        if max_new_tokens is None:
            max_new_tokens = self.config.max_completion_length

        from .data import STUDENT_PROMPT_TEMPLATE

        results = []
        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(self.model)

        for problem, solution in zip(problems, solutions):
            prompt_text = STUDENT_PROMPT_TEMPLATE.format(problem=problem)
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                warnings.warn(f"apply_chat_template failed for student prompt: {e}; using raw template")

            prompt_ids = self.tokenizer(
                prompt_text, return_tensors="pt", truncation=True, max_length=self.config.max_prompt_len
            ).input_ids.to(device)

            # Repeat for n_rollouts
            prompt_ids_rep = prompt_ids.repeat(n_rollouts, 1)
            attention_mask = torch.ones_like(prompt_ids_rep)

            with torch.no_grad():
                generated = unwrapped.generate(
                    input_ids=prompt_ids_rep,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=self.config.top_p,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    use_cache=True,
                )

            prompt_len = prompt_ids.shape[1]
            for i in range(n_rollouts):
                completion_ids = generated[i, prompt_len:]
                completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
                results.append({
                    "problem": problem,
                    "solution": solution,
                    "prompt_ids": prompt_ids[0],
                    "input_ids": generated[i],
                    "completion_ids": completion_ids,
                    "completion_text": completion_text,
                    "prompt_len": prompt_len,
                })

        # Score all rollouts
        texts = [r["completion_text"] for r in results]
        gts = [r["solution"] for r in results]
        rewards = batch_verify(texts, gts)
        for r, rew in zip(results, rewards):
            r["reward"] = float(rew)

        return results

    def compute_jsd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor | None = None,
        beta: float | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        if beta is None:
            beta = self.config.beta
        if temperature is None:
            temperature = self.config.temperature
        return _compute_jsd(student_logits, teacher_logits, labels, beta, temperature)

    def _build_teacher_inputs(self, rollouts: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build teacher input sequences: [teacher_prompt][student_completion]."""
        device = self.accelerator.device
        teacher_seqs = []
        labels_list = []

        for r in rollouts:
            teacher_prompt = TEACHER_PROMPT_TEMPLATE.format(
                problem=r["problem"], solution=r["solution"]
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

            # Full teacher seq: [teacher_prompt][completion]
            full_seq = torch.cat([teacher_prompt_ids, r["completion_ids"].cpu()])
            teacher_seqs.append(full_seq)

            # Labels: -100 for prompt, token ids for completion
            lab = full_seq.clone()
            lab[:len(teacher_prompt_ids)] = -100
            labels_list.append(lab)

        # Pad sequences
        max_len = max(s.shape[0] for s in teacher_seqs)
        teacher_ids_padded = torch.zeros(len(teacher_seqs), max_len, dtype=torch.long)
        attention_masks = torch.zeros(len(teacher_seqs), max_len, dtype=torch.long)
        labels_padded = torch.full((len(teacher_seqs), max_len), -100, dtype=torch.long)

        for i, (seq, lab) in enumerate(zip(teacher_seqs, labels_list)):
            teacher_ids_padded[i, :len(seq)] = seq
            attention_masks[i, :len(seq)] = 1
            labels_padded[i, :len(lab)] = lab

        return teacher_ids_padded.to(device), attention_masks.to(device), labels_padded.to(device)

    def _build_student_inputs(self, rollouts: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build student input sequences with labels."""
        device = self.accelerator.device
        student_seqs = []
        labels_list = []

        for r in rollouts:
            full_seq = r["input_ids"]
            prompt_len = r["prompt_len"]
            lab = full_seq.clone()
            lab[:prompt_len] = -100
            if self.tokenizer.pad_token_id is not None:
                lab[lab == self.tokenizer.pad_token_id] = -100
            student_seqs.append(full_seq)
            labels_list.append(lab)

        max_len = max(s.shape[0] for s in student_seqs)
        student_ids = torch.zeros(len(student_seqs), max_len, dtype=torch.long)
        attention_masks = torch.zeros(len(student_seqs), max_len, dtype=torch.long)
        labels = torch.full((len(student_seqs), max_len), -100, dtype=torch.long)

        for i, (seq, lab) in enumerate(zip(student_seqs, labels_list)):
            seq = seq.cpu()
            lab = lab.cpu()
            student_ids[i, :len(seq)] = seq
            attention_masks[i, :len(seq)] = 1
            labels[i, :len(lab)] = lab

        return student_ids.to(device), attention_masks.to(device), labels.to(device)

    def train_step(self, batch: dict) -> dict:
        """One training step: generate rollouts, compute loss, update."""
        problems = batch["problems"]
        solutions = batch["solutions"]

        # Generate rollouts (no grad)
        self.model.eval()
        rollouts = self.generate_rollouts(problems, solutions)
        self.model.train()

        if not rollouts:
            warnings.warn(f"Step {self.step}: rollout generation returned empty — skipping update")
            return {"loss": 0.0, "reward_mean": 0.0}

        # Build inputs
        student_ids, student_mask, labels = self._build_student_inputs(rollouts)
        teacher_ids, teacher_mask, teacher_labels = self._build_teacher_inputs(rollouts)

        # Student forward
        student_out = self.model(input_ids=student_ids, attention_mask=student_mask)
        student_logits = student_out.logits[:, :-1, :]  # shift

        # Teacher forward (no grad, same weights different context)
        with torch.no_grad():
            teacher_out = self.model(input_ids=teacher_ids, attention_mask=teacher_mask)
            teacher_logits = teacher_out.logits[:, :-1, :]

        # Align: labels are already shifted (predict next token)
        shifted_labels = labels[:, 1:]

        # Truncate to same length
        min_len = min(student_logits.shape[1], teacher_logits.shape[1], shifted_labels.shape[1])
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]
        shifted_labels = shifted_labels[:, :min_len]

        loss = self.compute_jsd_loss(student_logits, teacher_logits, shifted_labels)
        loss = loss / self.config.gradient_accumulation_steps
        self.accelerator.backward(loss)

        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.step += 1
        reward_mean = sum(r["reward"] for r in rollouts) / len(rollouts)
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "reward_mean": reward_mean,
            "n_rollouts": len(rollouts),
        }

    def evaluate(self, dataset: list[dict], n_rollouts: int = 4) -> dict:
        """Evaluate pass@1 on dataset."""
        self.model.eval()
        all_correct = []
        eval_tokens = self.config.eval_max_new_tokens or self.config.max_completion_length

        for item in dataset:
            rollouts = self.generate_rollouts(
                [item["problem"]], [item["solution"]], n_rollouts=n_rollouts, max_new_tokens=eval_tokens
            )
            rewards = [r["reward"] for r in rollouts]
            # pass@1: mean reward across rollouts
            all_correct.append(sum(rewards) / len(rewards) if rewards else 0.0)

        self.model.train()
        pass1 = sum(all_correct) / len(all_correct) if all_correct else 0.0
        return {"pass@1": pass1, "n_problems": len(all_correct)}
