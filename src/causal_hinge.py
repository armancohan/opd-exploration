"""Causal-Hinge OPD: verifier-calibrated token selection for on-policy distillation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .opsd_base import OPSDTrainer, OPSDConfig, _compute_jsd
from .verifier import batch_verify


@dataclass
class CHConfig(OPSDConfig):
    n_probe_positions: int = 2
    n_candidates: int = 4
    n_probes: int = 2
    max_probe_tokens: int = 150
    tau_benefit: float = 0.0
    # TopK_S ∪ TopK_T can produce up to n_candidates*2 unique tokens; this cap
    # prevents an unbounded candidate set when distributions are very spread.
    max_candidates_multiplier: int = 2
    # Efficient benefit estimation: use token-level KL(p_T||p_S) as benefit proxy
    # instead of probe sampling. Eliminates extra generation; ~2x step speedup.
    # Set to False to use the original branch-probe method.
    use_logit_benefit: bool = True


def _token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Per-token KL(teacher||student) divergence, shape [T]."""
    log_ps = F.log_softmax(student_logits, dim=-1)
    log_pt = F.log_softmax(teacher_logits, dim=-1)
    return F.kl_div(log_ps, log_pt, reduction="none", log_target=True).sum(dim=-1)


class CausalHingeOPSD(OPSDTrainer):
    def __init__(self, *args, config: CHConfig, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.ch_config = config

    def select_probe_positions(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        n_positions: int | None = None,
        labels: torch.Tensor | None = None,
    ) -> list[int]:
        """Select top-n positions by KL divergence (restricted to labeled positions)."""
        if n_positions is None:
            n_positions = self.ch_config.n_probe_positions

        kl = _token_kl(student_logits, teacher_logits)  # [T]

        if labels is not None:
            mask = labels != -100
            kl = kl * mask.float()

        topk = min(n_positions, kl.shape[0])
        _, indices = torch.topk(kl, k=topk)
        return sorted(indices.tolist())

    def probe_branch_values(
        self,
        prefix_ids: torch.Tensor,
        position: int,
        student_logits_t: torch.Tensor,
        teacher_logits_t: torch.Tensor,
        solution: str,
        n_candidates: int | None = None,
        n_probes: int | None = None,
        max_cont_tokens: int | None = None,
    ) -> float:
        """Compute the signed hinge benefit B_t at a single position via probe sampling."""
        if n_candidates is None:
            n_candidates = self.ch_config.n_candidates
        if n_probes is None:
            n_probes = self.ch_config.n_probes
        if max_cont_tokens is None:
            max_cont_tokens = self.ch_config.max_probe_tokens

        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(self.model)

        ps = F.softmax(student_logits_t, dim=-1)
        pt = F.softmax(teacher_logits_t, dim=-1)

        # Candidate set: TopK_S ∪ TopK_T ∪ {sampled_token}
        top_s = torch.topk(ps, k=min(n_candidates, ps.shape[0])).indices
        top_t = torch.topk(pt, k=min(n_candidates, pt.shape[0])).indices
        sampled_token = prefix_ids[position].unsqueeze(0).to(device) if position < prefix_ids.shape[0] else top_s[:1]
        candidates = torch.unique(torch.cat([top_s, top_t, sampled_token]))
        candidates = candidates[:n_candidates * self.ch_config.max_candidates_multiplier]

        # Build forced prefixes: [original_prefix_up_to_t][candidate_token]
        prefix_up_to_t = prefix_ids[:position + 1].clone()
        # Replace the token at position with each candidate
        forced_prefixes = []
        for cand in candidates:
            fp = prefix_up_to_t.clone()
            fp[position] = cand
            forced_prefixes.append(fp)

        if not forced_prefixes:
            return 0.0

        # Batch all (prefix, probe) combinations
        probe_prefixes = []
        probe_cands = []
        for cand, fp in zip(candidates, forced_prefixes):
            for _ in range(n_probes):
                probe_prefixes.append(fp)
                probe_cands.append(cand)

        # Left-pad to same length
        max_len = max(p.shape[0] for p in probe_prefixes)
        padded = torch.zeros(len(probe_prefixes), max_len, dtype=torch.long, device=device)
        attn = torch.zeros_like(padded)
        for i, p in enumerate(probe_prefixes):
            p = p.to(device)
            padded[i, max_len - p.shape[0]:] = p
            attn[i, max_len - p.shape[0]:] = 1

        with torch.no_grad():
            generated = unwrapped.generate(
                input_ids=padded,
                attention_mask=attn,
                max_new_tokens=max_cont_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode and verify
        completions = []
        for i in range(len(probe_prefixes)):
            gen_tokens = generated[i, max_len:]
            completions.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))

        rewards = batch_verify(completions, [solution] * len(completions))

        # Aggregate V_hat per candidate
        v_hat: dict[int, list[float]] = {}
        for i, (cand, reward) in enumerate(zip(probe_cands, rewards)):
            cid = cand.item()
            if cid not in v_hat:
                v_hat[cid] = []
            v_hat[cid].append(float(reward))

        v_mean = {cid: sum(vs) / len(vs) for cid, vs in v_hat.items()}

        # B_t = sum_{a in C_t} [p_T(a) - p_S(a)] * V_hat(a)
        B_t = 0.0
        for cand in candidates:
            cid = cand.item()
            if cid in v_mean:
                B_t += (pt[cid].item() - ps[cid].item()) * v_mean[cid]

        return B_t

    def _build_logit_hinge_mask(
        self,
        student_logits_full: torch.Tensor,
        teacher_logits_full: torch.Tensor,
        shifted_labels: torch.Tensor,
        rollouts: list[dict],
    ) -> tuple[torch.Tensor, int, int]:
        """Efficient hinge mask: gate on token-level KL(p_T||p_S) > tau.

        Returns (hinge_mask [B,T], total_probed, total_positive).
        """
        hinge_mask = torch.zeros_like(shifted_labels, dtype=torch.bool)
        total_probed = 0
        total_positive = 0
        tau = self.ch_config.tau_benefit

        for i in range(len(rollouts)):
            sl = student_logits_full[i].detach()
            tl = teacher_logits_full[i].detach()
            lab = shifted_labels[i]

            label_mask = lab != -100
            kl = _token_kl(sl, tl)  # [T], KL(teacher||student) per token
            hinge_mask[i] = (kl > tau) & label_mask

            total_probed += label_mask.sum().item()
            total_positive += hinge_mask[i].sum().item()

        return hinge_mask, total_probed, total_positive

    def _build_probe_hinge_mask(
        self,
        student_logits_full: torch.Tensor,
        teacher_logits_full: torch.Tensor,
        shifted_labels: torch.Tensor,
        rollouts: list[dict],
    ) -> tuple[torch.Tensor, int, int]:
        """Original hinge mask via branch probe sampling (expensive)."""
        hinge_mask = torch.zeros_like(shifted_labels, dtype=torch.bool)
        total_probed = 0
        total_positive = 0

        for i, rollout in enumerate(rollouts):
            sl = student_logits_full[i].detach()
            tl = teacher_logits_full[i].detach()
            lab = shifted_labels[i]

            positions = self.select_probe_positions(sl, tl, labels=lab)
            total_probed += len(positions)

            for pos in positions:
                if pos >= rollout["input_ids"].shape[0]:
                    continue
                B_t = self.probe_branch_values(
                    prefix_ids=rollout["input_ids"].cpu(),
                    position=pos,
                    student_logits_t=sl[pos],
                    teacher_logits_t=tl[pos],
                    solution=rollout["solution"],
                )
                if B_t > self.ch_config.tau_benefit:
                    hinge_mask[i, pos] = True
                    total_positive += 1

        return hinge_mask, total_probed, total_positive

    def train_step(self, batch: dict) -> dict:
        """CH-OPD training step with hinge masking."""
        problems = batch["problems"]
        solutions = batch["solutions"]

        self.model.eval()
        rollouts = self.generate_rollouts(problems, solutions)
        self.model.train()

        if not rollouts:
            warnings.warn(f"Step {self.step}: rollout generation returned empty — skipping update")
            return {"loss": 0.0, "reward_mean": 0.0}

        student_ids, student_mask, labels = self._build_student_inputs(rollouts)
        teacher_ids, teacher_mask, _ = self._build_teacher_inputs(rollouts)

        student_out = self.model(input_ids=student_ids, attention_mask=student_mask)
        student_logits_full = student_out.logits[:, :-1, :]

        with torch.no_grad():
            teacher_out = self.model(input_ids=teacher_ids, attention_mask=teacher_mask)
            teacher_logits_full = teacher_out.logits[:, :-1, :]

        shifted_labels = labels[:, 1:]
        min_len = min(student_logits_full.shape[1], teacher_logits_full.shape[1], shifted_labels.shape[1])
        student_logits_full = student_logits_full[:, :min_len, :]
        teacher_logits_full = teacher_logits_full[:, :min_len, :]
        shifted_labels = shifted_labels[:, :min_len]

        if self.ch_config.use_logit_benefit:
            hinge_mask, total_probed, total_positive = self._build_logit_hinge_mask(
                student_logits_full, teacher_logits_full, shifted_labels, rollouts
            )
        else:
            hinge_mask, total_probed, total_positive = self._build_probe_hinge_mask(
                student_logits_full, teacher_logits_full, shifted_labels, rollouts
            )

        # If no hinge positions found, fall back to standard loss on all labeled positions
        if not hinge_mask.any():
            effective_labels = shifted_labels
        else:
            effective_labels = shifted_labels.clone()
            effective_labels[~hinge_mask] = -100

        loss = self.compute_jsd_loss(student_logits_full, teacher_logits_full, effective_labels)
        del teacher_logits_full
        torch.cuda.empty_cache()
        loss = loss / self.config.gradient_accumulation_steps
        self.accelerator.backward(loss)

        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.step += 1
        reward_mean = sum(r["reward"] for r in rollouts) / len(rollouts)
        hinge_rate = total_positive / max(total_probed, 1)

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "reward_mean": reward_mean,
            "hinge_positive_rate": hinge_rate,
            "n_probed": total_probed,
        }
