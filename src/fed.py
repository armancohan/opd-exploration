"""Functional-Equivalence Distillation: class-level diversity preservation."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .opsd_base import OPSDTrainer, OPSDConfig, _compute_jsd
from .verifier import batch_verify

MATH_CLASSES = ["geometry", "trigonometry", "algebra", "number_theory", "combinatorics", "sequences", "other"]

_CLASS_KEYWORDS: dict[str, list[str]] = {
    "geometry": [
        "coordinate", "triangle", "circle", "polygon", "rectangle", "square", "quadrilateral",
        "area", "perimeter", "diagonal", "radius", "diameter", "angle", "parallel", "perpendicular",
        "inscribed", "circumscribed", "distance formula", "midpoint", "slope",
    ],
    "trigonometry": [
        "sin", "cos", "tan", "sine", "cosine", "tangent", "secant", "cosecant", "cotangent",
        "radian", "degree", "arcsin", "arccos", "arctan", "trigonometric", "identity",
        "pythagorean", "law of sines", "law of cosines",
    ],
    "algebra": [
        "polynomial", "factor", "root", "quadratic", "equation", "variable", "expression",
        "expand", "simplify", "solve for", "linear", "system of equations", "inequality",
        "absolute value", "rational", "exponent", "logarithm", "log",
    ],
    "number_theory": [
        "prime", "divisible", "divisor", "gcd", "lcm", "greatest common", "least common",
        "modular", "remainder", "congruent", "fermat", "euler", "digit", "integer",
        "coprime", "floor", "ceiling", "factorization",
    ],
    "combinatorics": [
        "choose", "combination", "permutation", "arrangement", "selection", "count",
        "probability", "pigeonhole", "inclusion-exclusion", "binomial", "ways to",
        "nCr", "nPr", "factorial",
    ],
    "sequences": [
        "sequence", "series", "arithmetic", "geometric", "fibonacci", "recurrence",
        "summation", "sigma", "telescoping", "convergence", "partial sum", "term",
    ],
}


def assign_functional_class(text: str) -> str:
    """Assign a math solution text to a functional strategy class."""
    text_lower = text.lower()
    scores: dict[str, int] = {cls: 0 for cls in MATH_CLASSES}
    for cls, keywords in _CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[cls] += 1
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "other"


def compute_class_distributions(
    student_logits_at_anchor: torch.Tensor,
    teacher_logits_at_anchor: torch.Tensor,
    ref_logits_at_anchor: torch.Tensor,
    candidate_actions: list[int],
    candidate_classes: list[str],
    candidate_rewards: list[float],
) -> tuple[dict, dict, dict, dict]:
    """Compute P_S(E), P_T(E), P_ref(E), V_hat(E) per class."""
    ps = F.softmax(student_logits_at_anchor, dim=-1)
    pt = F.softmax(teacher_logits_at_anchor, dim=-1)
    pr = F.softmax(ref_logits_at_anchor, dim=-1)

    classes = set(candidate_classes)
    P_S: dict[str, float] = {c: 0.0 for c in classes}
    P_T: dict[str, float] = {c: 0.0 for c in classes}
    P_ref: dict[str, float] = {c: 0.0 for c in classes}
    V_sum: dict[str, float] = {c: 0.0 for c in classes}
    V_count: dict[str, int] = {c: 0 for c in classes}

    for action, cls, reward in zip(candidate_actions, candidate_classes, candidate_rewards):
        if action < ps.shape[0]:
            P_S[cls] += ps[action].item()
            P_T[cls] += pt[action].item()
            P_ref[cls] += pr[action].item()
        V_sum[cls] += reward
        V_count[cls] += 1

    V_hat = {c: (V_sum[c] / V_count[c] if V_count[c] > 0 else 0.0) for c in classes}
    return P_S, P_T, P_ref, V_hat


@dataclass
class FEDConfig(OPSDConfig):
    n_anchor_positions: int = 2
    n_continuations_per_anchor: int = 8
    rho: float = 0.5
    beta_fed: float = 0.5
    tau_value: float = 0.3
    lambda_within: float = 0.1
    # Max tokens for anchor continuations; defaults to max_completion_length // 2
    anchor_max_new_tokens: int | None = None
    # Efficient strategy discovery: reuse the already-generated rollouts as the
    # strategy set instead of sampling new anchor continuations. Eliminates all
    # extra generation; ~4-5x step speedup. Set to False for original behavior.
    use_rollout_strategies: bool = True


class FEDTrainer(OPSDTrainer):
    def __init__(self, *args, config: FEDConfig, ref_model=None, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self.fed_config = config
        self.ref_model = ref_model
        if ref_model is None:
            warnings.warn(
                "FEDTrainer: ref_model is None — within-class diversity preservation is disabled. "
                "Pass a frozen copy of the initial model to enable it."
            )

    def _get_ref_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get reference model logits (frozen initial policy)."""
        if self.ref_model is not None:
            with torch.no_grad():
                out = self.ref_model(input_ids=input_ids, attention_mask=attention_mask)
                return out.logits
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

    def _sample_anchor_continuations(
        self,
        prefix_ids: torch.Tensor,
        position: int,
        n_continuations: int,
        solution: str,
    ) -> tuple[list[int], list[str], list[float]]:
        """Sample continuations from anchor position, classify them."""
        device = self.accelerator.device
        unwrapped = self.accelerator.unwrap_model(self.model)

        prefix = prefix_ids[:position + 1].to(device)
        prefixes = prefix.unsqueeze(0).repeat(n_continuations, 1)
        attn = torch.ones_like(prefixes)

        anchor_tokens = (
            self.fed_config.anchor_max_new_tokens
            or self.fed_config.max_completion_length // 2
        )
        with torch.no_grad():
            generated = unwrapped.generate(
                input_ids=prefixes,
                attention_mask=attn,
                max_new_tokens=anchor_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,
            )

        continuation_classes = []
        continuation_rewards = []
        candidate_actions = []

        prompt_len = position + 1
        for i in range(n_continuations):
            next_tok = generated[i, prompt_len].item() if generated[i].shape[0] > prompt_len else 0
            full_text = self.tokenizer.decode(generated[i, prompt_len:], skip_special_tokens=True)
            cls = assign_functional_class(full_text)
            reward = float(batch_verify([full_text], [solution])[0])
            candidate_actions.append(next_tok)
            continuation_classes.append(cls)
            continuation_rewards.append(reward)

        return candidate_actions, continuation_classes, continuation_rewards

    def _rollout_anchor_data(
        self,
        rollout_idx: int,
        anchor_position: int,
        rollouts: list[dict],
        student_ids: torch.Tensor,
        n_rollouts: int,
    ) -> tuple[list[int], list[str], list[float]]:
        """Build anchor candidate data from sibling rollouts (no extra generation).

        Uses the n_rollouts completions already generated for the same problem as
        the strategy set. Each sibling's next token at anchor_position is the
        candidate action; its completion text determines the strategy class.
        """
        problem_idx = rollout_idx // n_rollouts
        sibling_start = problem_idx * n_rollouts
        pad_id = self.tokenizer.pad_token_id or 0

        candidate_actions: list[int] = []
        candidate_classes: list[str] = []
        candidate_rewards: list[float] = []

        for j in range(n_rollouts):
            flat_j = sibling_start + j
            sib = rollouts[flat_j]

            # Token the sibling generated at anchor_position+1 (next-token prediction)
            if anchor_position + 1 < student_ids.shape[1]:
                tok_id = student_ids[flat_j, anchor_position + 1].item()
            else:
                tok_id = pad_id

            # Skip if padding (sibling completion ended before this position)
            if tok_id == 0 or tok_id == pad_id:
                continue

            candidate_actions.append(tok_id)
            candidate_classes.append(assign_functional_class(sib["completion_text"]))
            candidate_rewards.append(sib["reward"])

        return candidate_actions, candidate_classes, candidate_rewards

    def compute_fed_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        labels: torch.Tensor,
        anchor_data: list[dict],
    ) -> torch.Tensor:
        """Compute FED loss: class-level KL + within-class preservation."""
        eps = 1e-8
        rho = self.fed_config.rho
        beta_fed = self.fed_config.beta_fed
        tau = self.fed_config.tau_value
        lambda_within = self.fed_config.lambda_within

        base_loss = _compute_jsd(student_logits, teacher_logits, labels, self.config.beta, self.config.temperature)

        if not anchor_data:
            return base_loss

        fed_losses = []

        for ad in anchor_data:
            i = ad["batch_idx"]
            pos = ad["position"]

            if pos >= student_logits.shape[1] or i >= student_logits.shape[0]:
                continue

            sl_t = student_logits[i, pos]
            tl_t = teacher_logits[i, pos]
            rl_t = ref_logits[i, pos]

            P_S, P_T, P_ref, V_hat = compute_class_distributions(
                sl_t, tl_t, rl_t,
                ad["candidate_actions"],
                ad["candidate_classes"],
                ad["candidate_rewards"],
            )

            classes = list(P_S.keys())
            if len(classes) < 2:
                continue

            # Compute Q(E)
            q_unnorm = {}
            for cls in classes:
                q_unnorm[cls] = (
                    (P_ref.get(cls, eps) + eps) ** rho
                    * (P_T.get(cls, eps) + eps) ** (1 - rho)
                    * math.exp(beta_fed * V_hat.get(cls, 0.0))
                )
            q_total = sum(q_unnorm.values()) + eps
            Q = {cls: q_unnorm[cls] / q_total for cls in classes}

            # Class-level KL(Q || P_S)
            ps_total = sum(P_S.values()) + eps
            p_s_norm = {cls: (P_S[cls] + eps) / ps_total for cls in classes}
            l_class = sum(Q[cls] * (math.log(Q[cls] + eps) - math.log(p_s_norm[cls] + eps)) for cls in classes)

            # Within-class preservation loss for high-value classes
            l_within = 0.0
            if lambda_within > 0:
                for cls in classes:
                    if V_hat.get(cls, 0.0) > tau and P_ref.get(cls, 0.0) > eps:
                        actions_in_class = [
                            a for a, c in zip(ad["candidate_actions"], ad["candidate_classes"]) if c == cls
                        ]
                        if not actions_in_class:
                            continue
                        acts = torch.tensor(actions_in_class, device=student_logits.device)
                        ps_cls = F.softmax(sl_t[acts], dim=-1)
                        pr_cls = F.softmax(rl_t[acts], dim=-1)
                        l_within += F.kl_div(
                            torch.log(ps_cls + eps), torch.log(pr_cls + eps),
                            reduction="sum", log_target=True
                        ).item()

            fed_losses.append(l_class + lambda_within * l_within)

        if not fed_losses:
            return base_loss

        fed_tensor = torch.tensor(
            sum(fed_losses) / len(fed_losses), dtype=base_loss.dtype, device=base_loss.device
        )
        return base_loss + fed_tensor

    def train_step(self, batch: dict) -> dict:
        """FED training step."""
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
        student_logits = student_out.logits[:, :-1, :]

        with torch.no_grad():
            teacher_out = self.model(input_ids=teacher_ids, attention_mask=teacher_mask)
            teacher_logits = teacher_out.logits[:, :-1, :]
            ref_logits = self._get_ref_logits(student_ids, student_mask)[:, :-1, :]

        shifted_labels = labels[:, 1:]
        min_len = min(student_logits.shape[1], teacher_logits.shape[1], ref_logits.shape[1], shifted_labels.shape[1])
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]
        ref_logits = ref_logits[:, :min_len, :]
        shifted_labels = shifted_labels[:, :min_len]

        from .causal_hinge import _token_kl

        anchor_data = []
        n_rollouts = self.config.n_rollouts

        for i, rollout in enumerate(rollouts):
            sl = student_logits[i].detach()
            tl = teacher_logits[i].detach()
            lab = shifted_labels[i]

            # Select anchor positions by top-k KL divergence
            kl = _token_kl(sl, tl)
            label_mask = lab != -100
            kl = kl * label_mask.float()
            n_anchors = min(self.fed_config.n_anchor_positions, kl.shape[0])
            _, anchor_positions = torch.topk(kl, k=n_anchors)

            for pos in anchor_positions.tolist():
                if self.fed_config.use_rollout_strategies:
                    candidate_actions, candidate_classes, candidate_rewards = self._rollout_anchor_data(
                        rollout_idx=i,
                        anchor_position=pos,
                        rollouts=rollouts,
                        student_ids=student_ids,
                        n_rollouts=n_rollouts,
                    )
                else:
                    candidate_actions, candidate_classes, candidate_rewards = self._sample_anchor_continuations(
                        prefix_ids=rollout["input_ids"].cpu(),
                        position=pos,
                        n_continuations=self.fed_config.n_continuations_per_anchor,
                        solution=rollout["solution"],
                    )

                if not candidate_actions:
                    continue

                anchor_data.append({
                    "batch_idx": i,
                    "position": pos,
                    "candidate_actions": candidate_actions,
                    "candidate_classes": candidate_classes,
                    "candidate_rewards": candidate_rewards,
                })

        loss = self.compute_fed_loss(student_logits, teacher_logits, ref_logits, shifted_labels, anchor_data)
        loss = loss / self.config.gradient_accumulation_steps
        self.accelerator.backward(loss)

        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.step += 1
        reward_mean = sum(r["reward"] for r in rollouts) / len(rollouts)
        n_classes = len(set(
            cls for ad in anchor_data for cls in ad["candidate_classes"]
        ))

        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "reward_mean": reward_mean,
            "n_unique_classes": n_classes,
        }
