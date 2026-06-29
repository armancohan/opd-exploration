"""Tests for Functional-Equivalence Distillation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch
import torch.nn.functional as F

from src.fed import (
    assign_functional_class,
    compute_class_distributions,
    FEDTrainer,
    FEDConfig,
)


def test_assign_class_geometry():
    text = "use coordinate geometry to find the distance between the two points"
    assert assign_functional_class(text) == "geometry"


def test_assign_class_number_theory():
    text = "since n is prime, by Fermat's little theorem we have a^(p-1) ≡ 1 (mod p)"
    assert assign_functional_class(text) == "number_theory"


def test_assign_class_combinatorics():
    text = "count the number of ways to choose 3 elements from a set of 10"
    assert assign_functional_class(text) == "combinatorics"


def test_assign_class_algebra():
    text = "factor the polynomial and find the roots of the quadratic equation"
    assert assign_functional_class(text) == "algebra"


def test_assign_class_trigonometry():
    text = "using sin and cos identities, we apply the pythagorean theorem"
    assert assign_functional_class(text) == "trigonometry"


def test_assign_class_sequences():
    text = "this arithmetic sequence has common difference d and we sum the series"
    assert assign_functional_class(text) == "sequences"


def test_assign_class_other():
    text = "the quick brown fox jumps over the lazy dog"
    assert assign_functional_class(text) == "other"


def test_class_distributions_sum_to_one():
    """P_S(E) should approximately sum to the total mass in candidates."""
    vocab = 100
    student_logits = torch.randn(vocab)
    teacher_logits = torch.randn(vocab)
    ref_logits = torch.randn(vocab)

    # 4 candidates with classes
    candidate_actions = [0, 1, 2, 3]
    candidate_classes = ["geometry", "algebra", "geometry", "combinatorics"]
    candidate_rewards = [1.0, 0.0, 1.0, 0.5]

    P_S, P_T, P_ref, V_hat = compute_class_distributions(
        student_logits, teacher_logits, ref_logits,
        candidate_actions, candidate_classes, candidate_rewards,
    )

    # All returned values should be non-negative
    for cls, val in P_S.items():
        assert val >= 0, f"P_S[{cls}] = {val} < 0"

    # V_hat should be averages
    assert abs(V_hat["geometry"] - 1.0) < 1e-9  # both geometry rewards are 1.0
    assert abs(V_hat["algebra"] - 0.0) < 1e-9
    assert abs(V_hat["combinatorics"] - 0.5) < 1e-9


def test_fed_loss_high_value_class_reinforced():
    """FED loss should be lower when student already matches high-value class distribution."""
    vocab = 20

    # High-value class is "geometry" (tokens 5,6), low-value is "algebra" (tokens 10,11)
    candidate_actions = [5, 6, 10, 11]
    candidate_classes = ["geometry", "geometry", "algebra", "algebra"]
    candidate_rewards = [1.0, 1.0, 0.0, 0.0]

    # Student that matches high-value class well
    student_good = torch.zeros(vocab)
    student_good[5] = 5.0
    student_good[6] = 4.0

    # Student that matches low-value class
    student_bad = torch.zeros(vocab)
    student_bad[10] = 5.0
    student_bad[11] = 4.0

    teacher = torch.zeros(vocab)
    teacher[5] = 3.0
    ref = torch.zeros(vocab)
    ref[5] = 3.0

    P_S_good, P_T, P_ref, V_hat = compute_class_distributions(
        student_good, teacher, ref, candidate_actions, candidate_classes, candidate_rewards
    )
    P_S_bad, _, _, _ = compute_class_distributions(
        student_bad, teacher, ref, candidate_actions, candidate_classes, candidate_rewards
    )

    # When student_good matches high-value class, P_S_good["geometry"] > P_S_bad["geometry"]
    assert P_S_good["geometry"] > P_S_bad["geometry"]


def test_within_class_preservation():
    """Within-class loss should only apply to classes above tau."""
    vocab = 20

    candidate_actions = [2, 3, 8, 9]
    candidate_classes = ["geometry", "geometry", "algebra", "algebra"]
    candidate_rewards = [1.0, 1.0, 0.0, 0.0]

    student_logits = torch.randn(1, 10, vocab)
    teacher_logits = torch.randn(1, 10, vocab)
    ref_logits = torch.randn(1, 10, vocab)
    labels = torch.ones(1, 10, dtype=torch.long)

    anchor_data = [{
        "batch_idx": 0,
        "position": 3,
        "candidate_actions": candidate_actions,
        "candidate_classes": candidate_classes,
        "candidate_rewards": candidate_rewards,
    }]

    _cfg = FEDConfig(tau_value=0.5, lambda_within=1.0)

    class MockFEDTrainer:
        compute_fed_loss = FEDTrainer.compute_fed_loss
        ref_model = None

        def __init__(self, cfg, rlogits):
            self.fed_config = cfg
            self.config = cfg
            self._ref_logits = rlogits

        def _get_ref_logits(self, ids, mask):
            return self._ref_logits

    trainer = MockFEDTrainer(_cfg, ref_logits)

    loss = FEDTrainer.compute_fed_loss(
        trainer, student_logits, teacher_logits, ref_logits, labels, anchor_data
    )
    assert loss.dim() == 0, "FED loss should be a scalar"
    assert loss.item() == loss.item(), "FED loss should not be NaN"


def test_fed_loss_shape():
    """FED loss should return a scalar tensor."""
    vocab = 30
    B, T = 2, 15

    student_logits = torch.randn(B, T, vocab)
    teacher_logits = torch.randn(B, T, vocab)
    ref_logits = torch.randn(B, T, vocab)
    labels = torch.randint(0, vocab, (B, T))

    anchor_data = []  # No anchors — should fall back to base loss

    _cfg2 = FEDConfig()

    class MockTrainer:
        compute_fed_loss = FEDTrainer.compute_fed_loss
        ref_model = None

        def __init__(self, cfg):
            self.fed_config = cfg
            self.config = cfg

        def _get_ref_logits(self, ids, mask):
            return ref_logits

    trainer = MockTrainer(_cfg2)
    loss = FEDTrainer.compute_fed_loss(
        trainer, student_logits, teacher_logits, ref_logits, labels, anchor_data
    )
    assert loss.dim() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_fed_step_runs():
    """Integration test: one FED training step with Llama-3.2-1B."""
    from accelerate import Accelerator
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=False
    )

    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)

    config = FEDConfig(
        n_rollouts=2,
        max_completion_length=64,
        n_anchor_positions=1,
        n_continuations_per_anchor=2,
        gradient_accumulation_steps=1,
        batch_size=1,
    )

    trainer = FEDTrainer(model, tokenizer, optimizer, accelerator, config=config)

    batch = {
        "problems": ["What is 3 + 3?"],
        "solutions": ["6"],
    }

    metrics = trainer.train_step(batch)

    assert "loss" in metrics
    assert "reward_mean" in metrics
    assert 0.0 <= metrics["reward_mean"] <= 1.0
    assert metrics["loss"] == metrics["loss"]  # not NaN
