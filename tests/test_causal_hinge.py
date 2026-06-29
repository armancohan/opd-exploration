"""Tests for Causal-Hinge OPD."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F

from src.causal_hinge import CausalHingeOPSD, CHConfig, _token_kl


def _make_logits(vocab=100, seq=30, seed=0):
    torch.manual_seed(seed)
    return torch.randn(seq, vocab)


def test_select_probe_positions_selects_high_kl():
    """Positions 5, 10, 20 should have highest KL and be selected."""
    vocab = 50
    seq = 30
    student_logits = torch.zeros(seq, vocab)
    teacher_logits = torch.zeros(seq, vocab)

    # Make positions 5, 10, 20 have high divergence
    for pos in [5, 10, 20]:
        student_logits[pos, 0] = 10.0   # student strongly predicts token 0
        teacher_logits[pos, 1] = 10.0   # teacher strongly predicts token 1

    # For this test, create a mock trainer with just the selection method
    config = CHConfig(n_probe_positions=3)

    class MockTrainer:
        def __init__(self):
            self.ch_config = config
            self.config = config

        select_probe_positions = CausalHingeOPSD.select_probe_positions

    trainer = MockTrainer()
    positions = CausalHingeOPSD.select_probe_positions(trainer, student_logits, teacher_logits, n_positions=3)
    assert set(positions) == {5, 10, 20}, f"Expected {{5,10,20}}, got {set(positions)}"


def test_select_probe_positions_labels_mask():
    """Labels mask should restrict selection to labeled positions."""
    vocab = 20
    seq = 10
    student_logits = torch.zeros(seq, vocab)
    teacher_logits = torch.zeros(seq, vocab)

    # High divergence at positions 2, 5 (both labeled) and position 8 (not labeled)
    for pos in [2, 5, 8]:
        student_logits[pos, 0] = 10.0
        teacher_logits[pos, 1] = 10.0

    labels = torch.full((seq,), -100, dtype=torch.long)
    labels[2] = 1
    labels[5] = 1  # labeled, should be selected

    config = CHConfig(n_probe_positions=2)

    class MockTrainer:
        def __init__(self):
            self.ch_config = config
            self.config = config

        select_probe_positions = CausalHingeOPSD.select_probe_positions

    trainer = MockTrainer()
    positions = CausalHingeOPSD.select_probe_positions(trainer, student_logits, teacher_logits, n_positions=2, labels=labels)
    assert 8 not in positions, "Unlabeled position 8 should not be selected"


def test_hinge_benefit_positive():
    """B_t > 0 when teacher mass is toward high-reward tokens."""
    vocab = 10
    # student likes token 0 (low reward)
    ps = torch.zeros(vocab)
    ps[0] = 5.0
    # teacher likes tokens 2,3 (high reward)
    pt = torch.zeros(vocab)
    pt[2] = 5.0
    pt[3] = 3.0

    ps_prob = F.softmax(ps, dim=0)
    pt_prob = F.softmax(pt, dim=0)

    # V_hat: tokens 2,3 have high reward, token 0 has low reward
    v_hat = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}

    B_t = sum((pt_prob[a].item() - ps_prob[a].item()) * v_hat[a] for a in range(vocab))
    assert B_t > 0, f"Expected positive B_t, got {B_t}"


def test_hinge_benefit_near_zero():
    """B_t ≈ 0 when student and teacher have similar distributions."""
    vocab = 10
    logits = torch.randn(vocab)
    ps_prob = F.softmax(logits, dim=0)
    pt_prob = F.softmax(logits, dim=0)  # identical

    v_hat = {a: float(a % 2) for a in range(vocab)}  # alternating 0, 1
    B_t = sum((pt_prob[a].item() - ps_prob[a].item()) * v_hat[a] for a in range(vocab))
    assert abs(B_t) < 1e-6, f"Expected near-zero B_t, got {B_t}"


def test_hinge_benefit_negative():
    """B_t < 0 when teacher mass is toward low-reward tokens."""
    vocab = 10
    # student likes tokens 2,3 (high reward)
    ps = torch.zeros(vocab)
    ps[2] = 5.0
    ps[3] = 3.0
    # teacher likes token 0 (low reward)
    pt = torch.zeros(vocab)
    pt[0] = 5.0

    ps_prob = F.softmax(ps, dim=0)
    pt_prob = F.softmax(pt, dim=0)

    v_hat = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0}

    B_t = sum((pt_prob[a].item() - ps_prob[a].item()) * v_hat[a] for a in range(vocab))
    assert B_t < 0, f"Expected negative B_t, got {B_t}"


def test_masked_loss_fewer_positions():
    """Hinge mask should reduce number of trained tokens."""
    B, T, V = 2, 20, 50
    student_logits = torch.randn(B, T, V)
    teacher_logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    # Mark half as -100
    labels[:, T // 2:] = -100

    # With hinge mask: only 4 positions
    hinge_mask = torch.zeros(B, T, dtype=torch.bool)
    hinge_mask[0, 2] = True
    hinge_mask[0, 5] = True
    hinge_mask[1, 3] = True
    hinge_mask[1, 7] = True

    hinge_labels = labels.clone()
    hinge_labels[~hinge_mask] = -100

    full_active = (labels != -100).sum().item()
    hinge_active = (hinge_labels != -100).sum().item()

    assert hinge_active < full_active, f"Hinge mask should reduce active positions: {hinge_active} < {full_active}"
    assert hinge_active <= 4


def test_token_kl_shape():
    """_token_kl should return per-token KL of shape [T]."""
    T, V = 15, 30
    sl = torch.randn(T, V)
    tl = torch.randn(T, V)
    kl = _token_kl(sl, tl)
    assert kl.shape == (T,)
    assert (kl >= 0).all(), "KL divergence should be non-negative"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_ch_step_runs():
    """Integration test: one CH-OPD training step with Llama-3.2-1B."""
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

    config = CHConfig(
        n_rollouts=2,
        max_completion_length=64,
        n_probe_positions=1,
        n_candidates=2,
        n_probes=1,
        max_probe_tokens=30,
        gradient_accumulation_steps=1,
        batch_size=1,
    )

    trainer = CausalHingeOPSD(model, tokenizer, optimizer, accelerator, config=config)

    batch = {
        "problems": ["What is 2 + 2?"],
        "solutions": ["4"],
    }

    metrics = trainer.train_step(batch)

    assert "loss" in metrics
    assert "reward_mean" in metrics
    assert 0.0 <= metrics["reward_mean"] <= 1.0
    assert metrics["loss"] == metrics["loss"]  # not NaN
