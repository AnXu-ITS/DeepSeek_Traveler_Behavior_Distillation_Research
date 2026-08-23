"""Loss function correctness tests (KL, CE, Huber, total)."""
from __future__ import annotations

import math

import pytest
import torch

from traveler_distillation.student.losses import (
    kl_divergence,
    mode_cross_entropy,
    departure_huber,
    student_loss,
)


def test_kl_divergence():
    target = torch.tensor([[0.7, 0.3]])
    student = torch.tensor([[0.6, 0.4]])
    expected = 0.7 * math.log(0.7 / 0.6) + 0.3 * math.log(0.3 / 0.4)
    assert kl_divergence(target, student).item() == pytest.approx(expected, abs=1e-6)


def test_kl_zero_target_is_zero_contribution():
    target = torch.tensor([[1.0, 0.0]])
    student = torch.tensor([[0.5, 0.5]])
    # KL = 1*log(1/0.5) + 0*log(0/0.5) = log(2)
    assert kl_divergence(target, student).item() == pytest.approx(math.log(2.0), abs=1e-6)


def test_kl_no_nan():
    target = torch.tensor([[1.0, 0.0, 0.0]])
    student = torch.tensor([[0.98, 0.01, 0.01]])
    v = kl_divergence(target, student)
    assert torch.isfinite(v).all()


def test_mode_cross_entropy():
    probs = torch.tensor([[0.6, 0.4]])
    idx = torch.tensor([0])
    assert mode_cross_entropy(idx, probs).item() == pytest.approx(-math.log(0.6), abs=1e-6)


def test_departure_huber():
    pred = torch.tensor([1.0])
    tgt = torch.tensor([2.0])
    # |d| = 1 <= delta -> 0.5 * 1^2 = 0.5
    assert departure_huber(pred, tgt, delta=1.0).item() == pytest.approx(0.5, abs=1e-6)


def test_student_loss_shapes():
    out = {
        "mode_probabilities": torch.tensor([[0.6, 0.4], [0.3, 0.7]]),
        "departure_time_shift_min": torch.tensor([1.0, -2.0]),
    }
    target_probs = torch.tensor([[0.7, 0.3], [0.5, 0.5]])
    target_idx = torch.tensor([0, 1])
    target_dep = torch.tensor([2.0, -1.0])
    losses = student_loss(out, target_probs, target_idx, target_dep)
    assert losses["total"].shape == (2,)
    assert losses["action"].shape == (2,)
    assert losses["kl"].shape == (2,)
    # total = lambda_A * action + lambda_D * kl
    assert torch.allclose(
        losses["total"],
        losses["action"] + losses["kl"],
        atol=1e-6,
    )
