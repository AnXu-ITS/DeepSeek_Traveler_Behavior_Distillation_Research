"""S7 mechanism-loss unit tests.

The critical property (S7 instruction §13): the broken-path loss matches the
student's broken effect to the TEACHER's broken effect — it must NOT penalize
the student's broken effect magnitude per se, and pushing the student's broken
effect to zero while the teacher's is non-zero must increase the loss.
"""
from __future__ import annotations

import pytest
import torch

from traveler_distillation.student import (
    mechanism_fidelity_loss,
    broken_path_fidelity_loss,
)


def _one(*probs):
    return torch.tensor([list(probs)], dtype=torch.float32)


def test_mechanism_loss_zero_when_effects_match():
    p_a = _one(0.3, 0.5, 0.2, 0.0)
    p_b = _one(0.25, 0.5, 0.25, 0.0)
    p_d = _one(0.28, 0.52, 0.2, 0.0)
    out = mechanism_fidelity_loss(p_a, p_b, p_d, p_a, p_b, p_d)
    assert out.shape == (1,)
    assert out.item() == pytest.approx(0.0)


def test_mechanism_loss_penalizes_wrong_effect_direction():
    # teacher: natural intervention moves 0.1 from mode0 to mode2; student
    # moves the opposite way -> nonzero loss on the natural term.
    t_a = _one(0.4, 0.4, 0.2, 0.0)
    t_b = _one(0.3, 0.4, 0.3, 0.0)
    t_d = t_a.clone()  # mediator effect zero
    p_a = t_a.clone()
    p_b = _one(0.5, 0.4, 0.1, 0.0)  # opposite
    p_d = p_a.clone()
    out = mechanism_fidelity_loss(p_a, p_b, p_d, t_a, t_b, t_d)
    # effect gap per mode: |-0.1 - (+0.1)| + |0 - 0| + |+0.1 - (-0.1)| + |0 - 0|
    #                     = 0.2 + 0 + 0.2 + 0 = 0.4 ; /4 = 0.1  (mediator term = 0)
    assert out.item() == pytest.approx(0.4 / 4.0)


def test_mechanism_loss_mediator_term_only():
    # natural effects match; only the mediator term is wrong.
    t_a = _one(0.4, 0.4, 0.2, 0.0)
    t_b = _one(0.35, 0.4, 0.25, 0.0)
    t_d = _one(0.32, 0.4, 0.28, 0.0)
    p_a = t_a.clone()
    p_b = t_b.clone()
    p_d = t_a.clone()  # student ignores the mediator
    out = mechanism_fidelity_loss(p_a, p_b, p_d, t_a, t_b, t_d)
    # mediator term: |(-0.08)-(0)| + |0-0| + |(+0.08)-(0)| + |0-0| = 0.16 ; /4
    assert out.item() == pytest.approx(0.16 / 4.0)


def test_broken_loss_zero_when_student_matches_teacher_broken_effect():
    # teacher's broken effect is NON-ZERO; a student with the same non-zero
    # broken effect must get zero loss (never "broken effect -> 0").
    t_a = _one(0.4, 0.4, 0.2, 0.0)
    t_c = _one(0.33, 0.4, 0.27, 0.0)  # teacher broken effect = (-0.07, 0, +0.07, 0)
    p_a = _one(0.35, 0.45, 0.2, 0.0)
    p_c = _one(0.28, 0.45, 0.27, 0.0)  # same effect vector
    out = broken_path_fidelity_loss(p_a, p_c, t_a, t_c)
    assert out.item() == pytest.approx(0.0)


def test_broken_loss_penalizes_zeroing_the_broken_effect():
    # student shows NO broken-path response while the teacher does -> loss > 0.
    t_a = _one(0.4, 0.4, 0.2, 0.0)
    t_c = _one(0.33, 0.4, 0.27, 0.0)
    p_a = _one(0.35, 0.45, 0.2, 0.0)
    p_c = p_a.clone()
    out = broken_path_fidelity_loss(p_a, p_c, t_a, t_c)
    # |(-0.07) - 0| + |0-0| + |(+0.07) - 0| + |0-0| = 0.14 ; /4
    assert out.item() == pytest.approx(0.14 / 4.0)


def test_mask_restricts_denominator():
    t_a = _one(0.4, 0.4, 0.2, 0.0)
    t_b = _one(0.3, 0.4, 0.3, 0.0)
    p_a = t_a.clone()
    p_b = _one(0.5, 0.4, 0.1, 0.0)
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out = mechanism_fidelity_loss(p_a, p_b, p_a, t_a, t_b, t_a, mask=mask)
    # natural term only, first 3 modes: 0.2 + 0 + 0.2 = 0.4 ; /3
    assert out.item() == pytest.approx(0.4 / 3.0)


def test_batch_shape():
    b = 4
    m = 5
    t = torch.rand(b, m).softmax(dim=-1)
    s = torch.rand(b, m).softmax(dim=-1)
    out = mechanism_fidelity_loss(s, s, s, t, t, t)
    assert out.shape == (b,)
    out2 = broken_path_fidelity_loss(s, s, t, t)
    assert out2.shape == (b,)
