"""S7 mechanism-aware distillation losses.

Both losses supervise the STUDENT'S INTERVENTION EFFECT against the TEACHER'S
INTERVENTION EFFECT on A/B/C/D mechanism quadruplets (S6 causal-audit states):

  A = baseline                 (context label and mediator both at baseline)
  B = natural intervention     (label + mediator both change)
  C = broken-path intervention (label changes, mediator stays at baseline)
  D = mediator-only            (label at baseline, mediator changes)

The mechanism loss matches E_natural = P(B)-P(A) and E_mediator = P(D)-P(A);
the broken-path loss matches E_broken = P(C)-P(A).

CRITICAL DESIGN RULE (S7 instruction §13): the broken-path loss must NOT be
||E_S^broken||. The teacher's own broken-path effect is non-zero, so the
student is trained toward "Student broken effect == Teacher broken effect",
never toward zero.
"""
from __future__ import annotations

import torch

_EPS = 1e-8


def _effect_gap_l1(
    student_x: torch.Tensor,
    student_a: torch.Tensor,
    teacher_x: torch.Tensor,
    teacher_a: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample mean |(T_x - T_A) - (S_x - S_A)| over alternatives.

    ``*_x`` / ``*_a`` are dense mode-aligned probability vectors for the
    intervention state X and the baseline A (shape ``(B, M)``). ``mask`` (1.0
    for available alternatives) normalizes the mean over available modes;
    unavailable slots carry zero probability mass on both sides and add zero.
    """
    teacher_effect = teacher_x - teacher_a
    student_effect = student_x - student_a
    diff = (teacher_effect - student_effect).abs()
    if mask is not None:
        diff = diff * mask
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return diff.sum(dim=-1) / denom
    return diff.mean(dim=-1)


def mechanism_fidelity_loss(
    p_a: torch.Tensor,
    p_b: torch.Tensor,
    p_d: torch.Tensor,
    t_a: torch.Tensor,
    t_b: torch.Tensor,
    t_d: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_mechanism = ||E_T^nat - E_S^nat||_1 + ||E_T^med - E_S^med||_1.

    E^nat = P(B) - P(A) (natural intervention effect),
    E^med = P(D) - P(A) (mediator-only effect).

    Returns a per-sample vector (shape ``(B,)``); call ``.mean()`` for a batch
    scalar. Loss is zero iff the student reproduces the teacher's natural AND
    mediator intervention effects.
    """
    nat = _effect_gap_l1(p_b, p_a, t_b, t_a, mask=mask)
    med = _effect_gap_l1(p_d, p_a, t_d, t_a, mask=mask)
    return nat + med


def broken_path_fidelity_loss(
    p_a: torch.Tensor,
    p_c: torch.Tensor,
    t_a: torch.Tensor,
    t_c: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L_broken = ||E_T^broken - E_S^broken||_1.

    E^broken = P(C) - P(A). The student is matched to the TEACHER's broken-path
    effect, which is deliberately non-zero; this loss must not be replaced by
    ||E_S^broken|| (that would wrongly force the broken effect toward zero).
    """
    return _effect_gap_l1(p_c, p_a, t_c, t_a, mask=mask)
