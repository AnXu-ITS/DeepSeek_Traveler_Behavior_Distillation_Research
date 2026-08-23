"""Distillation losses: KL, CE, Huber, and the combined v0.2-A loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_EPS = 1e-8


def kl_divergence(target_probs: torch.Tensor, student_probs: torch.Tensor) -> torch.Tensor:
    """KL(P_teacher || P_student), numerically stable.

    ``target_probs`` may contain zeros (0 * log(0) treated as 0). ``student_probs``
    is clamped to ``[eps, 1]`` so log never sees 0.
    """
    student = student_probs.clamp(min=_EPS, max=1.0)
    target = target_probs.clamp(min=0.0)
    # target * log(target / student); where target==0 the term contributes 0.
    log_term = torch.log(target.clamp(min=_EPS) / student)
    return (target * log_term).sum(dim=-1)


def mode_cross_entropy(target_mode_idx: torch.Tensor, student_probs: torch.Tensor) -> torch.Tensor:
    """CE(a_T, P_S) with a_T = argmax teacher probability."""
    return F.cross_entropy(torch.log(student_probs.clamp(min=_EPS)), target_mode_idx, reduction="none")


def departure_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta, reduction="none")


def elasticity_l1(
    teacher_delta: torch.Tensor,
    student_delta: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample mean-|DeltaP_T - DeltaP_S| over available modes.

    ``teacher_delta`` / ``student_delta`` are mode-aligned dense vectors
    (``P(counterfactual) - P(baseline)``). ``mask`` (1.0 for available
    alternatives) restricts the mean to available modes; entries without a mask
    are ignored. Unavailable modes carry zero mass on both sides so they add
    zero regardless, but the mask keeps the normalization honest when choice
    sets vary across a batch.
    """
    diff = (teacher_delta - student_delta).abs()
    if mask is not None:
        diff = diff * mask
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return diff.sum(dim=-1) / denom
    return diff.mean(dim=-1)


def heterogeneity_l1(
    teacher_diff: torch.Tensor,
    student_diff: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Persona-contrast supervision: match the between-persona response gap.

    Same math as :func:`elasticity_l1` but applied to a pair of PERSONAS in the
    same travel situation: supervises the student's difference
    ``P_S(persona A) - P_S(persona B)`` to match the teacher's difference.
    ``mask`` should be the union availability mask (available in either
    persona); modes available to neither persona contribute zero mass.
    """
    return elasticity_l1(teacher_diff, student_diff, mask=mask)


def elasticity_direction_loss(
    teacher_delta: torch.Tensor,
    student_delta: torch.Tensor,
    mask: torch.Tensor | None = None,
    margin: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize per-mode responses that move OPPOSITE to the teacher.

    For each mode: ``max(0, -sign(DeltaT) * DeltaS + margin)`` — nonzero only
    when the student moves against (or, with margin>0, not far enough in) the
    teacher's direction. Modes where the teacher did not move (|DeltaT| < eps)
    are excluded entirely: with sign 0 they would otherwise be penalized by
    ``margin``, which is not intended.
    """
    sign_t = torch.sign(teacher_delta)
    moved = (teacher_delta.abs() > eps).float()
    violation = torch.clamp(-sign_t * student_delta + margin, min=0.0) * moved
    if mask is not None:
        violation = violation * mask
        denom = (mask * moved).sum(dim=-1).clamp(min=1.0)
    else:
        denom = moved.sum(dim=-1).clamp(min=1.0)
    return violation.sum(dim=-1) / denom


def elasticity_magnitude_loss(
    teacher_delta: torch.Tensor,
    student_delta: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize per-mode |response magnitude| mismatch.

    ``||DeltaT| - |DeltaS||`` per mode, averaged over modes. Pairs with the
    direction loss: direction is handled by :func:`elasticity_direction_loss`,
    magnitude by this term.
    """
    diff = (teacher_delta.abs() - student_delta.abs()).abs()
    if mask is not None:
        diff = diff * mask
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return diff.sum(dim=-1) / denom
    return diff.mean(dim=-1)


def elasticity_sign_agreement(
    teacher_delta: torch.Tensor,
    student_delta: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Fraction of modes where teacher and student deltas share sign (evaluation).

    A mode counts as agreeing if signs are equal OR the teacher did not move
    there (|DeltaT| < eps). Ranges [0, 1]; higher is better.
    """
    st = torch.sign(teacher_delta)
    ss = torch.sign(student_delta)
    agree = (st == ss) | (teacher_delta.abs() < eps)
    if mask is not None:
        denom = mask.sum(dim=-1).clamp(min=1.0)
        return (agree.float() * mask).sum(dim=-1) / denom
    return agree.float().mean(dim=-1)


def student_loss(
    student_out: dict,
    target_probs: torch.Tensor,
    target_mode_idx: torch.Tensor,
    target_departure: torch.Tensor,
    lambda_action: float = 1.0,
    lambda_distribution: float = 1.0,
    lambda_departure: float = 1.0,
    huber_delta: float = 1.0,
) -> dict:
    """Total v0.2-A loss = lambda_A * L_action + lambda_D * L_distribution.

    L_action = CE(mode) + lambda_departure * Huber(departure)
    L_distribution = KL(P_teacher || P_student)
    """
    ce = mode_cross_entropy(target_mode_idx, student_out["mode_probabilities"])
    huber = departure_huber(
        student_out["departure_time_shift_min"], target_departure, delta=huber_delta
    )
    kl = kl_divergence(target_probs, student_out["mode_probabilities"])

    l_action = ce + lambda_departure * huber
    total = lambda_action * l_action + lambda_distribution * kl

    return {
        "total": total,
        "ce": ce,
        "huber": huber,
        "action": l_action,
        "kl": kl,
    }
