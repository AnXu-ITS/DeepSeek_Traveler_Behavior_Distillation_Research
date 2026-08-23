"""Elasticity loss + counterfactual pair utilities tests."""
from __future__ import annotations

import pytest
import torch

from traveler_distillation.student import (
    elasticity_l1,
    elasticity_direction_loss,
    elasticity_magnitude_loss,
    elasticity_sign_agreement,
)
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget
from traveler_distillation.schemas.dataset import Perturbation
from traveler_distillation.student import build_baseline_index, make_counterfactual_pairs


def test_elasticity_l1_zero_when_deltas_match():
    t = torch.tensor([[0.1, -0.2, 0.3, 0.0]])
    s = t.clone()
    out = elasticity_l1(t, s)
    assert out.shape == (1,)
    assert out.item() == pytest.approx(0.0)


def test_elasticity_l1_mean_abs_diff():
    t = torch.tensor([[0.1, -0.2, 0.3, 0.0]])
    s = torch.tensor([[0.0, 0.0, 0.1, 0.0]])
    out = elasticity_l1(t, s)
    # |0.1-0| + |-0.2-0| + |0.3-0.1| + |0-0| = 0.5 ; /4 = 0.125
    assert out.item() == pytest.approx(0.125)


def test_elasticity_l1_mask_restricts_denominator():
    t = torch.tensor([[0.1, -0.2, 0.3, 0.0]])
    s = torch.tensor([[0.0, 0.0, 0.1, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out = elasticity_l1(t, s, mask=mask)
    # only first 3 modes count: 0.5 ; /3 = 0.1667
    assert out.item() == pytest.approx(0.5 / 3.0)


def _util_sample(sid, axis, level, baseline_id=None, split_group="SG0"):
    """AggregatedTeacherTarget with minimal fields for grouping utilities.

    ``model_construct`` bypasses pydantic validation so we don't need a full
    ``UniversalTravelerState`` for utilities that only read ids/perturbation.
    """
    return AggregatedTeacherTarget.model_construct(
        sample_id=sid,
        counterfactual_group_id=None if axis == "baseline" else "CF0",
        baseline_sample_id=baseline_id,
        split_group_id=split_group,
        perturbation=Perturbation(axis=axis, level=level),
    )


def test_build_baseline_index_and_pairs():
    base = _util_sample("B0", "baseline", 0.0)
    cf1 = _util_sample("C1", "weather_intensity", 0.5, baseline_id="B0")
    cf2 = _util_sample("C2", "fare_multiplier", 2.0, baseline_id="B0")
    orphan = _util_sample("C3", "fare_multiplier", 1.5, baseline_id="MISSING")

    idx = build_baseline_index([base, cf1, cf2, orphan])
    assert idx == {"B0": base}

    pairs = make_counterfactual_pairs([base, cf1, cf2, orphan], idx)
    assert len(pairs) == 2
    assert all(b.sample_id == "B0" for b, _ in pairs)
    assert {c.sample_id for _, c in pairs} == {"C1", "C2"}


def test_pair_dataset_mode_alignment(config):
    from traveler_distillation.student import (
        CounterfactualPairDataset, collate_pairs, FeatureExtractor,
    )
    from traveler_distillation.dataset.aggregation import TeacherAggregate, AggregationMetadata
    from traveler_distillation.generators import (
        PersonaGenerator, TripGenerator, BaselineStateGenerator,
        CounterfactualContextGenerator,
    )

    persona = PersonaGenerator(seed=1, config=config).generate(1)[0]
    trip = TripGenerator(seed=1, config=config).generate(1)[0]
    base_state = BaselineStateGenerator(config).generate(persona, trip)
    cf_state = CounterfactualContextGenerator(config).generate(
        base_state, "weather_intensity", [0.5]
    )[0].state

    def _t(sid, state, axis, level, base_id=None):
        return AggregatedTeacherTarget(
            sample_id=sid,
            counterfactual_group_id=None if axis == "baseline" else "CF0",
            baseline_sample_id=base_id,
            split_group_id="SG0",
            perturbation=Perturbation(axis=axis, level=level),
            state=state,
            teacher_aggregate=TeacherAggregate(
                mode_probabilities={a.mode: 1 / len(state.alternatives) for a in state.alternatives},
                selected_mode=state.available_modes[0],
                departure_time_shift_min=0.0,
                confidence_mean=0.8,
            ),
            aggregation_metadata=AggregationMetadata(k=3, prompt_version="t", model="m"),
        )

    base = _t("B0", base_state, "baseline", 0.0)
    cf = _t("C1", cf_state, "weather_intensity", 0.5, base_id="B0")

    extractor = FeatureExtractor().fit([base_state, cf_state])
    ds = CounterfactualPairDataset([(base, cf)], extractor)
    batch = collate_pairs([ds[0]])

    # baseline and CF share the same persona+trip -> identical alternative list
    assert torch.equal(batch["base"]["alt_mode_idx"], batch["cf"]["alt_mode_idx"])
    assert torch.equal(batch["base"]["alt_mask"], batch["cf"]["alt_mask"])
    assert batch["base"]["target_probs"].shape == batch["cf"]["target_probs"].shape


# ------------------------------------------------- decomposed elasticity loss

def test_direction_loss_penalizes_only_opposite_movement():
    t = torch.tensor([[0.2, -0.1, 0.0, 0.3]])
    s = torch.tensor([[0.1, 0.05, 0.2, 0.4]])  # mode2 (t=0) ok, others: m1 same dir, m2 opp dir, m3 opp? m4 same dir
    # teacher moved in modes 0,1,3 (mode2 unmoved -> excluded from denom).
    # mode0: sign_t=+1, s=+0.1 -> violation 0
    # mode1: sign_t=-1, s=+0.05 -> violation 0.05
    # mode2: unmoved -> 0
    # mode3: sign_t=+1, s=0.4 -> violation 0
    # denom = moved count = 3
    out = elasticity_direction_loss(t, s)
    assert out.item() == pytest.approx(0.05 / 3.0)


def test_direction_loss_margin():
    t = torch.tensor([[0.2, 0.0]])
    s = torch.tensor([[0.01, 0.5]])
    out = elasticity_direction_loss(t, s, margin=0.05)
    # mode0: max(0, -0.01+0.05)=0.04 ; mode1: teacher unmoved -> excluded
    # denom = moved count = 1
    assert out.item() == pytest.approx(0.04 / 1.0)


def test_magnitude_loss_absolute_magnitude_mismatch():
    t = torch.tensor([[0.2, -0.1, 0.0]])
    s = torch.tensor([[-0.15, 0.05, 0.1]])
    out = elasticity_magnitude_loss(t, s)
    # ||0.2|-|0.15|| = 0.05 ; ||0.1|-|0.05|| = 0.05 ; ||0|-|0.1|| = 0.1 ; /3
    assert out.item() == pytest.approx((0.05 + 0.05 + 0.1) / 3.0)


def test_sign_agreement_counts_teacher_unmoved_as_agree():
    t = torch.tensor([[0.2, -0.1, 0.0]])
    s = torch.tensor([[0.1, 0.05, 0.2]])
    out = elasticity_sign_agreement(t, s)
    # mode0 agree (+/+), mode1 disagree (-/+), mode2 teacher unmoved -> agree
    assert out.item() == pytest.approx(2 / 3.0)


def test_direction_and_magnitude_mask():
    t = torch.tensor([[0.2, -0.1, 0.0, 0.0]])
    s = torch.tensor([[0.1, 0.05, 0.2, 0.9]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out_dir = elasticity_direction_loss(t, s, mask=mask)
    # violations: mode1 = 0.05; denom = mask*moved = 2
    assert out_dir.item() == pytest.approx(0.05 / 2.0)
    out_mag = elasticity_magnitude_loss(t, s, mask=mask)
    # mode0 0.1, mode1 0.05, mode2 0.2 ; /3
    assert out_mag.item() == pytest.approx((0.1 + 0.05 + 0.2) / 3.0)
