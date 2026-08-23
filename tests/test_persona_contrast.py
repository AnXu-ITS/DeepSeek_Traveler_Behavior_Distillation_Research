"""Persona-contrast pairing + heterogeneity loss tests."""
from __future__ import annotations

import pytest
import torch

from traveler_distillation.student import heterogeneity_l1
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget, TeacherAggregate, AggregationMetadata
from traveler_distillation.schemas.dataset import Perturbation
from traveler_distillation.student import make_persona_contrast_pairs, PersonaContrastPairDataset, collate_contrast_pairs, FeatureExtractor


def test_heterogeneity_l1_matches_elasticity_math():
    t = torch.tensor([[0.3, -0.1, 0.0, 0.0]])
    s = torch.tensor([[0.1, 0.0, 0.1, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    # |0.3-0.1| + |-0.1-0| + |0-0.1| = 0.2+0.1+0.1 = 0.4 ; /3
    out = heterogeneity_l1(t, s, mask=mask)
    assert out.item() == pytest.approx(0.4 / 3.0)


def _sample(sid, persona_id, trip_id, axis, level):
    from traveler_distillation.schemas.persona import Persona
    from traveler_distillation.schemas.trip import Trip
    from traveler_distillation.schemas.state import UniversalTravelerState
    from traveler_distillation.schemas.context import DynamicContext, Weather
    from traveler_distillation.schemas.alternative import TravelAlternative

    persona = Persona(
        persona_id=persona_id, age_group="25-34", income_group="low",
        occupation="office_worker", household_size=1, has_children=False,
        car_ownership=False, driving_license=False, bike_ownership=True,
        transit_pass=True, habitual_mode="bike", schedule_flexibility="low",
        mobility_limitation="none",
    )
    trip = Trip(
        trip_id=trip_id, purpose="commute", origin_type="home",
        destination_type="work", distance_km=5.0, desired_departure_min=480,
        desired_arrival_min=510, time_constraint="hard",
    )
    state = UniversalTravelerState(
        persona=persona, trip=trip,
        context=DynamicContext(
            context_id="C", weather=Weather(condition="clear", intensity=0.0),
            road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
            road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
            congestion_charge=0.0,
        ),
        alternatives=[
            TravelAlternative(mode="car", available=False, travel_time_min=10, monetary_cost=5, access_time_min=3, transfers=0, reliability_delay_min=0, weather_exposure=0.05),
            TravelAlternative(mode="pt", available=True, travel_time_min=20, monetary_cost=3, access_time_min=8, transfers=1, reliability_delay_min=0, weather_exposure=0.4),
            TravelAlternative(mode="bike", available=True, travel_time_min=22, monetary_cost=0, access_time_min=1, transfers=0, reliability_delay_min=0, weather_exposure=0.9),
            TravelAlternative(mode="walk", available=True, travel_time_min=60, monetary_cost=0, access_time_min=0, transfers=0, reliability_delay_min=0, weather_exposure=1.0),
        ],
    )
    agg = TeacherAggregate(
        mode_probabilities={"pt": 0.5, "bike": 0.3, "walk": 0.2},
        selected_mode="pt", departure_time_shift_min=0.0, confidence_mean=0.8,
    )
    return AggregatedTeacherTarget(
        sample_id=sid,
        counterfactual_group_id=None if axis == "baseline" else "CF0",
        baseline_sample_id=None,
        split_group_id=f"{persona_id}::{trip_id}",
        perturbation=Perturbation(axis=axis, level=level),
        state=state, teacher_aggregate=agg,
        aggregation_metadata=AggregationMetadata(k=3, prompt_version="t", model="m"),
    )


def test_make_persona_contrast_pairs_groups_by_situation():
    # 3 personas, same trip, same axis+level -> all distinct pairs
    a = _sample("A", "P1", "T1", "baseline", 0.0)
    b = _sample("B", "P2", "T1", "baseline", 0.0)
    c = _sample("C", "P3", "T1", "baseline", 0.0)
    pairs = make_persona_contrast_pairs([a, b, c])
    pairs_set = {frozenset((x.sample_id, y.sample_id)) for x, y in pairs}
    assert pairs_set == {frozenset(("A", "B")), frozenset(("A", "C")), frozenset(("B", "C"))}


def test_make_persona_contrast_pairs_no_cross_situation_pairing():
    a = _sample("A", "P1", "T1", "baseline", 0.0)
    b = _sample("B", "P2", "T1", "weather_intensity", 0.5)   # different axis
    c = _sample("C", "P3", "T2", "baseline", 0.0)             # different trip
    pairs = make_persona_contrast_pairs([a, b, c])
    assert pairs == []


def test_contrast_pair_dataset_mode_alignment():
    a = _sample("A", "P1", "T1", "baseline", 0.0)
    b = _sample("B", "P2", "T1", "baseline", 0.0)
    extractor = FeatureExtractor().fit([a.state, b.state])
    ds = PersonaContrastPairDataset([(a, b)], extractor)
    batch = collate_contrast_pairs([ds[0]])
    assert torch.equal(batch["a"]["alt_mode_idx"], batch["b"]["alt_mode_idx"])
    # union availability mask works in loss (all three modes available in both)
    from traveler_distillation.student import heterogeneity_l1 as het
    teacher_diff = batch["a"]["target_probs"] - batch["b"]["target_probs"]
    student_diff = teacher_diff.clone()
    union = (batch["a"]["alt_mask"] + batch["b"]["alt_mask"]).clamp(max=1.0)
    assert het(teacher_diff, student_diff, mask=union).item() == pytest.approx(0.0)
