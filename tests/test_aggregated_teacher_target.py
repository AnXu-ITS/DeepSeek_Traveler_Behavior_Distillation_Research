"""AggregatedTeacherTarget schema tests."""
from __future__ import annotations

import pytest

from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.schemas.dataset import Perturbation
from traveler_distillation.dataset.aggregation import (
    AggregatedTeacherTarget,
    AggregationMetadata,
    aggregate_repeats,
)


def _action(probs, selected=None, shift=0, confidence=0.8):
    return UniversalTravelerAction(
        selected_mode=selected or max(probs, key=probs.get),
        mode_probabilities=probs,
        departure_time_shift_min=shift,
        confidence=confidence,
        reason_codes=[],
    )


def test_aggregate_mean_probabilities():
    actions = [
        _action({"bike": 0.7, "pt": 0.3}),
        _action({"bike": 0.5, "pt": 0.5}),
        _action({"bike": 0.9, "pt": 0.1}),
    ]
    agg = aggregate_repeats(actions)
    assert agg.mode_probabilities["bike"] == pytest.approx(0.7)
    assert agg.mode_probabilities["pt"] == pytest.approx(0.3)
    assert sum(agg.mode_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_aggregate_selected_mode_is_argmax_of_mean_not_majority():
    # bike wins 2/3 by majority but pt has higher MEAN probability
    actions = [
        _action({"bike": 0.51, "pt": 0.49}, selected="bike"),
        _action({"bike": 0.51, "pt": 0.49}, selected="bike"),
        _action({"bike": 0.0, "pt": 1.0}, selected="pt"),
    ]
    agg = aggregate_repeats(actions)
    # mean bike = 0.34, mean pt = 0.66 -> pt wins by mean
    assert agg.selected_mode == "pt"


def test_aggregate_mean_departure_and_confidence():
    actions = [
        _action({"bike": 1.0}, shift=0, confidence=0.8),
        _action({"bike": 1.0}, shift=10, confidence=0.9),
        _action({"bike": 1.0}, shift=-4, confidence=1.0),
    ]
    agg = aggregate_repeats(actions)
    assert agg.departure_time_shift_min == pytest.approx(2.0)
    assert agg.confidence_mean == pytest.approx(0.9)
    assert agg.confidence_std > 0.0


def test_aggregated_target_roundtrip(baseline_state):
    action = _action({"bike": 0.7, "pt": 0.3})
    agg = aggregate_repeats([action])
    target = AggregatedTeacherTarget(
        sample_id="S1",
        counterfactual_group_id="CF1",
        perturbation=Perturbation(axis="baseline", level=0.0),
        state=baseline_state,
        teacher_aggregate=agg,
        aggregation_metadata=AggregationMetadata(
            k=3, prompt_version="teacher_v0.1", model="m", source_completion_ids=["a", "b", "c"]
        ),
    )
    d = target.model_dump()
    assert d["sample_id"] == "S1"
    assert d["aggregation_metadata"]["k"] == 3
    assert d["teacher_aggregate"]["selected_mode"] == "bike"
