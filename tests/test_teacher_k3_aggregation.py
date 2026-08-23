"""K=3 teacher aggregation correctness tests."""
from __future__ import annotations

import pytest

from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.dataset.aggregation import aggregate_repeats


def _action(probs, shift, confidence):
    return UniversalTravelerAction(
        selected_mode=max(probs, key=probs.get),
        mode_probabilities=probs,
        departure_time_shift_min=shift,
        confidence=confidence,
    )


def test_k3_mean_distribution_sums_to_one():
    actions = [
        _action({"car": 0.5, "pt": 0.5}, 0, 0.8),
        _action({"car": 0.6, "pt": 0.4}, 0, 0.9),
        _action({"car": 0.7, "pt": 0.3}, 0, 0.7),
    ]
    agg = aggregate_repeats(actions)
    assert sum(agg.mode_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert agg.mode_probabilities["car"] == pytest.approx(0.6)


def test_k3_departure_shift_is_float_mean():
    actions = [
        _action({"pt": 1.0}, 5, 0.8),
        _action({"pt": 1.0}, -5, 0.8),
        _action({"pt": 1.0}, 3, 0.8),
    ]
    agg = aggregate_repeats(actions)
    # mean = 1.0, kept as float (not rounded to int)
    assert agg.departure_time_shift_min == pytest.approx(1.0)


def test_k3_selected_mode_uses_mean_distribution():
    actions = [
        _action({"bike": 0.6, "pt": 0.4}, 0, 0.8),
        _action({"bike": 0.6, "pt": 0.4}, 0, 0.8),
        _action({"bike": 0.1, "pt": 0.9}, 0, 0.8),
    ]
    agg = aggregate_repeats(actions)
    # mean bike = 0.433, mean pt = 0.567 -> pt
    assert agg.selected_mode == "pt"


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_repeats([])
