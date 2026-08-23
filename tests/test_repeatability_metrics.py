"""Repeatability metrics tests."""
from __future__ import annotations

import pytest

from traveler_distillation.audit import (
    selected_mode_agreement,
    probability_stability,
    mean_pairwise_l1,
    l1,
    analyze_repeatability,
    RepeatabilityRecord,
    state_hash,
)
from traveler_distillation.schemas.action import UniversalTravelerAction


def test_selected_mode_agreement():
    assert selected_mode_agreement(["bike", "bike", "bike"]) == 1.0
    assert selected_mode_agreement(["bike", "pt", "bike"]) == pytest.approx(2 / 3)


def test_l1_distance():
    assert l1({"bike": 1.0}, {"pt": 1.0}) == pytest.approx(2.0)
    assert l1({"bike": 0.5, "pt": 0.5}, {"bike": 0.5, "pt": 0.5}) == 0.0
    assert l1({"bike": 0.8, "pt": 0.2}, {"bike": 0.7, "pt": 0.3}) == pytest.approx(0.2)


def test_mean_pairwise_l1():
    dists = [{"bike": 1.0}, {"bike": 0.8, "pt": 0.2}]
    # only one pair: |1-0.8| + |0-0.2| = 0.4
    assert mean_pairwise_l1(dists) == pytest.approx(0.4)


def test_probability_stability():
    probs = [{"bike": 0.72}, {"bike": 0.75}, {"bike": 0.71}]
    out = probability_stability(probs)
    assert out["bike"]["mean"] == pytest.approx(0.7267, abs=1e-3)
    assert out["bike"]["range"] == pytest.approx(0.04, abs=1e-9)


def _repeat_record(baseline_state, rep, probs, shift=0, confidence=0.8):
    action = UniversalTravelerAction(
        selected_mode=max(probs, key=probs.get),
        mode_probabilities=probs,
        departure_time_shift_min=shift,
        confidence=confidence,
    )
    return RepeatabilityRecord(
        audit_state_id="A001",
        source_sample_id="S1",
        repeat_index=rep,
        state_hash=state_hash(baseline_state),
        state=baseline_state,
        teacher=action,
        timestamp="t",
    )


def test_analyze_repeatability_stable(baseline_state):
    recs = [
        _repeat_record(baseline_state, 0, {"bike": 0.70, "pt": 0.30}),
        _repeat_record(baseline_state, 1, {"bike": 0.72, "pt": 0.28}),
        _repeat_record(baseline_state, 2, {"bike": 0.71, "pt": 0.29}),
    ]
    res = analyze_repeatability(recs)
    assert res["aggregate"]["n_states"] == 1
    assert res["aggregate"]["mean_selected_mode_agreement"] == 1.0
    assert res["aggregate"]["mean_pairwise_probability_l1"] < 0.1


def test_analyze_repeatability_flip(baseline_state):
    recs = [
        _repeat_record(baseline_state, 0, {"bike": 0.6, "pt": 0.4}),
        _repeat_record(baseline_state, 1, {"bike": 0.4, "pt": 0.6}),
        _repeat_record(baseline_state, 2, {"bike": 0.6, "pt": 0.4}),
    ]
    res = analyze_repeatability(recs)
    assert res["per_group"][0]["selected_mode_agreement"] == pytest.approx(2 / 3, abs=1e-3)
