"""Audit flag tests: flags flag suspicious behavior without auto-invalidating."""
from __future__ import annotations

import pytest

from traveler_distillation.audit import compute_flags


def test_repeatability_selected_mode_flip():
    repeatability = {
        "per_group": [
            {
                "state_hash": "h1",
                "source_sample_id": "S1",
                "selected_mode_agreement": 0.33,
                "selected_modes": ["bike", "pt", "walk"],
                "mean_pairwise_l1": 0.0,
                "departure_shift": {"range": 0.0, "values": [0, 0, 0]},
                "confidence": {"std": 0.0},
                "probability_stability": {"bike": {"range": 0.0}},
            }
        ]
    }
    flags = compute_flags(repeatability=repeatability)
    assert any(f["flag"] == "selected_mode_flip" for f in flags)


def test_repeatability_high_variance():
    repeatability = {
        "per_group": [
            {
                "state_hash": "h2",
                "source_sample_id": "S2",
                "selected_mode_agreement": 1.0,
                "selected_modes": ["bike", "bike", "bike"],
                "mean_pairwise_l1": 0.45,
                "departure_shift": {"range": 0.0, "values": [0, 0, 0]},
                "confidence": {"std": 0.0},
                "probability_stability": {"bike": {"range": 0.1}},
            }
        ]
    }
    flags = compute_flags(repeatability=repeatability)
    assert any(f["flag"] == "high_repeat_variance" for f in flags)


def test_departure_shift_instability():
    repeatability = {
        "per_group": [
            {
                "state_hash": "h3",
                "source_sample_id": "S3",
                "selected_mode_agreement": 1.0,
                "selected_modes": ["bike", "bike", "bike"],
                "mean_pairwise_l1": 0.0,
                "departure_shift": {"range": 35.0, "values": [-20, 0, 15]},
                "confidence": {"std": 0.0},
                "probability_stability": {"bike": {"range": 0.0}},
            }
        ]
    }
    flags = compute_flags(repeatability=repeatability)
    assert any(f["flag"] == "departure_shift_instability" for f in flags)


def test_counterfactual_unexpected_fare_response():
    counterfactual = {
        "axes": {
            "fare_multiplier": {
                "mode_stats": {
                    "pt": {"spearman_rho": 0.5, "mean_direction_reversals": 0.0}
                }
            }
        }
    }
    flags = compute_flags(counterfactual=counterfactual)
    assert any(f["flag"] == "unexpected_positive_fare_response" for f in flags)


def test_counterfactual_unexpected_transit_delay_response():
    counterfactual = {
        "axes": {
            "transit_delay": {
                "mode_stats": {
                    "pt": {"spearman_rho": 0.4, "mean_direction_reversals": 0.0}
                }
            }
        }
    }
    flags = compute_flags(counterfactual=counterfactual)
    assert any(f["flag"] == "unexpected_positive_transit_delay_response" for f in flags)


def test_counterfactual_strong_reversal():
    counterfactual = {
        "axes": {
            "weather_intensity": {
                "mode_stats": {
                    "walk": {"spearman_rho": 0.0, "mean_direction_reversals": 3.0}
                }
            }
        }
    }
    flags = compute_flags(counterfactual=counterfactual)
    assert any(f["flag"] == "strong_counterfactual_reversal" for f in flags)


def test_empty_inputs_produce_no_flags():
    assert compute_flags() == []
    assert compute_flags(repeatability={}, counterfactual={}, persona={}, pilot_records=[]) == []
