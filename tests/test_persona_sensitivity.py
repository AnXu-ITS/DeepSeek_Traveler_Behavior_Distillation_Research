"""Persona sensitivity and audit flag tests."""
from __future__ import annotations

import pytest

from traveler_distillation.audit import analyze_persona, compute_flags, PersonaContrastRecord
from traveler_distillation.schemas.action import UniversalTravelerAction


def _action(mode, probs):
    return UniversalTravelerAction(
        selected_mode=mode, mode_probabilities=probs,
        departure_time_shift_min=0, confidence=0.8,
    )


def _record(i, probs, mode, income, gid="PG_income_000001"):
    p = {"persona_id": "P1", "age_group": "25-34", "income_group": income,
         "occupation": "office_worker", "household_size": 2, "has_children": False,
         "car_ownership": True, "driving_license": True, "bike_ownership": True,
         "transit_pass": False, "habitual_mode": "car", "schedule_flexibility": "medium",
         "mobility_limitation": "none"}
    from traveler_distillation.schemas.persona import Persona
    persona = Persona(**p)
    # minimal state
    from traveler_distillation.schemas.state import UniversalTravelerState
    from traveler_distillation.schemas.trip import Trip
    from traveler_distillation.schemas.context import DynamicContext, Weather
    from traveler_distillation.schemas.alternative import TravelAlternative
    trip = Trip(trip_id="T1", purpose="commute", origin_type="home", destination_type="work",
                distance_km=10.0, desired_departure_min=480, desired_arrival_min=540,
                time_constraint="hard")
    ctx = DynamicContext(context_id="C1", weather=Weather(condition="clear", intensity=0.0),
                         road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
                         road_disruption=False, fare_multiplier=1.0,
                         parking_cost_multiplier=1.0, congestion_charge=0.0)
    alt = TravelAlternative(mode="pt", available=True, travel_time_min=30.0,
                            monetary_cost=3.0, access_time_min=8.0, transfers=1,
                            reliability_delay_min=0.0, weather_exposure=0.4)
    state = UniversalTravelerState(persona=persona, trip=trip, context=ctx, alternatives=[alt])
    return PersonaContrastRecord(
        audit_state_id=f"A{i}", persona_group_id=gid, attribute="income_group",
        state_hash=f"h{i}", state=state, teacher=_action(mode, probs), timestamp="t",
    )


def test_analyze_persona_detects_difference():
    recs = [
        _record(0, {"pt": 0.2, "bike": 0.8}, "bike", "low"),
        _record(1, {"pt": 0.5, "bike": 0.5}, "bike", "medium"),
        _record(2, {"pt": 0.8, "bike": 0.2}, "pt", "high"),
    ]
    res = analyze_persona(recs)
    assert res["aggregate"]["n_groups"] == 1
    g = res["per_group"][0]
    assert g["mean_pairwise_l1"] > 0.3
    assert g["n_distinct_selected_modes"] == 2


def test_compute_flags_selected_mode_flip():
    repeatability = {
        "per_group": [
            {
                "state_hash": "h", "source_sample_id": "S1",
                "selected_mode_agreement": 0.33,
                "selected_modes": ["bike", "pt", "walk"],
                "mean_pairwise_l1": 0.1,
                "departure_shift": {"range": 0, "values": [0, 0, 0]},
                "confidence": {"std": 0.01},
                "probability_stability": {},
            }
        ]
    }
    flags = compute_flags(repeatability=repeatability)
    assert any(f["flag"] == "selected_mode_flip" for f in flags)


def test_compute_flags_positive_fare_response():
    counterfactual = {
        "axes": {
            "fare_multiplier": {
                "n_groups": 1,
                "mode_stats": {
                    "pt": {"spearman_rho": 0.5, "n_points": 3, "mean_direction_reversals": 0.0}
                },
            }
        }
    }
    flags = compute_flags(counterfactual=counterfactual)
    assert any(f["flag"] == "unexpected_positive_fare_response" for f in flags)


def test_compute_flags_strong_reversal():
    counterfactual = {
        "axes": {
            "weather_intensity": {
                "n_groups": 1,
                "mode_stats": {
                    "bike": {"spearman_rho": 0.0, "n_points": 5, "mean_direction_reversals": 2.0}
                },
            }
        }
    }
    flags = compute_flags(counterfactual=counterfactual)
    assert any(f["flag"] == "strong_counterfactual_reversal" for f in flags)
