"""Teacher validator tests."""
from __future__ import annotations

from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.teacher import TeacherResponseValidator


def _state_with_modes(modes: list[str]) -> UniversalTravelerState:
    # Build a minimal valid state with the given available modes.
    from traveler_distillation.schemas.persona import Persona
    from traveler_distillation.schemas.trip import Trip
    from traveler_distillation.schemas.context import DynamicContext, Weather
    from traveler_distillation.schemas.alternative import TravelAlternative

    persona = Persona(
        persona_id="P1", age_group="25-34", income_group="medium",
        occupation="office_worker", household_size=2, has_children=False,
        car_ownership=True, driving_license=True, bike_ownership=True,
        transit_pass=False, habitual_mode="car", schedule_flexibility="medium",
        mobility_limitation="none",
    )
    trip = Trip(
        trip_id="T1", purpose="commute", origin_type="home", destination_type="work",
        distance_km=10.0, desired_departure_min=480, desired_arrival_min=540,
        time_constraint="hard",
    )
    ctx = DynamicContext(
        context_id="C1", weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )
    alts = [
        TravelAlternative(
            mode=m, available=(m in modes), travel_time_min=20.0, monetary_cost=5.0,
            access_time_min=3.0, transfers=0, reliability_delay_min=0.0,
            weather_exposure=0.5,
        )
        for m in ["car", "pt", "bike", "walk"]
    ]
    return UniversalTravelerState(persona=persona, trip=trip, context=ctx, alternatives=alts)


def _action(selected, probs, shift=0, confidence=0.8):
    return UniversalTravelerAction(
        selected_mode=selected,
        mode_probabilities=probs,
        departure_time_shift_min=shift,
        confidence=confidence,
    )


def test_selected_unavailable_mode_invalid():
    v = TeacherResponseValidator()
    state = _state_with_modes(["pt", "bike"])
    action = _action("car", {"car": 0.5, "pt": 0.5})
    result = v.validate(state, action)
    assert result.valid is False
    assert result.reason == "selected_mode_unavailable"


def test_valid_distribution():
    v = TeacherResponseValidator()
    state = _state_with_modes(["pt", "bike"])
    action = _action("pt", {"pt": 0.5, "bike": 0.5})
    result = v.validate(state, action)
    assert result.valid is True


def test_probability_sum_invalid():
    v = TeacherResponseValidator()
    state = _state_with_modes(["pt", "bike"])
    action = _action("pt", {"pt": 0.6, "bike": 0.6})
    result = v.validate(state, action)
    assert result.valid is False
    assert result.reason == "probability_sum"


def test_missing_key_invalid():
    v = TeacherResponseValidator()
    state = _state_with_modes(["pt", "bike"])
    action = _action("pt", {"pt": 1.0})
    result = v.validate(state, action)
    assert result.valid is False
    assert result.reason == "missing_probability_key"


def test_extra_key_invalid():
    v = TeacherResponseValidator()
    state = _state_with_modes(["pt", "bike"])
    action = _action("pt", {"pt": 0.5, "bike": 0.5, "walk": 0.0})
    result = v.validate(state, action)
    assert result.valid is False
    assert result.reason == "extra_probability_key"


def test_no_silent_normalization():
    v = TeacherResponseValidator()
    state = _state_with_modes(["pt", "bike"])
    action = _action("pt", {"pt": 0.999999, "bike": 0.0})
    result = v.validate(state, action)
    # within tolerance -> valid (do not over-reject)
    assert result.valid is True
