"""Schema validation tests."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from traveler_distillation.schemas.persona import Persona
from traveler_distillation.schemas.trip import Trip
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.schemas.alternative import TravelAlternative
from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.schemas.state import UniversalTravelerState


def make_persona(**overrides) -> dict:
    base = {
        "persona_id": "P000001",
        "age_group": "25-34",
        "income_group": "medium",
        "occupation": "office_worker",
        "household_size": 2,
        "has_children": False,
        "car_ownership": True,
        "driving_license": True,
        "bike_ownership": False,
        "transit_pass": False,
        "habitual_mode": "car",
        "schedule_flexibility": "medium",
        "mobility_limitation": "none",
    }
    base.update(overrides)
    return base


def make_trip(**overrides) -> dict:
    base = {
        "trip_id": "T000001",
        "purpose": "commute",
        "origin_type": "home",
        "destination_type": "work",
        "distance_km": 10.0,
        "desired_departure_min": 480,
        "desired_arrival_min": 540,
        "time_constraint": "hard",
    }
    base.update(overrides)
    return base


def make_context(**overrides) -> dict:
    base = {
        "context_id": "C000001",
        "weather": {"condition": "clear", "intensity": 0.0},
        "road_congestion": 0.3,
        "transit_delay_min": 0,
        "transit_disruption": False,
        "road_disruption": False,
        "fare_multiplier": 1.0,
        "parking_cost_multiplier": 1.0,
        "congestion_charge": 0.0,
    }
    base.update(overrides)
    return base


def make_alternative(mode="car", **overrides) -> dict:
    base = {
        "mode": mode,
        "available": True,
        "travel_time_min": 20.0,
        "monetary_cost": 5.0,
        "access_time_min": 3.0,
        "transfers": 0,
        "reliability_delay_min": 0.0,
        "weather_exposure": 0.05,
    }
    base.update(overrides)
    return base


# ----------------------------------------------------------------- Persona
def test_valid_persona():
    p = Persona(**make_persona())
    assert p.model_dump()["persona_id"] == "P000001"


def test_persona_household_size_must_be_at_least_one():
    with pytest.raises(ValidationError):
        Persona(**make_persona(household_size=0))


# ----------------------------------------------------------------- Context
def test_context_weather_intensity_out_of_range():
    ctx = make_context(weather={"condition": "clear", "intensity": 1.5})
    with pytest.raises(ValidationError):
        DynamicContext(**ctx)


# ----------------------------------------------------------------- Alternative
def test_alternative_negative_cost_fails():
    with pytest.raises(ValidationError):
        TravelAlternative(**make_alternative(monetary_cost=-1.0))


# ----------------------------------------------------------------- Action
def test_action_confidence_out_of_range():
    with pytest.raises(ValidationError):
        UniversalTravelerAction(
            selected_mode="car",
            mode_probabilities={"car": 1.0},
            departure_time_shift_min=0,
            confidence=1.5,
        )


# ----------------------------------------------------------------- State
def test_state_requires_available_alternative():
    persona = Persona(**make_persona())
    trip = Trip(**make_trip())
    ctx = DynamicContext(**make_context())
    alt = TravelAlternative(**make_alternative(available=False))
    with pytest.raises(ValidationError):
        UniversalTravelerState(persona=persona, trip=trip, context=ctx, alternatives=[alt])


def test_state_rejects_empty_alternatives():
    persona = Persona(**make_persona())
    trip = Trip(**make_trip())
    ctx = DynamicContext(**make_context())
    with pytest.raises(ValidationError):
        UniversalTravelerState(persona=persona, trip=trip, context=ctx, alternatives=[])


def test_state_ok_with_one_available():
    persona = Persona(**make_persona())
    trip = Trip(**make_trip())
    ctx = DynamicContext(**make_context())
    alt = TravelAlternative(**make_alternative(available=True))
    state = UniversalTravelerState(persona=persona, trip=trip, context=ctx, alternatives=[alt])
    assert state.available_modes == ["car"]
