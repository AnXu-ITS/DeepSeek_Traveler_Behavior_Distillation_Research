"""Generator tests (reproducibility + availability rules)."""
from __future__ import annotations

from traveler_distillation.generators import PersonaGenerator, TripGenerator, AlternativeGenerator
from traveler_distillation.schemas.persona import Persona
from traveler_distillation.schemas.trip import Trip
from traveler_distillation.schemas.context import DynamicContext, Weather


def test_persona_generator_reproducible(config):
    a = PersonaGenerator(seed=42, config=config).generate(20)
    b = PersonaGenerator(seed=42, config=config).generate(20)
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]


def test_persona_generator_different_seed(config):
    a = PersonaGenerator(seed=1, config=config).generate(20)
    b = PersonaGenerator(seed=2, config=config).generate(20)
    assert [p.model_dump() for p in a] != [p.model_dump() for p in b]


def test_persona_count(config):
    assert len(PersonaGenerator(seed=42, config=config).generate(5)) == 5


def test_trip_generator_reproducible(config):
    a = TripGenerator(seed=42, config=config).generate(20)
    b = TripGenerator(seed=42, config=config).generate(20)
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]


def test_trip_times_consistent(config):
    trips = TripGenerator(seed=7, config=config).generate(50)
    for t in trips:
        assert 0 <= t.desired_departure_min < 1440
        assert 0 <= t.desired_arrival_min < 1440
        assert t.distance_km > 0
        assert t.desired_arrival_min > t.desired_departure_min


def _persona(**overrides):
    base = {
        "persona_id": "P1",
        "age_group": "25-34",
        "income_group": "medium",
        "occupation": "office_worker",
        "household_size": 2,
        "has_children": False,
        "car_ownership": True,
        "driving_license": True,
        "bike_ownership": True,
        "transit_pass": False,
        "habitual_mode": "car",
        "schedule_flexibility": "medium",
        "mobility_limitation": "none",
    }
    base.update(overrides)
    return Persona(**base)


def _trip():
    return Trip(
        trip_id="T1", purpose="commute", origin_type="home", destination_type="work",
        distance_km=10.0, desired_departure_min=480, desired_arrival_min=540,
        time_constraint="hard",
    )


def _context():
    return DynamicContext(
        context_id="C1", weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )


def _by_mode(alts):
    return {a.mode: a for a in alts}


def test_car_unavailable_without_license(config):
    alts = _by_mode(AlternativeGenerator(config).generate(_persona(driving_license=False), _trip(), _context()))
    assert alts["car"].available is False


def test_car_unavailable_without_ownership(config):
    alts = _by_mode(AlternativeGenerator(config).generate(_persona(car_ownership=False), _trip(), _context()))
    assert alts["car"].available is False


def test_bike_unavailable_without_ownership(config):
    alts = _by_mode(AlternativeGenerator(config).generate(_persona(bike_ownership=False), _trip(), _context()))
    assert alts["bike"].available is False


def test_walk_always_available(config):
    alts = _by_mode(AlternativeGenerator(config).generate(_persona(), _trip(), _context()))
    assert alts["walk"].available is True
