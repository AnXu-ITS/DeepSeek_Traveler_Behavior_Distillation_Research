"""Shared fixtures for the test suite."""
from __future__ import annotations

import pytest

from traveler_distillation.schemas.persona import Persona
from traveler_distillation.schemas.trip import Trip
from traveler_distillation.generators import PersonaGenerator, TripGenerator, BaselineStateGenerator


@pytest.fixture
def config() -> dict:
    return {
        "dataset": {"version": "test_v0.1"},
        "generation": {"seed": 42},
        "persona": {
            "age_group": {"18-24": 0.2, "25-34": 0.3, "35-44": 0.3, "45-64": 0.15, "65+": 0.05},
            "income_group": {"low": 0.4, "medium": 0.4, "high": 0.2},
            "occupation": {
                "student": 0.2, "office_worker": 0.3, "service_worker": 0.2,
                "manual_worker": 0.1, "retired": 0.1, "unemployed": 0.05, "other": 0.05,
            },
            "household_size": {"min": 1, "max": 4},
            "has_children_prob": 0.3,
            "car_ownership_prob": 0.6,
            "driving_license_prob": 0.7,
            "bike_ownership_prob": 0.4,
            "transit_pass_prob": 0.3,
            "habitual_mode": {"car": 0.4, "pt": 0.3, "bike": 0.1, "walk": 0.1, "mixed": 0.1},
            "schedule_flexibility": {"low": 0.3, "medium": 0.5, "high": 0.2},
            "mobility_limitation": {"none": 0.9, "mild": 0.07, "significant": 0.03},
        },
        "trip": {
            "purposes": {"commute": 0.4, "education": 0.15, "shopping": 0.2, "leisure": 0.25},
            "origin_type": "home",
            "templates": {
                "commute": {
                    "distance_km": [2, 30], "departure_window_min": [420, 540],
                    "arrival_window_min": [480, 600], "time_constraint": "hard",
                    "destination_type": "work",
                },
                "education": {
                    "distance_km": [1, 15], "departure_window_min": [420, 520],
                    "arrival_window_min": [480, 570], "time_constraint": "hard",
                    "destination_type": "school",
                },
                "shopping": {
                    "distance_km": [0.5, 12], "departure_window_min": [540, 1080],
                    "arrival_window_min": [600, 1140], "time_constraint": "soft",
                    "destination_type": "shop",
                },
                "leisure": {
                    "distance_km": [0.5, 20], "departure_window_min": [600, 1200],
                    "arrival_window_min": [660, 1260], "time_constraint": "soft",
                    "destination_type": "leisure",
                },
            },
        },
        "alternatives": {},
        "baseline_context": {},
        "perturbations": {
            "weather_intensity": {"levels": [0.0, 0.25, 0.5, 0.75, 1.0]},
            "road_congestion": {"levels": [0.2, 0.4, 0.6, 0.8]},
            "transit_delay": {"levels": [0, 5, 15, 30]},
            "fare_multiplier": {"levels": [1.0, 1.25, 1.5, 2.0]},
            "parking_cost_multiplier": {"levels": [1.0, 1.5, 2.0, 3.0]},
            "road_disruption": {"levels": [False, True]},
        },
    }


@pytest.fixture
def persona(config) -> Persona:
    return PersonaGenerator(seed=42, config=config).generate(1)[0]


@pytest.fixture
def trip(config) -> Trip:
    return TripGenerator(seed=42, config=config).generate(1)[0]


@pytest.fixture
def baseline_state(config, persona, trip):
    return BaselineStateGenerator(config).generate(persona, trip)
