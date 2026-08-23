"""Variable-choice-set student tests: different available modes in one batch."""
from __future__ import annotations

import pytest
import torch

from traveler_distillation.schemas.persona import Persona
from traveler_distillation.schemas.trip import Trip
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.generators import AlternativeGenerator
from traveler_distillation.student import FeatureExtractor, TravelerStudent, collate_batch


def _persona(pid, car=False, license=False, bike=False, **kw):
    base = {
        "persona_id": pid,
        "age_group": "25-34",
        "income_group": "medium",
        "occupation": "office_worker",
        "household_size": 2,
        "has_children": False,
        "car_ownership": car,
        "driving_license": license,
        "bike_ownership": bike,
        "transit_pass": False,
        "habitual_mode": "car",
        "schedule_flexibility": "medium",
        "mobility_limitation": "none",
    }
    base.update(kw)
    return Persona(**base)


def _trip(tid="T1"):
    return Trip(
        trip_id=tid,
        purpose="commute",
        origin_type="home",
        destination_type="work",
        distance_km=8.0,
        desired_departure_min=480,
        desired_arrival_min=540,
        time_constraint="hard",
    )


def _context(cid="C1"):
    return DynamicContext(
        context_id=cid,
        weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3,
        transit_delay_min=0,
        transit_disruption=False,
        road_disruption=False,
        fare_multiplier=1.0,
        parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )


def _state(pid, car=False, license=False, bike=False, **kw):
    persona = _persona(pid, car=car, license=license, bike=bike, **kw)
    trip = _trip(f"T_{pid}")
    ctx = _context(f"C_{pid}")
    alts = AlternativeGenerator({}).generate(persona, trip, ctx)
    return UniversalTravelerState(persona=persona, trip=trip, context=ctx, alternatives=alts)


def _make_batch(states, extractor):
    items = []
    for s in states:
        f = extractor.encode(s)
        items.append(
            {
                "global_cat": torch.tensor(f["global_cat"], dtype=torch.long),
                "global_num": torch.tensor(f["global_num"], dtype=torch.float32),
                "alt_mode_idx": torch.tensor(f["alt_mode_idx"], dtype=torch.long),
                "alt_num": torch.tensor(f["alt_num"], dtype=torch.float32),
                "alt_mask": torch.tensor(f["alt_available"], dtype=torch.float32),
            }
        )
    return collate_batch(items)


def test_variable_choice_set_one_batch():
    states = [
        _state("A", car=False, license=False, bike=True),   # pt, bike, walk
        _state("B", car=True, license=True, bike=False),    # car, pt, walk
        _state("C", car=True, license=True, bike=True),     # car, pt, bike, walk
    ]
    assert [s.available_modes for s in states] == [
        ["pt", "bike", "walk"],
        ["car", "pt", "walk"],
        ["car", "pt", "bike", "walk"],
    ]

    extractor = FeatureExtractor().fit(states)
    model = TravelerStudent(
        {"global_hidden_dim": 16, "alternative_hidden_dim": 8, "scorer_hidden_dim": 16,
         "cat_embedding_dim": 4, "mode_embedding_dim": 4, "dropout": 0.0},
        extractor.spec,
    )
    model.eval()
    batch = _make_batch(states, extractor)
    out = model(batch)
    probs = out["mode_probabilities"]

    # each row sums to 1
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)
    # padded positions (mask=0) have zero probability
    mask = batch["alt_mask"]
    assert ((probs * (1.0 - mask)).abs() < 1e-5).all()
    # departure within [-60, 60]
    assert (out["departure_time_shift_min"].abs() <= 60.0 + 1e-4).all()
