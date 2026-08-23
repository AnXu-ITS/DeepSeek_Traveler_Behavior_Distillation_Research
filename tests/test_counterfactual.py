"""Counterfactual integrity tests."""
from __future__ import annotations

import pytest

from traveler_distillation.generators import CounterfactualContextGenerator


def _by_mode(state):
    return {a.mode: a for a in state.alternatives}


def test_weather_counterfactual_integrity(baseline_state, config):
    cf = CounterfactualContextGenerator(config)
    samples = cf.generate(baseline_state, "weather_intensity", [0.5])
    assert len(samples) == 1
    new_state = samples[0].state

    # persona & trip identical
    assert new_state.persona.model_dump() == baseline_state.persona.model_dump()
    assert new_state.trip.model_dump() == baseline_state.trip.model_dump()

    # only weather intensity differs; everything else in context identical.
    # Weather v0.1.1: intensity > 0 now implies condition = rain.
    base_ctx = baseline_state.context
    new_ctx = new_state.context
    assert new_ctx.weather.intensity == 0.5
    assert new_ctx.weather.condition == "rain"
    assert base_ctx.weather.condition == "clear"
    assert new_ctx.road_congestion == base_ctx.road_congestion
    assert new_ctx.transit_delay_min == base_ctx.transit_delay_min
    assert new_ctx.fare_multiplier == base_ctx.fare_multiplier
    assert new_ctx.parking_cost_multiplier == base_ctx.parking_cost_multiplier
    assert new_ctx.road_disruption == base_ctx.road_disruption
    assert new_ctx.transit_disruption == base_ctx.transit_disruption


def test_weather_counterfactual_skips_baseline_level(baseline_state, config):
    cf = CounterfactualContextGenerator(config)
    # 0.0 equals baseline -> skipped; 0.5 genuine -> kept
    samples = cf.generate(baseline_state, "weather_intensity", [0.0, 0.5])
    assert [s.level for s in samples] == [0.5]


def test_fare_counterfactual_changes_pt_cost_only(baseline_state, config):
    cf = CounterfactualContextGenerator(config)
    samples = cf.generate(baseline_state, "fare_multiplier", [2.0])
    assert len(samples) == 1
    base_alt = _by_mode(baseline_state)
    new_alt = _by_mode(samples[0].state)

    # PT monetary cost doubles
    assert new_alt["pt"].monetary_cost == pytest.approx(base_alt["pt"].monetary_cost * 2.0, rel=1e-6)
    # other modes unchanged
    for mode in ("car", "bike", "walk"):
        assert new_alt[mode].monetary_cost == pytest.approx(base_alt[mode].monetary_cost, rel=1e-6)


def test_congestion_increases_car_time(baseline_state, config):
    cf = CounterfactualContextGenerator(config)
    samples = cf.generate(baseline_state, "road_congestion", [0.8])
    base_car = _by_mode(baseline_state)["car"]
    new_car = _by_mode(samples[0].state)["car"]
    assert new_car.travel_time_min > base_car.travel_time_min
    assert new_car.reliability_delay_min > base_car.reliability_delay_min
