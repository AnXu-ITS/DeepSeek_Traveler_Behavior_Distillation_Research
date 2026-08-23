"""Weather v0.1.1 semantics: weather_intensity is rain intensity.

intensity == 0.0  -> condition = clear
intensity >  0.0  -> condition = rain
"""
from __future__ import annotations

import pytest

from traveler_distillation.generators import CounterfactualContextGenerator
from traveler_distillation.generators.context_perturbation import perturb_context


def test_zero_intensity_is_clear(baseline_state):
    ctx = perturb_context(baseline_state.context, "weather_intensity", 0.0)
    assert ctx.weather.intensity == 0.0
    assert ctx.weather.condition == "clear"


@pytest.mark.parametrize("level", [0.25, 0.5, 0.75, 1.0])
def test_positive_intensity_is_rain(baseline_state, level):
    ctx = perturb_context(baseline_state.context, "weather_intensity", level)
    assert ctx.weather.intensity == level
    assert ctx.weather.condition == "rain"


def test_weather_perturbation_keeps_non_weather_axes_fixed(baseline_state, config):
    cf = CounterfactualContextGenerator(config)
    sample = cf.generate(baseline_state, "weather_intensity", [0.75])[0]

    base = baseline_state.context
    new = sample.state.context

    # weather changed per v0.1.1 semantics
    assert new.weather.intensity == 0.75
    assert new.weather.condition == "rain"
    assert base.weather.condition == "clear"

    # non-weather context axes unchanged
    assert new.road_congestion == base.road_congestion
    assert new.transit_delay_min == base.transit_delay_min
    assert new.transit_disruption == base.transit_disruption
    assert new.road_disruption == base.road_disruption
    assert new.fare_multiplier == base.fare_multiplier
    assert new.parking_cost_multiplier == base.parking_cost_multiplier
    assert new.congestion_charge == base.congestion_charge

    # persona & trip unchanged
    assert sample.state.persona.model_dump() == baseline_state.persona.model_dump()
    assert sample.state.trip.model_dump() == baseline_state.trip.model_dump()


def test_weather_recalculation_only_affects_time_exposed_modes(baseline_state, config):
    """Rain changes only time (via exposure) of bike/walk/pt; cost & non-weather
    alternative attributes (fare, parking) stay identical to baseline."""
    cf = CounterfactualContextGenerator(config)
    sample = cf.generate(baseline_state, "weather_intensity", [1.0])[0]

    base_alt = {a.mode: a for a in baseline_state.alternatives}
    new_alt = {a.mode: a for a in sample.state.alternatives}

    # bike/walk are the most weather-exposed; their travel time increases.
    assert new_alt["bike"].travel_time_min > base_alt["bike"].travel_time_min
    assert new_alt["walk"].travel_time_min > base_alt["walk"].travel_time_min

    # monetary cost for pt/car/bike/walk must not change with rain.
    for mode in ("car", "pt", "bike", "walk"):
        assert new_alt[mode].monetary_cost == pytest.approx(
            base_alt[mode].monetary_cost, rel=1e-6
        )
