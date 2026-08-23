"""Counterfactual context generator.

For a single persona+trip pair, this produces states where exactly ONE context
axis is changed at a time while everything else stays fixed, then recalculates
alternative attributes so context and alternatives stay internally consistent.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas.context import DynamicContext
from ..schemas.state import UniversalTravelerState
from .alternative_generator import AlternativeGenerator
from .context_perturbation import perturb_context


@dataclass
class CounterfactualSample:
    axis: str
    level: float | int | bool
    state: UniversalTravelerState


class CounterfactualContextGenerator:
    def __init__(self, config: dict | None = None):
        self.alt_gen = AlternativeGenerator(config)

    def generate(
        self,
        baseline_state: UniversalTravelerState,
        axis: str,
        levels: list[float | int | bool],
    ) -> list[CounterfactualSample]:
        """Produce counterfactual states for each level of ``axis``.

        Levels whose resulting context equals the baseline context are skipped
        (they would duplicate the baseline sample).
        """
        results: list[CounterfactualSample] = []
        persona = baseline_state.persona
        trip = baseline_state.trip
        baseline_ctx = baseline_state.context

        for level in levels:
            new_ctx = perturb_context(baseline_ctx, axis, level)
            if self._axis_equal(new_ctx, baseline_ctx, axis):
                continue
            new_ctx.context_id = f"C_CF_{axis}_{level}_{persona.persona_id}_{trip.trip_id}"
            state = UniversalTravelerState(
                persona=persona,
                trip=trip,
                context=new_ctx,
                alternatives=self.alt_gen.generate(persona, trip, new_ctx),
            )
            results.append(CounterfactualSample(axis=axis, level=level, state=state))
        return results

    @staticmethod
    def _axis_equal(a: DynamicContext, b: DynamicContext, axis: str) -> bool:
        getters = {
            "weather_intensity": lambda c: c.weather.intensity,
            "road_congestion": lambda c: c.road_congestion,
            "transit_delay": lambda c: c.transit_delay_min,
            "fare_multiplier": lambda c: c.fare_multiplier,
            "parking_cost_multiplier": lambda c: c.parking_cost_multiplier,
            "road_disruption": lambda c: c.road_disruption,
        }
        getter = getters.get(axis)
        if getter is None:
            return False
        va, vb = getter(a), getter(b)
        if isinstance(va, float) or isinstance(vb, float):
            return abs(float(va) - float(vb)) < 1e-9
        return va == vb
