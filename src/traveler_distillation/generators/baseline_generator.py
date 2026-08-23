"""Baseline state generator."""
from __future__ import annotations

from ..schemas.persona import Persona
from ..schemas.trip import Trip
from ..schemas.context import DynamicContext, Weather
from ..schemas.state import UniversalTravelerState
from .alternative_generator import AlternativeGenerator


class BaselineStateGenerator:
    """Build the baseline (unperturbed) state for a persona+trip pair."""

    def __init__(self, config: dict | None = None):
        self.cfg = (config or {}).get("baseline_context", {})
        self.alt_gen = AlternativeGenerator(config)

    def generate(self, persona: Persona, trip: Trip) -> UniversalTravelerState:
        context = self.baseline_context(persona.persona_id, trip.trip_id)
        alternatives = self.alt_gen.generate(persona, trip, context)
        return UniversalTravelerState(
            persona=persona, trip=trip, context=context, alternatives=alternatives
        )

    def baseline_context(self, persona_id: str, trip_id: str) -> DynamicContext:
        return DynamicContext(
            context_id=f"C_BASE_{persona_id}_{trip_id}",
            weather=Weather(
                condition=self.cfg.get("weather_condition", "clear"),
                intensity=self.cfg.get("weather_intensity", 0.0),
            ),
            road_congestion=self.cfg.get("road_congestion", 0.3),
            transit_delay_min=self.cfg.get("transit_delay_min", 0),
            transit_disruption=self.cfg.get("transit_disruption", False),
            road_disruption=self.cfg.get("road_disruption", False),
            fare_multiplier=self.cfg.get("fare_multiplier", 1.0),
            parking_cost_multiplier=self.cfg.get("parking_cost_multiplier", 1.0),
            congestion_charge=self.cfg.get("congestion_charge", 0.0),
        )
