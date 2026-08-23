"""Persona contrast generator (minimal v0.1 support).

Fixed Trip + Context + alternative generation rule, varying ONE persona
attribute at a time. Not generated at scale in v0.1, but the pipeline supports
it so future ``persona_group_id``-based heterogeneity analysis is possible.
"""
from __future__ import annotations

from ..schemas.persona import Persona
from ..schemas.trip import Trip
from ..schemas.context import DynamicContext
from ..schemas.state import UniversalTravelerState
from .alternative_generator import AlternativeGenerator

_SUPPORTED_ATTRIBUTES = ("income_group", "car_ownership", "schedule_flexibility")


class PersonaContrastGenerator:
    def __init__(self, config: dict | None = None):
        self.alt_gen = AlternativeGenerator(config)

    def generate(
        self,
        persona: Persona,
        trip: Trip,
        context: DynamicContext,
        attribute: str,
        values: list,
        group_id: str | None = None,
    ) -> list[UniversalTravelerState]:
        """Vary one persona attribute while keeping trip+context+generation rule fixed."""
        if attribute not in _SUPPORTED_ATTRIBUTES:
            raise ValueError(
                f"unsupported persona contrast attribute '{attribute}'. "
                f"Supported: {_SUPPORTED_ATTRIBUTES}"
            )
        states = []
        for value in values:
            p = persona.model_copy(deep=True)
            setattr(p, attribute, value)
            ctx = context.model_copy(deep=True)
            ctx.context_id = f"C_PG_{attribute}_{value}_{persona.persona_id}_{trip.trip_id}"
            states.append(
                UniversalTravelerState(
                    persona=p,
                    trip=trip,
                    context=ctx,
                    alternatives=self.alt_gen.generate(p, trip, ctx),
                )
            )
        return states
