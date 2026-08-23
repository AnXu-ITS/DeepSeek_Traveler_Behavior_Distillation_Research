"""Universal traveler state schema."""
from __future__ import annotations

from pydantic import BaseModel, model_validator

from .persona import Persona
from .trip import Trip
from .context import DynamicContext
from .alternative import TravelAlternative


class UniversalTravelerState(BaseModel):
    """A complete, simulator-independent description of one decision situation.

    persona + trip + dynamic context + available travel alternatives.
    """

    persona: Persona
    trip: Trip
    context: DynamicContext
    alternatives: list[TravelAlternative]

    @model_validator(mode="after")
    def _check_alternatives(self) -> "UniversalTravelerState":
        if not self.alternatives:
            raise ValueError("alternatives must not be empty")
        if not any(a.available for a in self.alternatives):
            raise ValueError("at least one alternative must be available")
        modes = [a.mode for a in self.alternatives]
        if len(modes) != len(set(modes)):
            raise ValueError("alternative modes must be unique")
        return self

    @property
    def available_modes(self) -> list[str]:
        return [a.mode for a in self.alternatives if a.available]
