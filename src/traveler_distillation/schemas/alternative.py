"""Travel alternative schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TravelAlternative(BaseModel):
    """Attributes of one transportation option.

    ``mode`` is intentionally NOT a closed enum: v0.1 generators use
    ``car``/``pt``/``bike``/``walk``, but future modes (metro, bus, ridehail,
    taxi, drt, shared_bike, ...) must remain representable.
    """

    mode: str = Field(min_length=1)
    available: bool
    travel_time_min: float = Field(gt=0.0)
    monetary_cost: float = Field(ge=0.0)
    access_time_min: float = Field(ge=0.0)
    transfers: int = Field(ge=0)
    reliability_delay_min: float = Field(ge=0.0)
    weather_exposure: float = Field(ge=0.0, le=1.0)
