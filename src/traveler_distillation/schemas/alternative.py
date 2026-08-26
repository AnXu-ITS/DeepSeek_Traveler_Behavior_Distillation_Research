"""Travel alternative schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class TravelAlternative(BaseModel):
    """Attributes of one transportation option.

    ``mode`` is intentionally NOT a closed enum: v0.1 generators use
    ``car``/``pt``/``bike``/``walk``, but future modes (metro, bus, ridehail,
    taxi, drt, shared_bike, ...) must remain representable.

    S8 (transit accessibility adaptation) adds OPTIONAL per-alternative fields
    with zero defaults, so legacy states validate identically and the frozen
    S7-W3 feature extractor (which only reads the six core attributes) is
    unaffected. Only the S8 extractor reads the new fields; non-pt alternatives
    keep them at 0.
    """

    mode: str = Field(min_length=1)
    available: bool
    travel_time_min: float = Field(gt=0.0)
    monetary_cost: float = Field(ge=0.0)
    access_time_min: float = Field(ge=0.0)
    transfers: int = Field(ge=0)
    reliability_delay_min: float = Field(ge=0.0)
    weather_exposure: float = Field(ge=0.0, le=1.0)
    # ---- S8 optional real-supply accessibility attributes (city-independent) ----
    pt_feasible: float = Field(default=0.0, ge=0.0, le=1.0)
    egress_time_min: float = Field(default=0.0, ge=0.0)
    wait_time_min: float = Field(default=0.0, ge=0.0)
    in_vehicle_time_min: float = Field(default=0.0, ge=0.0)
    transfer_time_min: float = Field(default=0.0, ge=0.0)
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
