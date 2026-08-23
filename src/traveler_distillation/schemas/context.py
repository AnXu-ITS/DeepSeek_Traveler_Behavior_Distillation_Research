"""Dynamic context schema."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

WeatherCondition = Literal["clear", "rain", "snow", "heat", "cold", "wind", "other"]


class Weather(BaseModel):
    condition: WeatherCondition
    intensity: float = Field(ge=0.0, le=1.0)


class DynamicContext(BaseModel):
    """Time-varying environment conditions affecting travel."""

    context_id: str
    weather: Weather
    road_congestion: float = Field(ge=0.0, le=1.0)
    transit_delay_min: int = Field(ge=0)
    transit_disruption: bool
    road_disruption: bool
    fare_multiplier: float = Field(gt=0.0)
    parking_cost_multiplier: float = Field(gt=0.0)
    congestion_charge: float = Field(ge=0.0)
