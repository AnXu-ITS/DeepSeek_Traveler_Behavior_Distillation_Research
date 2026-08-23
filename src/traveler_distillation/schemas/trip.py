"""Trip schema."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TripPurpose = Literal[
    "commute",
    "education",
    "shopping",
    "leisure",
    "healthcare",
    "escort",
    "other",
]
TimeConstraint = Literal["soft", "medium", "hard"]


class Trip(BaseModel):
    """A single trip a traveler needs to make.

    All times are minutes after midnight (e.g. 08:00 == 480).
    """

    trip_id: str
    purpose: TripPurpose
    origin_type: str = Field(min_length=1)
    destination_type: str = Field(min_length=1)
    distance_km: float = Field(gt=0.0)
    desired_departure_min: int = Field(ge=0, lt=1440)
    desired_arrival_min: int = Field(ge=0, lt=1440)
    time_constraint: TimeConstraint
