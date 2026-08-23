"""Persona schema."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AgeGroup = Literal["18-24", "25-34", "35-44", "45-64", "65+"]
IncomeGroup = Literal["low", "medium", "high"]
Occupation = Literal[
    "student",
    "office_worker",
    "service_worker",
    "manual_worker",
    "retired",
    "unemployed",
    "other",
]
HabitualMode = Literal["car", "pt", "bike", "walk", "mixed"]
ScheduleFlexibility = Literal["low", "medium", "high"]
MobilityLimitation = Literal["none", "mild", "significant"]


class Persona(BaseModel):
    """Static characteristics of a synthetic traveler."""

    persona_id: str
    age_group: AgeGroup
    income_group: IncomeGroup
    occupation: Occupation
    household_size: int = Field(ge=1)
    has_children: bool
    car_ownership: bool
    driving_license: bool
    bike_ownership: bool
    transit_pass: bool
    habitual_mode: HabitualMode
    schedule_flexibility: ScheduleFlexibility
    mobility_limitation: MobilityLimitation
