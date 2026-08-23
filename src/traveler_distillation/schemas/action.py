"""Universal traveler action / teacher response schema."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class UniversalTravelerAction(BaseModel):
    """Structured behavioral response produced by the teacher.

    ``mode_probabilities`` maps mode -> probability and must support a variable
    choice set (only available alternatives are expected). Range checks here are
    hard schema constraints; state-dependent checks (selected-mode availability,
    key set, sum-to-one) live in ``TeacherResponseValidator``.
    """

    selected_mode: str = Field(min_length=1)
    mode_probabilities: dict[str, float]
    departure_time_shift_min: int = Field(ge=-60, le=60)
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str] = Field(default_factory=list)

    @field_validator("departure_time_shift_min", mode="before")
    @classmethod
    def _coerce_shift(cls, v):
        # Reasoning models often emit a float (e.g. 0.0, 5.5) for an integer
        # minutes-shift field. Round to nearest integer; the range check below
        # still applies afterward (out-of-range values are NOT silently clamped).
        if isinstance(v, bool):
            return v
        if isinstance(v, float):
            return int(round(v))
        if isinstance(v, str):
            s = v.strip()
            try:
                return int(s)
            except ValueError:
                try:
                    return int(round(float(s)))
                except ValueError:
                    return v
        return v

    @field_validator("mode_probabilities")
    @classmethod
    def _check_probabilities(cls, v: dict[str, float]) -> dict[str, float]:
        for mode, p in v.items():
            if not isinstance(p, (int, float)):
                raise ValueError(f"probability for '{mode}' is not numeric: {p!r}")
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probability for '{mode}' out of [0,1]: {p}")
        return v
