"""Validate a teacher action against the state it responds to."""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas.state import UniversalTravelerState
from ..schemas.action import UniversalTravelerAction


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ""


class TeacherResponseValidator:
    """Check that a teacher action is consistent with the traveler state."""

    def __init__(
        self,
        probability_tolerance: float = 1e-3,
        departure_shift_min: int = -60,
        departure_shift_max: int = 60,
    ):
        self.probability_tolerance = probability_tolerance
        self.departure_shift_min = departure_shift_min
        self.departure_shift_max = departure_shift_max

    def validate(
        self, state: UniversalTravelerState, action: UniversalTravelerAction
    ) -> ValidationResult:
        available = set(state.available_modes)

        if action.selected_mode not in available:
            return ValidationResult(False, "selected_mode_unavailable")

        keys = set(action.mode_probabilities.keys())
        if keys != available:
            if available - keys:
                return ValidationResult(False, "missing_probability_key")
            if keys - available:
                return ValidationResult(False, "extra_probability_key")

        for mode, p in action.mode_probabilities.items():
            if not (0.0 <= p <= 1.0):
                return ValidationResult(False, "probability_out_of_range")

        total = sum(action.mode_probabilities.values())
        if abs(total - 1.0) > self.probability_tolerance:
            return ValidationResult(False, "probability_sum")

        if not (self.departure_shift_min <= action.departure_time_shift_min <= self.departure_shift_max):
            return ValidationResult(False, "departure_shift_out_of_range")

        if not (0.0 <= action.confidence <= 1.0):
            return ValidationResult(False, "confidence_out_of_range")

        return ValidationResult(True, "ok")
