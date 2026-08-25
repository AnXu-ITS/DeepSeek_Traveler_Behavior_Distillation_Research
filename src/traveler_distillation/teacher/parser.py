"""Parse raw teacher output into a structured action."""
from __future__ import annotations

import json

from pydantic import ValidationError

from ..schemas.action import UniversalTravelerAction


class TeacherParseError(Exception):
    pass


class TeacherResponseParser:
    """Parse raw API text -> UniversalTravelerAction.

    Handles markdown fences, surrounding whitespace, and stray prose. Does NOT
    attempt aggressive regex repair of malformed JSON: if the payload cannot be
    reliably parsed it is reported as a failure.
    """

    def parse(self, raw: str, clip_departure: bool = False) -> UniversalTravelerAction:
        text = self._strip_fences(raw.strip())
        data = self._load_json(text)
        if clip_departure and "departure_time_shift_min" in data:
            data["departure_time_shift_min"] = self._clip_shift(data["departure_time_shift_min"])
        try:
            return UniversalTravelerAction.model_validate(data)
        except ValidationError as exc:
            raise TeacherParseError(f"action schema validation failed: {exc}")

    @staticmethod
    def _clip_shift(v, lo: float = -60.0, hi: float = 60.0):
        """Clip a departure shift to the student's representable range [lo, hi].

        Used for S5 joint states: severe multi-axis contexts make the teacher
        suggest departures beyond ±60 min, which the lightweight student
        (60·tanh head) cannot represent. The MODE PROBABILITIES are unaffected;
        only the (secondary) departure target is clipped. Mirrors the S2
        "distillation range == student representable range" boundary, but keeps
        the state instead of discarding it.
        """
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, float)):
            return int(round(max(lo, min(hi, float(v)))))
        if isinstance(v, str):
            s = v.strip()
            try:
                return int(round(max(lo, min(hi, float(s)))))
            except ValueError:
                return v
        return v

    @staticmethod
    def _strip_fences(text: str) -> str:
        if text.startswith("```"):
            text = text[3:]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    @staticmethod
    def _load_json(text: str) -> dict:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise TeacherParseError("no JSON object found in teacher response")
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise TeacherParseError(f"malformed JSON in teacher response: {exc}")
        if not isinstance(data, dict):
            raise TeacherParseError("teacher response JSON is not an object")
        return data
