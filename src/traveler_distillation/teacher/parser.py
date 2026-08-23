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

    def parse(self, raw: str) -> UniversalTravelerAction:
        text = self._strip_fences(raw.strip())
        data = self._load_json(text)
        try:
            return UniversalTravelerAction.model_validate(data)
        except ValidationError as exc:
            raise TeacherParseError(f"action schema validation failed: {exc}")

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
