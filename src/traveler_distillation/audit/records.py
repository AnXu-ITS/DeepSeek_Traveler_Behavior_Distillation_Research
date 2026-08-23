"""Audit record models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ..schemas.state import UniversalTravelerState
from ..schemas.action import UniversalTravelerAction
from ..schemas.dataset import TeacherMetadata


class RepeatabilityRecord(BaseModel):
    """One repeated teacher call for the same state."""

    audit_state_id: str
    source_sample_id: str
    repeat_index: int
    state_hash: str
    state: UniversalTravelerState
    teacher: UniversalTravelerAction | None = None
    teacher_metadata: TeacherMetadata | None = None
    timestamp: str
    force_repeat: bool = True


class PersonaContrastRecord(BaseModel):
    """A persona-contrast or ablation teacher call."""

    audit_state_id: str
    kind: Literal["persona_contrast", "ablation"] = "persona_contrast"
    persona_group_id: str | None = None
    attribute: str | None = None
    state_hash: str
    state: UniversalTravelerState
    teacher: UniversalTravelerAction | None = None
    teacher_metadata: TeacherMetadata | None = None
    timestamp: str
    note: str = ""
