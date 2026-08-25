"""Dataset sample schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .state import UniversalTravelerState
from .action import UniversalTravelerAction


class Perturbation(BaseModel):
    """Which context axis was changed and to what level.

    ``level`` may be a float (weather intensity, multipliers), an int
    (transit delay minutes) or a bool (road disruption).

    For S5 multi-axis (joint) samples, ``axis == "joint"`` and ``level`` is a
    dummy (0.0) while ``joint_axes`` carries the ordered list of
    ``{"axis": ..., "level": ...}`` entries. This keeps single-axis consumers
    backward compatible (they skip/ignore non-baseline axis values) while
    giving joint states an explicit, machine-readable decomposition.
    """

    axis: str = Field(min_length=1)
    level: float | int | bool
    joint_axes: list[dict] = Field(default_factory=list)


class TeacherMetadata(BaseModel):
    model: str
    prompt_version: str
    dataset_version: str


class TeacherDatasetSample(BaseModel):
    """One row of the teacher dataset.

    ``counterfactual_group_id`` links samples that share the same persona+trip
    and differ only by one perturbation axis. ``persona_group_id`` is reserved
    for future persona-contrast groups (may be null). ``baseline_sample_id``
    links a counterfactual sample to its baseline reference sample.
    """

    sample_id: str
    counterfactual_group_id: str | None = None
    persona_group_id: str | None = None
    baseline_sample_id: str | None = None
    perturbation: Perturbation
    state: UniversalTravelerState
    teacher: UniversalTravelerAction | None = None
    teacher_metadata: TeacherMetadata


class FailedSample(BaseModel):
    """A sample that could not be produced (parse/validation/API failure)."""

    sample_id: str
    counterfactual_group_id: str | None = None
    persona_group_id: str | None = None
    baseline_sample_id: str | None = None
    perturbation: Perturbation | None = None
    failure_type: str
    attempts: int
    error: str = ""
    state: UniversalTravelerState | None = None
