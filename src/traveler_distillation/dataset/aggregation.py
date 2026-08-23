"""K=3 aggregated teacher target: mean behavioral preference distribution."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..schemas.state import UniversalTravelerState
from ..schemas.action import UniversalTravelerAction
from ..schemas.dataset import Perturbation


class TeacherAggregate(BaseModel):
    """Aggregated (K-call mean) teacher behavioral target."""

    mode_probabilities: dict[str, float]
    selected_mode: str
    departure_time_shift_min: float
    confidence_mean: float
    confidence_std: float = 0.0


class AggregationMetadata(BaseModel):
    k: int
    aggregation_method: str = "mean_probability"
    prompt_version: str
    model: str
    source_completion_ids: list[str] = Field(default_factory=list)


class AggregatedTeacherTarget(BaseModel):
    """One training sample: a state plus its K=3 aggregated teacher target."""

    sample_id: str
    counterfactual_group_id: str | None = None
    persona_group_id: str | None = None
    baseline_sample_id: str | None = None
    # Split grouping key: ties a baseline sample to ALL of its counterfactual
    # samples so elasticity curves stay intact in one split. Defaults to the
    # persona::trip pair key and is used by group-aware splitting.
    split_group_id: str | None = None
    perturbation: Perturbation
    state: UniversalTravelerState
    teacher_aggregate: TeacherAggregate
    aggregation_metadata: AggregationMetadata
    status: str = "complete"


def aggregate_repeats(actions: list[UniversalTravelerAction]) -> TeacherAggregate:
    """Aggregate K repeat actions into a mean behavioral distribution.

    - mode_probabilities = element-wise mean over repeats (renormalized to 1.0)
    - selected_mode = argmax of the MEAN distribution (NOT majority vote)
    - departure_time_shift_min = mean over repeats (kept as float)
    - confidence = mean / std (metadata only; not a training target)
    """
    if not actions:
        raise ValueError("cannot aggregate zero actions")

    modes: set[str] = set()
    for a in actions:
        modes |= set(a.mode_probabilities.keys())

    n = len(actions)
    mean_probs = {m: sum(a.mode_probabilities.get(m, 0.0) for a in actions) / n for m in modes}

    total = sum(mean_probs.values())
    if total <= 0:
        raise ValueError("aggregated probabilities sum to zero")
    mean_probs = {m: p / total for m, p in mean_probs.items()}

    selected_mode = max(mean_probs, key=mean_probs.get)
    mean_shift = sum(a.departure_time_shift_min for a in actions) / n
    confs = [float(a.confidence) for a in actions]
    conf_mean = sum(confs) / n
    conf_std = (sum((c - conf_mean) ** 2 for c in confs) / n) ** 0.5

    return TeacherAggregate(
        mode_probabilities={m: round(p, 6) for m, p in mean_probs.items()},
        selected_mode=selected_mode,
        departure_time_shift_min=round(mean_shift, 4),
        confidence_mean=round(conf_mean, 6),
        confidence_std=round(conf_std, 6),
    )
