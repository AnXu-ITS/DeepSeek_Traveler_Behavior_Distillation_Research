"""Dataset statistics."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class StatisticsAccumulator:
    """Incremental statistics; avoids holding all samples in memory."""

    def __init__(self):
        self.total_samples = 0
        self.valid_samples = 0
        self.failed_samples = 0
        self.parse_failures = 0
        self.validation_failures = 0
        self.api_failures = 0
        self.probability_sum_failures = 0
        self.persona_ids: set[str] = set()
        self.trip_ids: set[str] = set()
        self.cf_group_ids: set[str] = set()
        self.samples_by_axis: Counter = Counter()
        self.mode_distribution: Counter = Counter()
        self.confidence_sum = 0.0
        self.confidence_count = 0

    def record_valid(self, sample) -> None:
        self.total_samples += 1
        self.valid_samples += 1
        self.persona_ids.add(sample.state.persona.persona_id)
        self.trip_ids.add(sample.state.trip.trip_id)
        if sample.counterfactual_group_id:
            self.cf_group_ids.add(sample.counterfactual_group_id)
        self.samples_by_axis[sample.perturbation.axis] += 1
        if sample.teacher is not None:
            self.mode_distribution[sample.teacher.selected_mode] += 1
            self.confidence_sum += sample.teacher.confidence
            self.confidence_count += 1

    def record_failure(self, failed) -> None:
        self.total_samples += 1
        self.failed_samples += 1
        if failed.state is not None:
            self.persona_ids.add(failed.state.persona.persona_id)
            self.trip_ids.add(failed.state.trip.trip_id)
        ft = failed.failure_type
        if ft == "parse_failure":
            self.parse_failures += 1
        elif ft in ("validation_failure",):
            self.validation_failures += 1
            if "probability_sum" in (failed.error or ""):
                self.probability_sum_failures += 1
        elif ft in ("api_error", "timeout", "config_error", "empty_content"):
            self.api_failures += 1

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "failed_samples": self.failed_samples,
            "num_personas": len(self.persona_ids),
            "num_trips": len(self.trip_ids),
            "num_counterfactual_groups": len(self.cf_group_ids),
            "samples_by_axis": dict(self.samples_by_axis),
            "selected_mode_distribution": dict(self.mode_distribution),
            "mean_teacher_confidence": round(
                self.confidence_sum / self.confidence_count, 4
            ) if self.confidence_count else None,
            "parse_failures": self.parse_failures,
            "validation_failures": self.validation_failures,
            "probability_sum_failures": self.probability_sum_failures,
            "api_failures": self.api_failures,
        }


def compute_statistics(dataset_path: str | Path, failed_path: str | Path | None = None) -> dict:
    """Read a dataset JSONL (and optional failed JSONL) and compute stats."""
    acc = StatisticsAccumulator()
    for line in Path(dataset_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        acc.valid_samples += 1
        acc.total_samples += 1
        state = obj.get("state", {})
        persona = state.get("persona", {})
        trip = state.get("trip", {})
        if persona.get("persona_id"):
            acc.persona_ids.add(persona["persona_id"])
        if trip.get("trip_id"):
            acc.trip_ids.add(trip["trip_id"])
        if obj.get("counterfactual_group_id"):
            acc.cf_group_ids.add(obj["counterfactual_group_id"])
        axis = (obj.get("perturbation") or {}).get("axis", "unknown")
        acc.samples_by_axis[axis] += 1
        teacher = obj.get("teacher")
        if teacher:
            acc.mode_distribution[teacher.get("selected_mode", "unknown")] += 1
            conf = teacher.get("confidence")
            if conf is not None:
                acc.confidence_sum += conf
                acc.confidence_count += 1

    if failed_path and Path(failed_path).exists():
        for line in Path(failed_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            acc.failed_samples += 1
            acc.total_samples += 1
            ft = obj.get("failure_type", "")
            if ft == "parse_failure":
                acc.parse_failures += 1
            elif ft == "validation_failure":
                acc.validation_failures += 1
            elif ft in ("api_error", "timeout", "config_error", "empty_content"):
                acc.api_failures += 1

    return acc.to_dict()
