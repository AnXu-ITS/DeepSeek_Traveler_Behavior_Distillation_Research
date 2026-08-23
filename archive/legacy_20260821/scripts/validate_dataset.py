#!/usr/bin/env python
"""Validate a teacher dataset JSONL file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.schemas.dataset import TeacherDatasetSample
from traveler_distillation.teacher import TeacherResponseValidator
from pydantic import ValidationError


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="path to teacher_dataset JSONL")
    ap.add_argument("--failed", default=None, help="optional failed_samples JSONL")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    validator = TeacherResponseValidator(probability_tolerance=args.tolerance)
    rows = 0
    valid = 0
    invalid = 0
    warnings = 0

    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                obj = json.loads(line)
                sample = TeacherDatasetSample.model_validate(obj)
            except (json.JSONDecodeError, ValidationError) as exc:
                invalid += 1
                print(f"[invalid] row {rows}: {exc}")
                continue

            # state-level checks already enforced by schema validation; verify
            # teacher vs state consistency if a teacher is present.
            if sample.teacher is not None:
                res = validator.validate(sample.state, sample.teacher)
                if not res.valid:
                    warnings += 1
                    print(f"[warning] {sample.sample_id}: teacher invalid -> {res.reason}")
            else:
                warnings += 1
                print(f"[warning] {sample.sample_id}: no teacher response")

            # baseline link presence for counterfactual samples
            if sample.counterfactual_group_id is not None and sample.baseline_sample_id is None:
                warnings += 1
                print(f"[warning] {sample.sample_id}: counterfactual missing baseline_sample_id")

            valid += 1

    print("\n" + "=" * 50)
    print("DATASET VALIDATION")
    print("=" * 50)
    print(f"rows scanned: {rows}")
    print(f"valid: {valid}")
    print(f"invalid: {invalid}")
    print(f"warnings: {warnings}")
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
