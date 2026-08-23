#!/usr/bin/env python
"""Analyze teacher audit outputs. Offline only — makes no API calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.schemas.dataset import TeacherDatasetSample
from traveler_distillation.audit import (
    RepeatabilityRecord,
    PersonaContrastRecord,
    analyze_repeatability,
    analyze_counterfactual,
    elasticity_preview,
    analyze_persona,
    compute_flags,
    generate_report,
    decide_recommendation,
)


def _read(path: Path, model):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(model.model_validate(json.loads(line)))
    return records


def _read_failed(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/teacher_audit_v0_1")
    args = ap.parse_args()
    out = Path(args.output_dir)

    pilot = _read(out / "pilot_samples.jsonl", TeacherDatasetSample)
    repeats = _read(out / "repeatability_samples.jsonl", RepeatabilityRecord)
    personas = _read(out / "persona_contrast_samples.jsonl", PersonaContrastRecord)
    failed = _read_failed(out / "failed_samples.jsonl")

    # ---- metrics ----
    repeatability = analyze_repeatability(repeats)
    counterfactual = analyze_counterfactual(pilot)
    elasticity = elasticity_preview(pilot)
    persona = analyze_persona(personas)
    flags = compute_flags(repeatability, counterfactual, persona, pilot)
    recommendation = decide_recommendation(repeatability, counterfactual, persona)

    # ---- API reliability stats ----
    n_pilot_valid = sum(1 for s in pilot if s.teacher is not None)
    n_failed = len(failed)
    parse_failures = sum(1 for f in failed if f.get("failure_type") == "parse_failure")
    validation_failures = sum(1 for f in failed if f.get("failure_type") == "validation_failure")
    api_failures = sum(
        1 for f in failed if f.get("failure_type") in ("api_error", "timeout", "empty_content", "config_error")
    )
    n_repeat_ok = sum(1 for r in repeats if r.teacher is not None)
    n_persona_ok = sum(1 for p in personas if p.teacher is not None)

    api_stats = {
        "pilot": {
            "attempted": n_pilot_valid + n_failed,
            "valid": n_pilot_valid,
            "failed": n_failed,
            "parse_failures": parse_failures,
            "validation_failures": validation_failures,
            "api_failures": api_failures,
        },
        "repeatability": {
            "attempted": len(repeats),
            "valid": n_repeat_ok,
            "failed": len(repeats) - n_repeat_ok,
        },
        "persona_contrast_and_ablation": {
            "attempted": len(personas),
            "valid": n_persona_ok,
            "failed": len(personas) - n_persona_ok,
        },
        "total_attempted": (n_pilot_valid + n_failed) + len(repeats) + len(personas),
    }

    # ---- write metrics ----
    _write_json(out / "repeatability_metrics.json", repeatability)
    _write_json(out / "counterfactual_metrics.json", counterfactual)
    _write_json(out / "persona_sensitivity_metrics.json", persona)
    _write_json(out / "teacher_elasticity_preview.json", elasticity)

    with (out / "audit_flags.jsonl").open("w", encoding="utf-8") as f:
        for fl in flags:
            f.write(json.dumps(fl, ensure_ascii=False) + "\n")

    report = generate_report(api_stats, repeatability, counterfactual, persona, elasticity, flags, recommendation)
    (out / "teacher_audit_report.md").write_text(report, encoding="utf-8")

    # ---- console summary ----
    print("=" * 60)
    print("TEACHER AUDIT SUMMARY")
    print("=" * 60)
    print(json.dumps(api_stats, ensure_ascii=False, indent=2))
    print()
    print("Repeatability aggregate:")
    print(json.dumps(repeatability["aggregate"], ensure_ascii=False, indent=2))
    print()
    print(f"Counterfactual axes: {list(counterfactual['axes'].keys())}")
    print(f"Persona groups: {persona['aggregate']['n_groups']}")
    print(f"Flags: {len(flags)}")
    print()
    print(f"RECOMMENDATION: {recommendation}")
    print()
    print(f"Report written to {out / 'teacher_audit_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
