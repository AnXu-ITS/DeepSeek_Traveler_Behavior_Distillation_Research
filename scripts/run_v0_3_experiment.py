#!/usr/bin/env python
"""Run the full v0.3 experiment: validate data, train A/B/C, compare, summarize.

Usage:
    python scripts/run_v0_3_experiment.py \
        --dataset data/student_v0_3/aggregated_teacher_dataset.jsonl \
        --repeats data/student_v0_3/repeat_records.jsonl \
        --outputs outputs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

RUNS = [
    ("A", "scripts/train_student_v0_2_a.py", "configs/student_v0_3_a.yaml", "student_v0_3_a"),
    ("B", "scripts/train_student_v0_2_b.py", "configs/student_v0_3_b.yaml", "student_v0_3_b"),
    ("C", "scripts/train_student_v0_2_c.py", "configs/student_v0_3_c.yaml", "student_v0_3_c"),
]


def _run(cmd: list[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call([str(PY)] + cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/student_v0_3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--repeats", default="data/student_v0_3/repeat_records.jsonl")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not dataset.exists():
        print(f"[FAIL] dataset not found: {dataset}")
        return 1

    if not args.skip_validate:
        rc = _run([
            "scripts/validate_aggregated_dataset.py",
            "--dataset", args.dataset,
            "--repeats", args.repeats,
        ])
        if rc != 0:
            print("[FAIL] dataset validation failed")
            return rc

    out_dir = Path(args.outputs)
    if not args.skip_train:
        for label, script, config, outname in RUNS:
            print(f"\n===== train v0.3-{label} =====", flush=True)
            rc = _run([script, "--config", config, "--dataset", args.dataset,
                       "--output", str(out_dir / outname)])
            if rc != 0:
                print(f"[FAIL] v0.3-{label} training failed")
                return rc

    # comparisons
    for label in ("B", "C"):
        print(f"\n===== compare A vs {label} =====", flush=True)
        rc = _run([
            "scripts/compare_students.py",
            "--baseline", str(out_dir / "student_v0_3_a"),
            "--candidate", str(out_dir / f"student_v0_3_{label.lower()}"),
            "--baseline-label", "v0.3-A",
            "--candidate-label", f"v0.3-{label}",
            "--output", str(out_dir / f"comparison_v0_3_A_vs_{label}.md"),
        ])
        if rc != 0:
            print(f"[FAIL] comparison A vs {label} failed")
            return rc

    # summary table
    rows = {}
    for label, _, _, outname in RUNS:
        metrics_path = out_dir / outname / "evaluation_metrics.json"
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows[label] = {
            "mode_accuracy": m["test"].get("mode_accuracy"),
            "kl": m["test"].get("kl"),
            "probability_l1": m["test"].get("probability_l1"),
            "delta_p_l1": m.get("counterfactual_preview", {}).get("mean_abs_delta_p_diff"),
            "sign_agreement": m.get("counterfactual_sign_agreement", {}).get("mean_sign_agreement"),
            "heterogeneity": m.get("heterogeneity_preview", {}).get("mean_abs_delta_p_diff"),
            "persona_breakdown": m.get("persona_breakdown"),
        }

    lines = ["# v0.3 Summary\n", "| metric | A | B | C |", "|---|---|---|---|"]
    for metric, name in [
        ("mode_accuracy", "mode_accuracy"),
        ("kl", "KL"),
        ("probability_l1", "probability L1"),
        ("delta_p_l1", "counterfactual |dP_T - dP_S|"),
        ("sign_agreement", "counterfactual sign agreement"),
        ("heterogeneity", "heterogeneity |dP_T - dP_S|"),
    ]:
        vals = [rows[l].get(metric) for l in ("A", "B", "C")]
        cell = lambda v: "n/a" if v is None else f"{v:.4f}"
        lines.append(f"| {name} | {cell(vals[0])} | {cell(vals[1])} | {cell(vals[2])} |")

    summary_path = out_dir / "v0_3_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {summary_path}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
