#!/usr/bin/env python
"""Run a post-augmentation v0.3 experiment on an EXTENDED dataset (train A/B/C).

Generic across augmentation stages: ``--tag`` controls the output-directory
suffix and labels, so the pre-augmentation checkpoints are always preserved for
before/after comparison.

    # S1 (weather + fare + congestion)
    python scripts/run_v0_3_s1_experiment.py --tag s1 ...
    # S2 (adds transit_delay + parking_cost)
    python scripts/run_v0_3_s1_experiment.py --tag s2 ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"


def _run(cmd: list[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call([str(PY)] + cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s1",
                    help="stage tag: output dirs become student_v0_3_{tag}_* and "
                         "the default dataset becomes data/student_v0_3_{tag}/")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--repeats", default=None)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    tag = args.tag
    dataset = Path(args.dataset or f"data/student_v0_3_{tag}/aggregated_teacher_dataset.jsonl")
    repeats = Path(args.repeats or f"data/student_v0_3_{tag}/repeat_records.jsonl")

    runs = [
        ("A", "scripts/train_student_v0_2_a.py", "configs/student_v0_3_a.yaml", f"student_v0_3_{tag}_a"),
        ("B", "scripts/train_student_v0_2_b.py", "configs/student_v0_3_b.yaml", f"student_v0_3_{tag}_b"),
        ("C", "scripts/train_student_v0_2_c.py", "configs/student_v0_3_c.yaml", f"student_v0_3_{tag}_c"),
    ]

    if not dataset.exists():
        print(f"[FAIL] dataset not found: {dataset}")
        return 1

    if not args.skip_validate:
        rc = _run([
            "scripts/validate_aggregated_dataset.py",
            "--dataset", str(dataset),
            "--repeats", str(repeats),
        ])
        if rc != 0:
            print("[FAIL] dataset validation failed")
            return rc

    out_dir = Path(args.outputs)
    if not args.skip_train:
        for label, script, config, outname in runs:
            print(f"\n===== train v0.3-{tag}-{label} =====", flush=True)
            rc = _run([script, "--config", config, "--dataset", str(dataset),
                       "--output", str(out_dir / outname)])
            if rc != 0:
                print(f"[FAIL] v0.3-{tag}-{label} training failed")
                return rc

    for label in ("B", "C"):
        print(f"\n===== compare {tag}-A vs {tag}-{label} =====", flush=True)
        rc = _run([
            "scripts/compare_students.py",
            "--baseline", str(out_dir / f"student_v0_3_{tag}_a"),
            "--candidate", str(out_dir / f"student_v0_3_{tag}_{label.lower()}"),
            "--baseline-label", f"v0.3-{tag}-A",
            "--candidate-label", f"v0.3-{tag}-{label}",
            "--output", str(out_dir / f"comparison_v0_3_{tag}_A_vs_{label}.md"),
        ])
        if rc != 0:
            print(f"[FAIL] comparison {tag}-A vs {tag}-{label} failed")
            return rc

    rows = {}
    for label, _, _, outname in runs:
        metrics_path = out_dir / outname / "evaluation_metrics.json"
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows[label] = {
            "mode_accuracy": m["test"].get("mode_accuracy"),
            "kl": m["test"].get("kl"),
            "probability_l1": m["test"].get("probability_l1"),
            "delta_p_l1": m.get("counterfactual_preview", {}).get("mean_abs_delta_p_diff"),
            "sign_agreement": m.get("counterfactual_sign_agreement", {}).get("mean_sign_agreement"),
            "heterogeneity": m.get("heterogeneity_preview", {}).get("mean_abs_delta_p_diff"),
        }

    lines = [f"# v0.3-{tag} Summary\n", "| metric | A | B | C |", "|---|---|---|---|"]
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

    summary_path = out_dir / f"v0_3_{tag}_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {summary_path}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
