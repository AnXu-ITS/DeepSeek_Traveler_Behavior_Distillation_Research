#!/usr/bin/env python
"""One-shot S5 pipeline: validate -> train M1/M2 -> eval -> report.

    python scripts/run_s5_experiment.py            # full run
    python scripts/run_s5_experiment.py --skip-validate --skip-train

Assumes the joint data has been generated (data/student_s5_joint/).
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
    ap.add_argument("--joint-dir", default="data/student_s5_joint")
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--init-checkpoint", default="outputs/student_v0_3_s3_c/checkpoints/best.pt")
    ap.add_argument("--split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--config", default="configs/student_s5_joint.yaml")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    jd = Path(args.joint_dir)
    dataset_full = jd / "aggregated_teacher_dataset.jsonl"
    dataset_k5 = jd / "aggregated_teacher_dataset_k5.jsonl"
    repeats = jd / "repeat_records.jsonl"

    if not dataset_full.exists():
        print(f"[FAIL] joint dataset not found: {dataset_full}")
        return 1

    # deterministic rebuild of the M1 (base-K) view (backfills any interrupted subset states)
    rc = _run([
        "scripts/rebuild_k5_view.py",
        "--full", str(dataset_full),
        "--repeats", str(repeats),
        "--joint-config", args.joint_config,
        "--output", str(dataset_k5),
    ])
    if rc != 0:
        print("[FAIL] rebuild_k5_view failed")
        return rc

    if not args.skip_validate:
        for name, ds in (("full", dataset_full), ("k5", dataset_k5)):
            rc = _run([
                "scripts/validate_joint_dataset.py",
                "--dataset", str(ds),
                "--repeats", str(repeats),
                "--base-dataset", args.base_dataset,
                "--joint-config", args.joint_config,
            ])
            if rc != 0:
                print(f"[FAIL] validation failed for {name} view")
                return rc

    if not args.skip_train:
        for model_name, ds in (("m1", dataset_k5), ("m2", dataset_full)):
            print(f"\n===== train S5-{model_name.upper()} =====", flush=True)
            rc = _run([
                "scripts/train_student_s5_joint.py",
                "--base-dataset", args.base_dataset,
                "--joint-dataset", str(ds),
                "--joint-config", args.joint_config,
                "--init-checkpoint", args.init_checkpoint,
                "--split-manifest", args.split_manifest,
                "--config", args.config,
                "--output", f"outputs/student_s5_joint_{model_name}",
            ])
            if rc != 0:
                print(f"[FAIL] training failed for {model_name.upper()}")
                return rc

    print("\n===== eval =====", flush=True)
    rc = _run([
        "scripts/eval_s5_joint.py",
        "--base-dataset", args.base_dataset,
        "--joint-k5", str(dataset_k5),
        "--joint-full", str(dataset_full),
        "--joint-config", args.joint_config,
        "--split-manifest", args.split_manifest,
        "--m0", args.init_checkpoint,
        "--m1", "outputs/student_s5_joint_m1/checkpoints/best.pt",
        "--m2", "outputs/student_s5_joint_m2/checkpoints/best.pt",
        "--output", "outputs/s5_joint_eval",
    ])
    if rc != 0:
        print("[FAIL] eval failed")
        return rc

    print("\n===== report =====", flush=True)
    rc = _run([
        "scripts/make_s5_report.py",
        "--eval", "outputs/s5_joint_eval/eval_metrics.json",
        "--manifest", str(jd / "generation_manifest.json"),
        "--output", "outputs/EXPERIMENT_REPORT_S5_MULTI_AXIS.md",
    ])
    if rc != 0:
        print("[FAIL] report failed")
        return rc

    print("\nS5 pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
