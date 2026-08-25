#!/usr/bin/env python
"""One-shot S7 pipeline: build mechanism dataset -> train 5 variants ->
causal eval (test-only, bootstrap) -> regression eval -> VAL-based selection ->
final report.

Selection protocol (val ONLY — the final causal test set is never consulted):
  1. regression gate: val legacy KL <= C1 val legacy KL * 1.10 AND
     val seen-joint KL <= C1 val seen-joint KL * 1.10  (S7 §22-23 bounds);
  2. among gated variants pick the smallest val_mean_mechanism_gap
     (primary metric on val);
  3. if no variant passes the gate, the smallest-gap variant is selected and
     the regression violation is flagged in the selection record.

Usage:
    python scripts/run_s7_experiment.py                 # full pipeline
    python scripts/run_s7_experiment.py --skip-train    # eval+selection+report only
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

VARIANTS = [
    ("W1", "configs/student_s7_w1.yaml", "outputs/student_s7_w1"),
    ("W2", "configs/student_s7_w2.yaml", "outputs/student_s7_w2"),
    ("W3", "configs/student_s7_w3.yaml", "outputs/student_s7_w3"),
    ("mech_only", "configs/student_s7_mech_only.yaml", "outputs/student_s7_mech_only"),
    ("broken_only", "configs/student_s7_broken_only.yaml", "outputs/student_s7_broken_only"),
]


def _run(cmd: list[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call([str(PY)] + cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    mechanism_ds = "data/student_s7_mechanism/quadruplets.jsonl"
    base_ds = "data/student_v0_3_s3/aggregated_teacher_dataset.jsonl"
    joint_ds = "data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl"
    joint_cfg = "configs/joint_sampling.yaml"
    manifest = "outputs/student_v0_3_s3_c/split_manifest.json"
    c1 = "outputs/student_s5_joint_m2/checkpoints/best.pt"
    c0 = "outputs/student_v0_3_s3_c/checkpoints/best.pt"

    print("\n===== step 1: build mechanism quadruplet dataset =====", flush=True)
    rc = _run(["scripts/build_s7_mechanism_dataset.py"])
    if rc != 0:
        return rc

    if not args.skip_train:
        print("\n===== step 2: train 5 S7 variants =====", flush=True)
        for name, cfg, outdir in VARIANTS:
            if Path(outdir, "checkpoints", "best.pt").exists() and not args.force:
                print(f"  [skip] {name} checkpoint exists")
                continue
            print(f"\n----- train {name} -----", flush=True)
            rc = _run([
                "scripts/train_student_s7.py",
                "--base-dataset", base_ds,
                "--joint-dataset", joint_ds,
                "--joint-config", joint_cfg,
                "--mechanism-dataset", mechanism_ds,
                "--init-checkpoint", c1,
                "--split-manifest", manifest,
                "--config", cfg,
                "--output", outdir,
            ])
            if rc != 0:
                print(f"[FAIL] training {name}")
                return rc

    print("\n===== step 3: causal repair eval (test-only, paired bootstrap) =====", flush=True)
    variant_args = []
    for name, _, outdir in VARIANTS:
        variant_args += ["--variant", f"{name}={outdir}/checkpoints/best.pt"]
    rc = _run([
        "scripts/eval_s7_causal_repair.py",
        "--mechanism-dataset", mechanism_ds,
        "--c0", c0,
        "--c1", c1,
        *variant_args,
        "--output", "outputs/s7_causal_eval/eval_metrics.json",
    ])
    if rc != 0:
        return rc

    print("\n===== step 4: legacy + joint regression eval =====", flush=True)
    rc = _run([
        "scripts/eval_s7_regression.py",
        "--base-dataset", base_ds,
        "--joint-k5", joint_ds,
        "--joint-config", joint_cfg,
        "--split-manifest", manifest,
        "--c1", c1,
        *variant_args,
        "--output", "outputs/s7_regression/eval_metrics.json",
    ])
    if rc != 0:
        return rc

    print("\n===== step 5: val-based selection =====", flush=True)
    out_sel = Path("outputs/s7_selection")
    out_sel.mkdir(parents=True, exist_ok=True)
    c1_val_path = out_sel / "c1_val_metrics.json"
    rc = _run([
        "scripts/eval_s7_val_metrics.py",
        "--checkpoint", c1,
        "--base-dataset", base_ds,
        "--joint-dataset", joint_ds,
        "--joint-config", joint_cfg,
        "--mechanism-dataset", mechanism_ds,
        "--split-manifest", manifest,
        "--output", str(c1_val_path),
    ])
    if rc != 0:
        return rc
    c1_val = json.loads(c1_val_path.read_text(encoding="utf-8"))

    candidates = []
    for name, _, outdir in VARIANTS:
        vm = json.loads(Path(outdir, "val_metrics.json").read_text(encoding="utf-8"))
        leg_ok = vm["val_legacy"]["kl"] <= c1_val["val_legacy"]["kl"] * 1.10
        jnt_ok = vm["val_seen_joint"]["kl"] <= c1_val["val_seen_joint"]["kl"] * 1.10
        candidates.append({
            "name": name,
            "val_mean_mechanism_gap": vm["val_mean_mechanism_gap"],
            "val_legacy_kl": vm["val_legacy"]["kl"],
            "val_seen_joint_kl": vm["val_seen_joint"]["kl"],
            "regression_gate_ok": bool(leg_ok and jnt_ok),
            "lambda_mechanism": vm["lambda_mechanism"],
            "lambda_broken": vm["lambda_broken"],
            "best_epoch": vm["best_epoch"],
        })
    gated = [c for c in candidates if c["regression_gate_ok"]]
    pool = gated or candidates
    selected = min(pool, key=lambda c: c["val_mean_mechanism_gap"])
    selection = {
        "protocol": "val-only: regression gate (val legacy KL + val seen-joint KL <= C1 x 1.10) "
                    "then min val_mean_mechanism_gap; causal test set never consulted",
        "c1_val": c1_val,
        "candidates": candidates,
        "selected": selected["name"],
        "selected_gate_ok": selected["regression_gate_ok"],
        "note": "if no variant passed the gate, the min-gap variant was selected and the "
                "violation is flagged above",
    }
    (out_sel / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected["name"], "gate_ok": selected["regression_gate_ok"],
                      "val_mean_gap": selected["val_mean_mechanism_gap"]}, indent=2))

    print("\n===== step 5b: variant checkpoint distances (ablation diagnostics) =====", flush=True)
    rc = _run([
        "scripts/s7_variant_distances.py",
        "--c1", c1,
        "--output", str(out_sel / "variant_distances.json"),
    ])
    if rc != 0:
        return rc

    print("\n===== step 6: final report =====", flush=True)
    rc = _run([
        "scripts/make_s7_report.py",
        "--causal-eval", "outputs/s7_causal_eval/eval_metrics.json",
        "--regression-eval", "outputs/s7_regression/eval_metrics.json",
        "--selection", str(out_sel / "selection.json"),
        "--variant-distances", str(out_sel / "variant_distances.json"),
        "--output", "outputs/EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md",
    ])
    if rc != 0:
        return rc

    print("\nS7 pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
