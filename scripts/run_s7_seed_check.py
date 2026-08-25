#!/usr/bin/env python
"""S7 seed-robustness check (final small check, no API, no new experiment line).

Re-trains the W3 config (lambda_mechanism=1.0, lambda_broken=1.0) under three
additional training seeds and compares each seed's C2-vs-C1 deltas on the SAME
final test-only quadruplets and regression sets.

Question answered: does the S7 improvement exceed training-seed variance?
Key metrics: parking G_med, congestion G_broken (+ Gap_shortcut), legacy KL,
seen joint KL — all vs C1, paired bootstrap CIs.

Verdict rule (documented):
  * "stable"  : all seeds share the direction of the seed-42 result on the
                repair metrics, and NO seed shows a significant KL increase
                (regression CI entirely above 0) on legacy or seen joint.
  * otherwise : S7 is described as a preliminary / unstable local repair.

Usage:
    python scripts/run_s7_seed_check.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"

SEEDS = [42, 7, 123, 2024]
W3_CONFIG = "configs/student_s7_w3.yaml"


def _run(cmd: list[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.call([str(PY)] + cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    mechanism_ds = "data/student_s7_mechanism/quadruplets.jsonl"
    base_ds = "data/student_v0_3_s3/aggregated_teacher_dataset.jsonl"
    joint_ds = "data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl"
    joint_cfg = "configs/joint_sampling.yaml"
    manifest = "outputs/student_v0_3_s3_c/split_manifest.json"
    c1 = "outputs/student_s5_joint_m2/checkpoints/best.pt"
    c0 = "outputs/student_v0_3_s3_c/checkpoints/best.pt"
    out_dir = Path("outputs/s7_seed_check")

    if not args.skip_train:
        for seed in SEEDS:
            if seed == 42:
                print(f"[skip] seed 42 = existing outputs/student_s7_w3")
                continue
            outdir = f"outputs/student_s7_w3_seed{seed}"
            if Path(outdir, "checkpoints", "best.pt").exists():
                print(f"[skip] {outdir} exists")
                continue
            print(f"\n===== train W3 seed={seed} =====", flush=True)
            rc = _run([
                "scripts/train_student_s7.py",
                "--base-dataset", base_ds,
                "--joint-dataset", joint_ds,
                "--joint-config", joint_cfg,
                "--mechanism-dataset", mechanism_ds,
                "--init-checkpoint", c1,
                "--split-manifest", manifest,
                "--config", W3_CONFIG,
                "--seed", str(seed),
                "--output", outdir,
            ])
            if rc != 0:
                print(f"[FAIL] seed {seed}")
                return rc

    def _variant_args():
        out = []
        for seed in SEEDS:
            path = ("outputs/student_s7_w3/checkpoints/best.pt" if seed == 42
                    else f"outputs/student_s7_w3_seed{seed}/checkpoints/best.pt")
            out += ["--variant", f"W3_s{seed}={path}"]
        return out

    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n===== causal eval (test-only, per seed) =====", flush=True)
    rc = _run([
        "scripts/eval_s7_causal_repair.py",
        "--mechanism-dataset", mechanism_ds,
        "--c0", c0,
        "--c1", c1,
        *_variant_args(),
        "--output", str(out_dir / "causal_eval_seeds.json"),
    ])
    if rc != 0:
        return rc

    print("\n===== regression eval (per seed) =====", flush=True)
    rc = _run([
        "scripts/eval_s7_regression.py",
        "--base-dataset", base_ds,
        "--joint-k5", joint_ds,
        "--joint-config", joint_cfg,
        "--split-manifest", manifest,
        "--c1", c1,
        *_variant_args(),
        "--output", str(out_dir / "regression_eval_seeds.json"),
    ])
    if rc != 0:
        return rc

    print("\n===== seed summary =====", flush=True)
    causal = json.loads((out_dir / "causal_eval_seeds.json").read_text(encoding="utf-8"))
    regr = json.loads((out_dir / "regression_eval_seeds.json").read_text(encoding="utf-8"))

    # key metrics: (source, metric) pairs
    causal_metrics = [
        ("parking_cost", "G_med", "parking G_med (repair)"),
        ("congestion", "G_broken", "congestion G_broken"),
        ("congestion", "Gap_shortcut", "congestion Gap_shortcut (repair)"),
    ]
    rows = []
    for seed in SEEDS:
        name = f"W3_s{seed}"
        d_c = causal["deltas_vs_c1"][name]
        d_r = regr["deltas_vs_c1"][name]
        row = {"seed": seed}
        for axis, metric, _ in causal_metrics:
            x = d_c[axis][metric]
            row[f"{axis}_{metric}"] = {
                "mean": x["mean"],
                "ci_low": x["ci_low"],
                "ci_high": x["ci_high"],
                "excl_zero": x["ci_low"] > 0 or x["ci_high"] < 0,
            }
        for metric in ("legacy_kl_delta", "seen_joint_kl_delta"):
            x = d_r[metric]
            row[metric] = {"mean": x["mean"], "ci_low": x["ci_low"], "ci_high": x["ci_high"],
                           "excl_zero": x["ci_low"] > 0 or x["ci_high"] < 0}
        rows.append(row)

    def _signs(key, want_negative=True):
        return [(r["seed"], r[key]["mean"], r[key]["excl_zero"]) for r in rows
                if (r[key]["mean"] < 0) == want_negative]

    verdict = {}
    for axis, metric, label in causal_metrics:
        key = f"{axis}_{metric}"
        neg = sum(1 for r in rows if r[key]["mean"] < 0)
        pos = sum(1 for r in rows if r[key]["mean"] > 0)
        verdict[key] = {"label": label, "seeds_negative": neg, "seeds_positive": pos,
                        "values": [round(r[key]["mean"], 4) for r in rows]}
    regressed_seeds = []
    for r in rows:
        if r["legacy_kl_delta"]["mean"] > 0 and r["legacy_kl_delta"]["excl_zero"]:
            regressed_seeds.append((r["seed"], "legacy_kl"))
        if r["seen_joint_kl_delta"]["mean"] > 0 and r["seen_joint_kl_delta"]["excl_zero"]:
            regressed_seeds.append((r["seed"], "seen_joint_kl"))

    repair_metrics = ["parking_cost_G_med", "congestion_Gap_shortcut"]
    stable = (all(verdict[k]["seeds_negative"] == len(SEEDS) for k in repair_metrics)
              and not regressed_seeds)
    summary = {
        "seeds": SEEDS,
        "per_seed": rows,
        "verdict": verdict,
        "regressed_seeds": regressed_seeds,
        "stable_across_seeds": bool(stable),
        "interpretation": (
            "S7 improvement > training-seed variance: the repair direction is consistent "
            "across all seeds and no seed regresses legacy/seen-joint KL."
            if stable else
            "S7 improvement does NOT consistently exceed training-seed variance: describe "
            "as a preliminary / unstable local repair."
        ),
    }
    (out_dir / "seed_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:3000])

    # ---- seed-check section appended to the main S7 report ----
    report_path = Path("outputs/EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md")
    section = ["\n## 10. Seed Robustness Check（最终小检查：S7 改善 vs 训练种子方差）\n",
               f"- W3 配置 × 4 个训练种子（{', '.join(str(s) for s in SEEDS)}），零 API，"
               f"同一 test-only 评估管线（配对 bootstrap 与主报告一致）。\n",
               "| seed | parking G_med | congestion G_broken | congestion Gap_shortcut | Δ legacy KL | Δ seen joint KL |\n"
               "|---|---|---|---|---|---|"]
    for r in rows:
        def _fmt(x):
            star = "*" if x["excl_zero"] else ""
            return f"{x['mean']:+.4f} [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]{star}"
        section.append(f"| {r['seed']} | {_fmt(r['parking_cost_G_med'])} | {_fmt(r['congestion_G_broken'])} | "
                       f"{_fmt(r['congestion_Gap_shortcut'])} | {_fmt(r['legacy_kl_delta'])} | "
                       f"{_fmt(r['seen_joint_kl_delta'])} |")
    section.append("")
    if stable:
        section.append(
            "**结论：stable across seeds** — parking G_med 与 congestion Gap_shortcut 的修复方向在全部 "
            "4 个种子下一致，且没有任何种子在 legacy / seen-joint KL 上出现显著回退（回归 CI 完全在 0 之上）。"
            "S7 的改善超过训练种子方差，可以放心 **Freeze S7-W3** 进入 Singapore。\n")
    else:
        section.append(
            "**结论：preliminary / unstable** — 修复方向未在所有种子下复现（或某种子出现显著 KL 回退）。"
            "S7 应描述为 preliminary / unstable local repair，不作为稳定贡献；Freeze 决策退回 "
            "S5 或按整体性能择优。\n")
    section.append(f"- 明细：`outputs/s7_seed_check/seed_summary.json`；"
                   f"评估：`outputs/s7_seed_check/causal_eval_seeds.json`、`regression_eval_seeds.json`。\n")
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(section) + "\n")
    print(f"appended seed-check section to {report_path}")
    print(f"STABLE={stable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
