#!/usr/bin/env python
"""Compare two student runs (e.g. v0.2-A vs v0.2-B) side by side.

Produces a markdown table of test metrics + elasticity preview, so the effect of
L_elasticity (or any other loss change) on behavioral-response preservation is
visible at a glance.

Usage:
    python scripts/compare_students.py \
        --baseline outputs/student_v0_2_a \
        --candidate outputs/student_v0_2_b \
        --output outputs/comparison_v0_2_a_vs_b.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_ORDER = [
    "mode_accuracy",
    "cross_entropy",
    "kl",
    "probability_l1",
    "departure_mae",
    "departure_rmse",
    "departure_huber",
    "departure_sign_agreement",
]


def _load_metrics(run_dir: Path) -> dict:
    d = json.loads((run_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline-label", default="baseline")
    ap.add_argument("--candidate-label", default="candidate")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base_dir = Path(args.baseline)
    cand_dir = Path(args.candidate)
    b = _load_metrics(base_dir)
    c = _load_metrics(cand_dir)

    lines = []
    a = lines.append
    a("# Student Comparison\n")
    a(f"- baseline: `{base_dir}` ({args.baseline_label})")
    a(f"- candidate: `{cand_dir}` ({args.candidate_label})\n")

    a("## Test Metrics\n")
    a("| metric | baseline | candidate | Δ |")
    a("|---|---|---|---|")
    for m in METRIC_ORDER:
        bv = b["test"].get(m, float("nan"))
        cv = c["test"].get(m, float("nan"))
        delta = (cv - bv) if isinstance(bv, (int, float)) and isinstance(cv, (int, float)) else ""
        a(f"| {m} | {bv:.6g} | {cv:.6g} | {delta if delta == '' else f'{delta:+.6g}'} |")

    bcf = b.get("counterfactual_preview", {}).get("mean_abs_delta_p_diff")
    ccf = c.get("counterfactual_preview", {}).get("mean_abs_delta_p_diff")
    a("\n## Behavioral Elasticity (counterfactual |ΔP_T - ΔP_S|)\n")
    if bcf is not None and ccf is not None:
        a(f"| baseline | candidate | Δ |")
        a("|---|---|---|")
        a(f"| {bcf:.4f} | {ccf:.4f} | {ccf - bcf:+.4f} |")
        a(f"\nlower is better; negative Δ = elasticity improved by the candidate loss.\n")

    a("## Run Info\n")
    a("| field | baseline | candidate |")
    a("|---|---|---|")
    for field in ("trainable_parameters", "best_epoch", "runtime_seconds", "device", "seed"):
        a(f"| {field} | {b.get(field)} | {c.get(field)} |")

    out = Path(args.output)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
