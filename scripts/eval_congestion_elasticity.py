#!/usr/bin/env python
"""Evaluate a Student checkpoint's counterfactual elasticity, per axis.

Designed for the S1 decisive experiment: measure how well a checkpoint's
counterfactual response matches the teacher, axis by axis (road_congestion vs
the already-trained weather/fare axes). Loads any saved checkpoint (old v0.3
"before" or the post-S1 "after" checkpoint) and evaluates it on the same
persona-holdout TEST split.

Usage:
    # "before": v0.3-C checkpoint (no congestion training) on the extended data
    python scripts/eval_congestion_elasticity.py \
        --dataset data/student_v0_3_s1/aggregated_teacher_dataset.jsonl \
        --checkpoint outputs/student_v0_3_c/checkpoints/best.pt \
        --config configs/student_v0_3_c.yaml

    # "after": post-S1 checkpoint (same dataset)
    python scripts/eval_congestion_elasticity.py \
        --dataset data/student_v0_3_s1/aggregated_teacher_dataset.jsonl \
        --checkpoint outputs/student_v0_3_s1_c/checkpoints/best.pt \
        --config configs/student_v0_3_c.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget  # noqa: E402
from traveler_distillation.student import (  # noqa: E402
    FeatureExtractor,
    TravelerStudent,
    group_aware_split,
)
from traveler_distillation.student.eval import predict_probs  # noqa: E402


def _load_dataset(path: Path) -> list[AggregatedTeacherTarget]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(AggregatedTeacherTarget.model_validate(json.loads(line)))
    return samples


def _load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    extractor = FeatureExtractor.from_state_dict(ckpt["extractor_state"])
    model = TravelerStudent(ckpt["config"], ckpt["feature_spec"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, extractor


@torch.no_grad()
def per_axis_elasticity(samples, model, extractor, device) -> dict:
    """Counterfactual elasticity metrics grouped by perturbation axis."""
    by_id = {s.sample_id: s for s in samples}
    axes: dict[str, list] = defaultdict(list)

    for s in samples:
        if s.perturbation.axis == "baseline":
            continue
        base = by_id.get(s.baseline_sample_id)
        if base is None:
            continue
        t_base = base.teacher_aggregate.mode_probabilities
        t_cf = s.teacher_aggregate.mode_probabilities
        s_base = predict_probs(base, model, extractor, device)
        s_cf = predict_probs(s, model, extractor, device)
        modes = set(t_base) | set(t_cf) | set(s_base) | set(s_cf)

        dt = {m: t_cf.get(m, 0.0) - t_base.get(m, 0.0) for m in modes}
        ds = {m: s_cf.get(m, 0.0) - s_base.get(m, 0.0) for m in modes}
        # mean absolute teacher move (elasticity magnitude), student move, and
        # per-mode teacher-vs-student gap, all averaged over the mode set
        nmodes = len(modes)
        teacher_move = sum(abs(dt[m]) for m in modes) / nmodes
        student_move = sum(abs(ds[m]) for m in modes) / nmodes
        delta_gap = sum(abs(dt[m] - ds[m]) for m in modes) / nmodes

        agree = 0
        for m in modes:
            if abs(dt[m]) < 1e-3 or (dt[m] > 0) == (ds[m] > 0):
                agree += 1
        sign_agree = agree / nmodes

        axes[s.perturbation.axis].append(
            {
                "teacher_move": teacher_move,
                "student_move": student_move,
                "delta_gap": delta_gap,
                "sign_agree": sign_agree,
                "level": s.perturbation.level,
            }
        )

    out = {}
    for axis, rows in sorted(axes.items()):
        n = len(rows)
        out[axis] = {
            "n_pairs": n,
            "teacher_move_mean": round(sum(r["teacher_move"] for r in rows) / n, 4),
            "student_move_mean": round(sum(r["student_move"] for r in rows) / n, 4),
            "mean_abs_delta_p_diff": round(sum(r["delta_gap"] for r in rows) / n, 4),
            "mean_sign_agreement": round(sum(r["sign_agree"] for r in rows) / n, 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sp_cfg = cfg.get("split", {})
    samples = _load_dataset(Path(args.dataset))

    # apply the same persona-holdout split used in training
    for s in samples:
        if s.persona_group_id is None:
            s.persona_group_id = s.state.persona.persona_id
    _, _, test = group_aware_split(
        samples,
        group_field=sp_cfg.get("group_field", "persona_group_id"),
        train_ratio=sp_cfg.get("train_ratio", 0.7),
        val_ratio=sp_cfg.get("val_ratio", 0.15),
        test_ratio=sp_cfg.get("test_ratio", 0.15),
        seed=cfg.get("training", {}).get("seed", 42),
    )
    test_personas = sorted({s.state.persona.persona_id for s in test})
    print(f"dataset={len(samples)} samples, test={len(test)} samples, "
          f"test_personas={test_personas}")

    model, extractor = _load_checkpoint(Path(args.checkpoint))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    per_axis = per_axis_elasticity(test, model, extractor, device)

    print("\nper-axis counterfactual elasticity (test split):")
    print(f"{'axis':<22}{'pairs':>6}{'|dP_T|':>10}{'|dP_S|':>10}{'|dP_T-dP_S|':>12}{'sign':>8}")
    for axis, m in per_axis.items():
        print(f"{axis:<22}{m['n_pairs']:>6}{m['teacher_move_mean']:>10.4f}"
              f"{m['student_move_mean']:>10.4f}{m['mean_abs_delta_p_diff']:>12.4f}"
              f"{m['mean_sign_agreement']:>8.4f}")

    if args.json_out:
        payload = {
            "checkpoint": str(args.checkpoint),
            "dataset": str(args.dataset),
            "test_personas": test_personas,
            "per_axis": per_axis,
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
