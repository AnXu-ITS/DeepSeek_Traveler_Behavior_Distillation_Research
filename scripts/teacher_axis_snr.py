#!/usr/bin/env python
"""Per-axis teacher signal / noise / SNR on the persona-holdout TEST split.

Mirrors the methodology used in the S1/S2 reports (EXPERIMENT_REPORT_v0_3_S1.md /
S2.md): for each perturbation axis, on the test split, report the teacher's mean
absolute counterfactual move |Delta P_T| (signal), the mean per-state pairwise L1
across the K repeats (sampling noise), and their ratio (SNR).

Usage:
    python scripts/teacher_axis_snr.py \
        --dataset data/student_v0_3_s4/aggregated_teacher_dataset.jsonl \
        --repeats data/student_v0_3_s4/repeat_records.jsonl \
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

from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget  # noqa: E402
from traveler_distillation.student import group_aware_split  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repeats", required=True)
    ap.add_argument("--config", default="configs/student_v0_3_c.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sp_cfg = cfg.get("split", {})
    samples = [AggregatedTeacherTarget.model_validate(r)
               for r in _load_jsonl(Path(args.dataset))]
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
    test_ids = {s.sample_id for s in test}
    by_id = {s.sample_id: s for s in samples}

    # noise: per-state mean pairwise L1 over K repeats (ALL states of the axis,
    # matching the S1/S2 report definition — noise is a dataset property)
    reps = _load_jsonl(Path(args.repeats))
    by_state: dict[str, list] = defaultdict(list)
    for r in reps:
        if r.get("action") is not None:
            by_state[r["sample_id"]].append(r)
    sid_to_axis = {s.sample_id: s.perturbation.axis for s in samples}
    noise_by_axis: dict[str, list] = defaultdict(list)
    for sid, rs in by_state.items():
        if len(rs) < 2:
            continue
        probs = [r["action"]["mode_probabilities"] for r in rs]
        l1s = []
        for i in range(len(probs)):
            for j in range(i + 1, len(probs)):
                modes = set(probs[i]) | set(probs[j])
                l1s.append(sum(abs(probs[i].get(m, 0.0) - probs[j].get(m, 0.0))
                               for m in modes))
        noise_by_axis[sid_to_axis[sid]].append(sum(l1s) / len(l1s))

    # signal: per-axis mean |Delta P_T| vs baseline (test counterfactuals only)
    axes: dict[str, list] = defaultdict(list)
    for s in test:
        if s.perturbation.axis == "baseline":
            continue
        base = by_id.get(s.baseline_sample_id)
        if base is None:
            continue
        t_base = base.teacher_aggregate.mode_probabilities
        t_cf = s.teacher_aggregate.mode_probabilities
        modes = set(t_base) | set(t_cf)
        dt = {m: t_cf.get(m, 0.0) - t_base.get(m, 0.0) for m in modes}
        move = sum(abs(dt[m]) for m in modes) / len(modes)
        axes[s.perturbation.axis].append((s.sample_id, move))

    print(f"test split: {len(test)} samples; test personas = "
          f"{sorted({s.state.persona.persona_id for s in test})}")
    print(f"{'axis':<24}{'pairs':>6}{'|dP_T|':>10}{'noise':>10}{'SNR':>8}")
    out = {}
    for axis in sorted(axes):
        rows = axes[axis]
        n = len(rows)
        signal = sum(m for _, m in rows) / n
        noises = noise_by_axis[axis]
        noise = (sum(noises) / len(noises)) if noises else float("nan")
        snr = signal / noise if noise else float("nan")
        out[axis] = {
            "n_pairs": n,
            "teacher_move_mean": round(signal, 4),
            "noise_pairwise_l1_mean": round(noise, 4),
            "snr": round(snr, 4),
        }
        print(f"{axis:<24}{n:>6}{signal:>10.4f}{noise:>10.4f}{snr:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
