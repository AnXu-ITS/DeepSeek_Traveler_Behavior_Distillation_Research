#!/usr/bin/env python
"""Teacher-side dataset analysis for experiment reports.

Computes, from a generated K=3 aggregated dataset (+ repeat records):
  1. overview: samples / personas / trips / axes / selected-mode distribution;
  2. per-axis mean response curves (mean P(mode) per perturbation level) and
     mean DeltaP vs baseline;
  3. behavioral heterogeneity: std of baseline mode probabilities across
     personas (how much personas disagree in the same situation);
  4. teacher sampling noise: per-state pairwise L1 across the K repeats
     (the "how good could a student possibly be" reference).

Usage:
    python scripts/analyze_teacher_dataset.py \
        --dataset data/student_v0_3/aggregated_teacher_dataset.jsonl \
        --repeats data/student_v0_3/repeat_records.jsonl \
        --output outputs/teacher_analysis_v0_3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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
    ap.add_argument("--repeats", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    samples = _load_jsonl(Path(args.dataset))
    by_id = {s["sample_id"]: s for s in samples}
    baselines = [s for s in samples if s["perturbation"]["axis"] == "baseline"]
    cfs = [s for s in samples if s["perturbation"]["axis"] != "baseline"]

    personas = sorted({s["state"]["persona"]["persona_id"] for s in samples})
    trips = sorted({s["state"]["trip"]["trip_id"] for s in samples})
    axes = sorted({s["perturbation"]["axis"] for s in samples if s["perturbation"]["axis"] != "baseline"})

    mode_count = defaultdict(int)
    for s in samples:
        mode_count[s["teacher_aggregate"]["selected_mode"]] += 1

    overview = {
        "n_samples": len(samples),
        "n_baselines": len(baselines),
        "n_counterfactuals": len(cfs),
        "n_personas": len(personas),
        "n_trips": len(trips),
        "axes": axes,
        "selected_mode_distribution": dict(sorted(mode_count.items())),
    }

    # --- per-axis mean response curves ---
    curves = {}
    for axis in axes:
        levels = sorted({s["perturbation"]["level"] for s in cfs if s["perturbation"]["axis"] == axis},
                        key=lambda v: (isinstance(v, bool), v))
        rows = []
        for lvl in levels:
            members = [s for s in cfs if s["perturbation"]["axis"] == axis and s["perturbation"]["level"] == lvl]
            if not members:
                continue
            mode_probs: dict[str, list] = defaultdict(list)
            deltas: dict[str, list] = defaultdict(list)
            for s in members:
                probs = s["teacher_aggregate"]["mode_probabilities"]
                base = by_id.get(s["baseline_sample_id"])
                base_probs = base["teacher_aggregate"]["mode_probabilities"] if base else {}
                for m, p in probs.items():
                    mode_probs[m].append(p)
                for m in set(probs) | set(base_probs):
                    deltas[m].append(probs.get(m, 0.0) - base_probs.get(m, 0.0))
            rows.append({
                "level": lvl,
                "n": len(members),
                "mean_probabilities": {m: round(sum(v) / len(v), 4) for m, v in mode_probs.items()},
                "mean_delta_p_vs_baseline": {m: round(sum(v) / len(v), 4) for m, v in deltas.items()},
            })
        curves[axis] = rows

    # --- heterogeneity: std of baseline probs across personas ---
    het = {}
    if baselines:
        per_mode: dict[str, list] = defaultdict(list)
        for s in baselines:
            for m, p in s["teacher_aggregate"]["mode_probabilities"].items():
                per_mode[m].append(p)
        for m, vals in per_mode.items():
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            het[m] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(vals)}

    # --- teacher sampling noise (per-state pairwise L1 over K repeats) ---
    noise = None
    if args.repeats:
        reps = _load_jsonl(Path(args.repeats))
        by_state: dict[str, list] = defaultdict(list)
        for r in reps:
            if r.get("action") is not None:
                by_state[r["sample_id"]].append(r)
        pairwise = []
        per_state_l1 = {}
        for sid, rs in by_state.items():
            if len(rs) < 2:
                continue
            probs_list = [r["action"]["mode_probabilities"] for r in rs]
            l1s = []
            for i in range(len(probs_list)):
                for j in range(i + 1, len(probs_list)):
                    modes = set(probs_list[i]) | set(probs_list[j])
                    l1 = sum(abs(probs_list[i].get(m, 0.0) - probs_list[j].get(m, 0.0)) for m in modes)
                    l1s.append(l1)
            if l1s:
                per_state_l1[sid] = {"mean_pairwise_l1": round(sum(l1s) / len(l1s), 4),
                                     "max_pairwise_l1": round(max(l1s), 4)}
                pairwise.append(sum(l1s) / len(l1s))
        if pairwise:
            noise = {
                "n_states_with_repeats": len(per_state_l1),
                "mean_pairwise_l1": round(sum(pairwise) / len(pairwise), 4),
                "max_pairwise_l1": round(max(per_state_l1.values(), key=lambda d: d["max_pairwise_l1"])["max_pairwise_l1"], 4),
                "per_state": per_state_l1,
            }

    result = {
        "overview": overview,
        "per_axis_response_curves": curves,
        "baseline_heterogeneity": het,
        "teacher_sampling_noise": noise,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_axis_response_curves"},
                     ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
