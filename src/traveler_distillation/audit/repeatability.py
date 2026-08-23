"""Repeatability analysis: same state -> multiple teacher calls -> stability."""
from __future__ import annotations

from collections import defaultdict

from .metrics import mean, std, value_range, mean_pairwise_l1, selected_mode_agreement
from .records import RepeatabilityRecord


def probability_stability(probs: list[dict]) -> dict:
    """Per-mode mean/std/range over repeated probability distributions."""
    modes = set()
    for p in probs:
        modes |= set(p)
    out = {}
    for m in sorted(modes):
        vals = [p.get(m, 0.0) for p in probs]
        out[m] = {
            "mean": round(mean(vals), 4),
            "std": round(std(vals), 4),
            "range": round(value_range(vals), 4),
            "values": [round(v, 4) for v in vals],
        }
    return out


def analyze_repeatability(records: list[RepeatabilityRecord]) -> dict:
    """Compute repeatability metrics grouped by state_hash."""
    groups: dict[str, list[RepeatabilityRecord]] = defaultdict(list)
    for r in records:
        if r.teacher is not None:
            groups[r.state_hash].append(r)

    per_group = []
    for h in sorted(groups):
        rs = groups[h]
        modes = [r.teacher.selected_mode for r in rs]
        probs = [r.teacher.mode_probabilities for r in rs]
        shifts = [float(r.teacher.departure_time_shift_min) for r in rs]
        confs = [r.teacher.confidence for r in rs]
        per_group.append(
            {
                "state_hash": h,
                "source_sample_id": rs[0].source_sample_id,
                "n_repeats": len(rs),
                "selected_modes": modes,
                "selected_mode_agreement": round(selected_mode_agreement(modes), 4),
                "probability_stability": probability_stability(probs),
                "mean_pairwise_l1": round(mean_pairwise_l1(probs), 4),
                "departure_shift": {
                    "mean": round(mean(shifts), 4),
                    "std": round(std(shifts), 4),
                    "range": round(value_range(shifts), 4),
                    "values": shifts,
                },
                "confidence": {
                    "mean": round(mean(confs), 4),
                    "std": round(std(confs), 4),
                    "range": round(value_range(confs), 4),
                    "values": confs,
                },
            }
        )

    agreements = [g["selected_mode_agreement"] for g in per_group]
    l1s = [g["mean_pairwise_l1"] for g in per_group]
    conf_stds = [g["confidence"]["std"] for g in per_group]
    shift_stds = [g["departure_shift"]["std"] for g in per_group]
    # mean of per-mode std across groups
    avg_prob_std = []
    for g in per_group:
        modes_stds = [v["std"] for v in g["probability_stability"].values()]
        if modes_stds:
            avg_prob_std.append(mean(modes_stds))

    aggregate = {
        "n_states": len(per_group),
        "mean_selected_mode_agreement": round(mean(agreements), 4),
        "mean_pairwise_probability_l1": round(mean(l1s), 4),
        "mean_probability_std": round(mean(avg_prob_std), 4),
        "mean_confidence_std": round(mean(conf_stds), 4),
        "mean_departure_shift_std": round(mean(shift_stds), 4),
    }
    return {"per_group": per_group, "aggregate": aggregate}
