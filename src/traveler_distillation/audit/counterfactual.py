"""Counterfactual response analysis: recover response curves from pilot samples."""
from __future__ import annotations

from collections import defaultdict

from ..schemas.dataset import TeacherDatasetSample
from .metrics import mean, spearman_rank, direction_reversals

_CONTINUOUS_AXES = {
    "weather_intensity",
    "road_congestion",
    "transit_delay",
    "fare_multiplier",
    "parking_cost_multiplier",
}

_AXIS_LEVEL = {
    "weather_intensity": lambda c: c.weather.intensity,
    "road_congestion": lambda c: c.road_congestion,
    "transit_delay": lambda c: float(c.transit_delay_min),
    "fare_multiplier": lambda c: c.fare_multiplier,
    "parking_cost_multiplier": lambda c: c.parking_cost_multiplier,
    "road_disruption": lambda c: float(c.road_disruption),
}


def _axis_level(context, axis: str) -> float:
    if axis not in _AXIS_LEVEL:
        raise ValueError(f"no level mapping for axis '{axis}'")
    return _AXIS_LEVEL[axis](context)


def recover_curves(samples: list[TeacherDatasetSample]) -> list[dict]:
    """Recover (baseline + counterfactual) response curves grouped by group id."""
    by_id = {s.sample_id: s for s in samples}
    groups: dict[str, list[TeacherDatasetSample]] = defaultdict(list)
    for s in samples:
        if s.counterfactual_group_id and s.teacher is not None:
            groups[s.counterfactual_group_id].append(s)

    curves = []
    for gid in sorted(groups):
        members = groups[gid]
        baseline_id = members[0].baseline_sample_id
        baseline = by_id.get(baseline_id) if baseline_id else None
        if baseline is None or baseline.teacher is None:
            continue
        axis = members[0].perturbation.axis
        base_level = _axis_level(baseline.state.context, axis)
        points: dict[float, dict] = {base_level: baseline.teacher.mode_probabilities}
        for m in members:
            points.setdefault(float(m.perturbation.level), m.teacher.mode_probabilities)
        points = sorted(points.items(), key=lambda kv: kv[0])
        curves.append(
            {
                "group_id": gid,
                "axis": axis,
                "persona_id": members[0].state.persona.persona_id,
                "trip_id": members[0].state.trip.trip_id,
                "points": [{"level": lvl, "probs": probs} for lvl, probs in points],
            }
        )
    return curves


def analyze_counterfactual(samples: list[TeacherDatasetSample]) -> dict:
    curves = recover_curves(samples)
    by_axis: dict[str, list[dict]] = defaultdict(list)
    for c in curves:
        by_axis[c["axis"]].append(c)

    axes = {}
    for axis in sorted(by_axis):
        cs = by_axis[axis]
        mode_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for c in cs:
            for pt in c["points"]:
                for mode, p in pt["probs"].items():
                    mode_points[mode].append((pt["level"], p))

        mode_stats = {}
        for mode in sorted(mode_points):
            pts = sorted(mode_points[mode])
            levels = [l for l, _ in pts]
            vals = [p for _, p in pts]
            rho = spearman_rank(levels, vals) if len(pts) >= 2 else 0.0
            revs = []
            for c in cs:
                seq = [pt["probs"].get(mode, 0.0) for pt in c["points"]]
                revs.append(float(direction_reversals(seq)))
            mode_stats[mode] = {
                "spearman_rho": round(rho, 4),
                "n_points": len(pts),
                "mean_direction_reversals": round(mean(revs), 4),
            }

        axes[axis] = {"n_groups": len(cs), "mode_stats": mode_stats}

    return {"axes": axes, "curves": curves}


def elasticity_preview(samples: list[TeacherDatasetSample]) -> dict:
    """Preview teacher elasticity: (P(C_i)-P(C_0)) / (C_i - C_0) for continuous axes."""
    curves = recover_curves(samples)
    result: dict[str, list[dict]] = {}
    for c in curves:
        axis = c["axis"]
        if axis not in _CONTINUOUS_AXES:
            continue
        points = c["points"]
        base_level, base_probs = points[0]["level"], points[0]["probs"]
        entries = []
        for pt in points[1:]:
            dl = pt["level"] - base_level
            if abs(dl) < 1e-9:
                continue
            modes = set(base_probs) | set(pt["probs"])
            deltas = {}
            elastics = {}
            for m in modes:
                dp = pt["probs"].get(m, 0.0) - base_probs.get(m, 0.0)
                deltas[m] = round(dp, 4)
                elastics[m] = round(dp / dl, 4)
            entries.append(
                {"level": pt["level"], "delta_p": deltas, "elasticity": elastics}
            )
        result.setdefault(axis, []).append(
            {
                "group_id": c["group_id"],
                "persona_id": c["persona_id"],
                "trip_id": c["trip_id"],
                "baseline_level": base_level,
                "entries": entries,
            }
        )
    return result
