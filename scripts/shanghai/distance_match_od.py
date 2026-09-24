#!/usr/bin/env python
"""Distance-matched OD assignment for a fair cross-city comparison.

The production default picks OD by a deterministic HASH of persona_id — it
ignores the survey ``distance_km`` entirely, so Singapore-calibrated trips
(8-22 km) get collapsed onto a small bbox (see analyze_pt_share.py). This
script reassigns each trip's (origin_node, dest_node) so the straight-line OD
distance matches the survey ``distance_km`` as closely as the bbox allows, and
writes a new CSV with explicit ``origin_node`` / ``dest_node`` columns that the
reference pipeline consumes verbatim.

Trips longer than the bbox diagonal are clamped to the farthest achievable
pair (reported in the summary). Deterministic: seeded by trip_id.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WB))
SRC = _WB / "examples/sample_population.csv"
NODES = _WB / "data/shanghai/transit/activity_nodes.json"
NETWORK = _WB / "data/shanghai/transit/network_with_transit.xml"
DST = _WB / "examples/sample_population_shanghai_dmatched.csv"

HOME_SAMPLES = 150  # candidate homes sampled per trip


def _seed(trip_id: str) -> int:
    return int(hashlib.sha256(("dmatched:" + trip_id).encode()).hexdigest()[:16], 16)


def load_nodes() -> list[dict]:
    """Valid activity nodes ONLY — identical filter to the pipeline
    (car-accessible AND walk largest connected component)."""
    from reference_pipeline.matsim_adapter import build_supply_view
    _g, _xy, _tt, activity_nodes = build_supply_view(NETWORK, NODES)
    return activity_nodes


def main() -> int:
    try:
        import numpy as np
    except ImportError:
        print("numpy required", file=sys.stderr)
        return 2

    nodes = load_nodes()
    ids = [str(n["node"]) for n in nodes]
    xy = np.array([[float(n["x"]), float(n["y"])] for n in nodes], dtype=np.float64)
    n = len(xy)
    # farthest achievable pair (bbox diagonal bound): a farthest pair always has
    # one endpoint among the 4 extreme-axis nodes
    extreme_idx = [int(np.argmin(xy[:, 0])), int(np.argmin(xy[:, 1])),
                   int(np.argmax(xy[:, 0])), int(np.argmax(xy[:, 1]))]
    max_d = max(float(np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1]).max()) for i in extreme_idx)
    print(f"activity nodes={n}, max achievable straight-line distance={max_d/1000:.2f} km")

    rng_map: dict[str, object] = {}

    def assign(trip_id: str, target_km: float) -> tuple[str, str, float]:
        import random
        if trip_id not in rng_map:
            rng_map[trip_id] = random.Random(_seed(trip_id))
        rng = rng_map[trip_id]
        target_m = float(target_km) * 1000.0
        best = None
        cand = rng.sample(range(n), HOME_SAMPLES)
        for h in cand:
            d = np.hypot(xy[:, 0] - xy[h, 0], xy[:, 1] - xy[h, 1])
            j = int(np.argmin(np.abs(d - target_m)))
            err = abs(float(d[j]) - target_m)
            if best is None or err < best[0]:
                best = (err, h, j)
        _, h, j = best
        achieved = float(np.hypot(xy[j, 0] - xy[h, 0], xy[j, 1] - xy[h, 1])) / 1000.0
        return ids[h], ids[j], achieved

    with open(SRC, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["origin_node", "dest_node"]
        rows = []
        print(f"\n{'trip':9} {'target_km':>9} {'achieved_km':>11}")
        for row in reader:
            tid = row["trip_id"]
            target = float(row["distance_km"])
            o, d, achieved = assign(tid, target)
            row["origin_node"] = o
            row["dest_node"] = d
            rows.append(row)
            flag = "  <-- clamped" if achieved < target * 0.9 else ""
            print(f"{tid:9} {target:9.2f} {achieved:11.2f}{flag}")

    with open(DST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
