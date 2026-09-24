#!/usr/bin/env python
"""Diagnose why Shanghai PT share is low: OD distances + metro access per trip.

Joins the S9 decision manifest with the assigned activity nodes and the snapped
metro stops, and reports, per trip: straight-line OD distance, nearest-metro-stop
distance for home and destination, the persona's habitual_mode, the CSV
survey distance, and the chosen mode.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]
OUT = _WB / "outputs/reference_pipeline/shanghai_xuhui"
NODES = _WB / "data/shanghai/transit/activity_nodes.json"
STOPS = _WB / "data/shanghai/transit/prep_stops.jsonl"


def load_nodes() -> dict[str, tuple[float, float]]:
    raw = NODES.read_text(encoding="utf-8")
    # activity_nodes.json is a JSON list of {"node","x","y"} (possibly wrapped)
    data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("nodes", data.get("activity_nodes", []))
    return {str(d["node"]): (float(d["x"]), float(d["y"])) for d in data}


def load_stops() -> list[tuple[float, float]]:
    pts = []
    for line in STOPS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        pts.append((float(d["x"]), float(d["y"])))
    return pts


def dist_km(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) / 1000.0


def main() -> int:
    nodes = load_nodes()
    stops = load_stops()
    # persona attributes by trip_id
    persona: dict[str, dict] = {}
    with open(_WB / "examples/sample_population.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            persona[row["trip_id"]] = row

    rows = []
    with open(OUT / "decision_manifest.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            h, d = nodes.get(r["home_node"]), nodes.get(r["dest_node"])
            if h is None or d is None:
                rows.append((r, None, None, None))
                continue
            od = dist_km(h, d)
            acc_h = min(dist_km(h, s) for s in stops)
            acc_d = min(dist_km(d, s) for s in stops)
            rows.append((r, od, acc_h, acc_d))

    print(f"{'trip':9} {'mode':5} {'habit':6} {'csv_km':>6} {'od_km':>6} {'acc_home':>8} {'acc_dest':>8}")
    pt, car, bike, walk = 0, 0, 0, 0
    od_list, acc_list = [], []
    for r, od, acc_h, acc_d in rows:
        m = r["student_mode"]
        p = persona.get(r["trip_id"], {})
        hab = p.get("habitual_mode", "?")
        csv_km = p.get("distance_km", "?")
        if od is None:
            print(f"{r['trip_id']:9} {m:5} {hab:6} {csv_km:>6} {'?':>6} {'?':>8} {'?':>8}  (node missing)")
            continue
        od_list.append(od)
        acc_list.extend([acc_h, acc_d])
        print(f"{r['trip_id']:9} {m:5} {hab:6} {csv_km:>6} {od:6.2f} {acc_h:8.2f} {acc_d:8.2f}")
        if m == "pt":
            pt += 1
        elif m == "car":
            car += 1
        elif m == "bike":
            bike += 1
        else:
            walk += 1

    print(f"\nmode share: car={car} bike={bike} pt={pt} walk={walk} (n={car+bike+pt+walk})")
    print(f"OD straight-line km: min={min(od_list):.2f} med={sorted(od_list)[len(od_list)//2]:.2f} max={max(od_list):.2f}")
    print(f"metro access (home+dest) km: min={min(acc_list):.2f} med={sorted(acc_list)[len(acc_list)//2]:.2f} max={max(acc_list):.2f}")
    far = sum(1 for a in acc_list if a > 1.0)
    print(f"OD endpoints farther than 1 km from a metro stop: {far}/{len(acc_list)} ({far/len(acc_list)*100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
