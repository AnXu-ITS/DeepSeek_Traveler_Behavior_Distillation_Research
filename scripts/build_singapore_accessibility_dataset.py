#!/usr/bin/env python
"""S8 — build the stratified Singapore real-supply accessibility dataset.

Steps 4-7 of S8_TRANSIT_ACCESSIBILITY_TRAINING_INSTRUCTIONS.md:
  1. load the Singapore supply (OSM network + GTFS indexes);
  2. probe candidate OD pairs at a reference departure and stratify into
     accessibility Classes A-E;
  3. build split-disjoint OD pools (persona + OD + accessibility holds);
  4. assign each (persona, trip) group a counterfactual accessibility curve;
  5. materialize records: state = persona + trip + baseline context + real
     alternatives (pt carries the real accessibility vector; NO location ids);
  6. freeze the splits and run the sanity checks.

Outputs (data/singapore_accessibility/):
    records.jsonl            S8 states (student/teacher-facing; city-independent)
    od_manifest.json         anonymous od_index -> routing provenance (internal only)
    split_manifest.json      persona / OD / accessibility holds + curve groups
    boundary_pairs.json      K=5 teacher-sampling candidates (S8 §13)
    candidate_probe.jsonl    stratification evidence for every probed OD
    sanity_report.json       feature sanity checks (S8 §5)

Usage:
    python scripts/build_singapore_accessibility_dataset.py [--n-random 1500 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

from traveler_distillation.accessibility.accessibility_dataset import (
    ALL_CLASSES,
    PERSONA_SPLIT,
    build_dataset,
    build_groups,
    build_od_pools,
    sample_od_candidates,
    sanity_check,
)
from traveler_distillation.accessibility.gtfs_accessibility import SupplyIndex
from traveler_distillation.config import load_yaml
from traveler_distillation.generators import PersonaGenerator, TripGenerator

SUPPLY = {
    "network": "data/singapore/transit/network_with_transit.xml",
    "stops": "data/singapore/transit/prep_stops.jsonl",
    "snapping": "data/singapore/transit/stop_snapping_report.json",
    "trips_by_stop": "data/singapore/transit/trips_by_stop.json",
    "activity_nodes": "data/singapore/transit/activity_nodes.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/singapore_accessibility")
    ap.add_argument("--n-random", type=int, default=1500)
    ap.add_argument("--n-near-stops", type=int, default=700)
    ap.add_argument("--n-mid-access", type=int, default=800)
    ap.add_argument("--n-no-stops", type=int, default=500)
    ap.add_argument("--probe-dep-min", type=float, default=480.0, help="probe departure (8:00 AM)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = _ROOT
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    supply = {k: str(root / v) for k, v in SUPPLY.items()}

    t0 = time.time()
    print("loading Singapore supply ...")
    idx = SupplyIndex(supply)
    print(f"supply loaded in {time.time() - t0:.1f}s: "
          f"{len(idx.stops)} stops, {len(idx.trips_by_stop)} served stops, "
          f"{len(idx.trip_seq)} trips, {len(idx.activity_nodes)} activity nodes")

    probe_dep = args.probe_dep_min * 60.0
    print(f"probing candidate OD pairs (dep={args.probe_dep_min} min) ...")
    cands = sample_od_candidates(
        idx, n_random=args.n_random, n_near_stops=args.n_near_stops,
        n_mid_access=args.n_mid_access, n_no_stops=args.n_no_stops,
        dep_sec=probe_dep, seed=args.seed,
    )
    cls_counts = Counter(c["class"] for c in cands)
    print(f"candidates: {len(cands)}; class distribution: {dict(cls_counts)}")

    pools = build_od_pools(cands, np.random.default_rng(args.seed))
    print("OD pools per split:")
    for split, by_cls in pools.items():
        print(f"  {split}: " + ", ".join(f"{c.split('_')[0]}={len(v)}" for c, v in by_cls.items()))

    # personas (S3-convention 40) and trips
    gen_cfg = load_yaml(str(root / "configs" / "generation_v0_1.yaml"))
    personas = PersonaGenerator(seed=42, config=gen_cfg).generate(20, id_offset=0)
    personas += PersonaGenerator(seed=43, config=gen_cfg).generate(20, id_offset=20)
    assert len({p.persona_id for p in personas}) == 40
    trips_by_persona: dict[str, list] = {}
    for i, p in enumerate(personas[:20]):
        trips_by_persona[p.persona_id] = TripGenerator(seed=42, config=gen_cfg).generate(2, 0)
    for i, p in enumerate(personas[20:]):
        trips_by_persona[p.persona_id] = TripGenerator(seed=43, config=gen_cfg).generate(2, 2)

    personas_by_split = {
        s: [p for p in personas if p.persona_id in PERSONA_SPLIT[s]]
        for s in ("train", "val", "test")
    }
    print("personas:", {s: len(v) for s, v in personas_by_split.items()})

    groups = build_groups(personas_by_split, trips_by_persona, pools,
                          np.random.default_rng(args.seed + 1))
    print(f"curve groups: {len(groups)} "
          f"({Counter(g['split'] for g in groups)})")

    print("materializing records (real itinerary per state) ...")
    t1 = time.time()
    records, od_manifest, boundary_pairs = build_dataset(idx, groups)
    print(f"records: {len(records)} in {time.time() - t1:.1f}s")

    # identity-leak guard: forbid every assigned OD node id + every stop id + route ids
    forbidden = {m["origin"] for m in od_manifest} | {m["dest"] for m in od_manifest}
    forbidden |= set(idx.stops.keys())
    forbidden |= {t["route_id"] for entries in idx.trips_by_stop.values() for t in entries[:5]}
    report = sanity_check(records, od_manifest, personas_by_split,
                          forbidden_strings=sorted(forbidden))
    print("sanity:", json.dumps(report, ensure_ascii=False, indent=2))

    # split manifest (frozen)
    split_manifest = {
        "dataset": "singapore_accessibility",
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "supply_sources": {
            "network": SUPPLY["network"], "stops": SUPPLY["stops"],
            "trips_by_stop": SUPPLY["trips_by_stop"],
            "note": "Singapore OSM + community-built GTFS feed (see data/singapore/DATA_README.md); "
                    "accessibility attributes are city-independent (S8 §3)",
        },
        "persona_split": PERSONA_SPLIT,
        "persona_split_counts": {s: len(v) for s, v in personas_by_split.items()},
        "od_split": {
            s: [m["od_index"] for m in od_manifest if m["split"] == s]
            for s in ("train", "val", "test")
        },
        "accessibility_holdout": {
            "held_out_from_train": "feasible ODs with walking burden access_time+egress_time >= 15 min (high-burden profiles appear only in test)",
            "transfer_rule": "planner caps at 1 vehicle-to-vehicle transfer (same routing rule as MATSimAdapter._plan_pt)",
            "class_e_policy": "Class E (infeasible) present in every split (FVR learnable + testable)",
        },
        "curve_groups": {
            s: sorted({r["curve_group"] for r in records if r["split"] == s})
            for s in ("train", "val", "test")
        },
        "boundary_pairs_k5": boundary_pairs,
        "sanity": report,
    }

    with (out / "records.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "od_manifest.json").write_text(json.dumps(od_manifest, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    (out / "boundary_pairs.json").write_text(json.dumps(boundary_pairs, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    with (out / "candidate_probe.jsonl").open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (out / "sanity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"total states: {len(records)} (target 300-800); boundary pairs: {len(boundary_pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
