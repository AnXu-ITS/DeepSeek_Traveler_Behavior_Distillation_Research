#!/usr/bin/env python
"""Phase B.5C step 2: calibration operating-point tests (10k baseline each).

Schemes:
  gc35 / gc45 / gc55  : intersection-aware effective approach capacities
                        (green-ratio sensitivity, C_eff = C_osm * g/C)
  sample              : MATSim flowCapacityFactor/storageCapacityFactor = 0.12
                        (sample-factor proxy, no network change)

Each scheme runs ONE 10k baseline with the same seed; congestion metrics are
collected with the shared collector. The operating point is chosen as
"light-to-moderate congestion with dynamic headroom" (slow-passage share in
~0.5-5%, links delay>15s share in ~1-10%, no gridlock: failed trips ≈ 0).

Usage:
    python scripts/singapore/run_b5c_calibration.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traveler_distillation.config import load_yaml
from traveler_distillation.generators import PersonaGenerator, TripGenerator
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.matsim import MATSimAdapter
from traveler_distillation.singapore.scenario_metrics import load_link_attrs, collect_metrics

_TMP = "C:/Users/xuan1/AppData/Local/Temp"
_CP = f"{_TMP}/matsim_run;{_TMP}/matsim_rel/matsim-2026.0.jar;{_TMP}/matsim_rel/libs/*"

BASE_SUPPLY = {
    "network": "data/singapore/transit/network_with_transit.xml",
    "schedule": "data/singapore/transit/transitSchedule.xml",
    "vehicles": "data/singapore/transit/transitVehicles.xml",
    "stops": "data/singapore/transit/prep_stops.jsonl",
    "snapping": "data/singapore/transit/stop_snapping_report.json",
    "trips_by_stop": "data/singapore/transit/trips_by_stop.json",
    "activity_nodes": "data/singapore/transit/activity_nodes.json",
}

SCHEMES = [
    ("gc35", {"network": "data/singapore/transit/network_gc35.xml"}, {}),
    ("gc45", {"network": "data/singapore/transit/network_gc45.xml"}, {}),
    ("gc55", {"network": "data/singapore/transit/network_gc55.xml"}, {}),
    ("sample", {}, {"flow_capacity_factor": 0.12, "storage_capacity_factor": 0.12}),
]


def _make_context() -> DynamicContext:
    return DynamicContext(
        context_id="C_SG_B5C_BASELINE",
        weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/student_s7_w3/checkpoints/best.pt")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--n-agents", type=int, default=10000)
    ap.add_argument("--output", default="outputs/singapore_phase_b5/calibration")
    ap.add_argument("--skip-matsim", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    gen_cfg = load_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    adapter = MATSimAdapter(args.checkpoint)
    context = _make_context()
    n = args.n_agents

    personas = PersonaGenerator(seed=4000, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=4000, config=gen_cfg).generate(n)
    trips_per_persona = [[t] for t in trips]

    results = {}
    existing_path = out / "calibration_results.json"
    if existing_path.exists():
        results = json.loads(existing_path.read_text(encoding="utf-8"))
        print(f"resuming: {list(results)} already done")
    for name, supply_over, qsim_kwargs in SCHEMES:
        if name in results and results[name].get("matsim_exit_code") is not None:
            print(f"[skip] {name} (done)")
            continue
        supply = {k: str((root / v).resolve()) for k, v in BASE_SUPPLY.items()}
        supply.update({k: str((root / v).resolve()) for k, v in supply_over.items()})
        sdir = out / f"baseline_{name}"
        print(f"\n===== scheme {name} =====", flush=True)
        t0 = time.time()
        manifest = adapter.build_real_scenario(
            personas, trips, context, sdir,
            network_path=supply["network"], schedule_path=supply["schedule"],
            vehicles_path=supply["vehicles"], stops_path=supply["stops"],
            snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
            activity_nodes_path=supply["activity_nodes"],
            trips_per_persona=trips_per_persona,
            **qsim_kwargs,
        )
        build_s = time.time() - t0
        record = {"scheme": name, "n_agents": n, "build_seconds": round(build_s, 1),
                  "student_modes": dict(Counter(m["student_mode"] for m in manifest))}
        if not args.skip_matsim:
            with open(sdir / "java_run.log", "w", encoding="utf-8", errors="replace") as f:
                t0 = time.time()
                proc = subprocess.run(
                    ["java", "-Xmx8g", "-cp", _CP, "RunMatsimPreloaded", "config.xml"],
                    cwd=str(sdir), stdout=f, stderr=subprocess.STDOUT, timeout=10800,
                )
                record["matsim_exit_code"] = proc.returncode
                record["runtime_seconds"] = round(time.time() - t0, 1)
            links = load_link_attrs(supply["network"])
            record["metrics"] = collect_metrics(sdir / "output" / "ITERS" / "it.0" / "0.events.xml.zst", links)
            print(json.dumps(record, ensure_ascii=False, indent=2))
        results[name] = record
        (out / "calibration_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Phase B.5C — 标定方案对比（10k baseline）\n",
             "| 方案 | slow-passage | links>15s | mean delay | car travel (min) | failed trips | runtime |\n"
             "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        m = r.get("metrics") or {}
        lines.append(
            f"| {name} | {m.get('network_congestion_slow_share', '—')} | "
            f"{m.get('share_links_delay_gt_15s', '—')} | {m.get('road_delay_mean_s_per_passage', '—')} | "
            f"{m.get('car_mean_travel_time_min', '—')} | {m.get('failed_trips', '—')} | "
            f"{r.get('runtime_seconds', '—')} |")
    lines.append("\n## 选择规则\n- 轻中度拥堵且有余量：slow-passage ≈0.5–5%、links>15s ≈1–10%、"
                 "failed trips ≈0（无 gridlock）。\n- 冻结后 Phase C 全部情景统一使用。\n")
    (out / "calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'calibration_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
