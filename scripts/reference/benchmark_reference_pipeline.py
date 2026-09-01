#!/usr/bin/env python
"""Reference pipeline benchmark: first run (cold cache) vs second run (warm cache).

Generates a dev-seed population CSV + a benchmark YAML under
outputs/reference_pipeline/bench/, then runs the reference pipeline twice on
the SAME config: run 1 builds the route cache, run 2 reuses it. Reports the
wall-clock breakdown and cache hit-rate from both run_summary.json files.

Usage:
    python scripts/reference/benchmark_reference_pipeline.py [--n 500]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import PersonaGenerator, TripGenerator  # noqa: E402
from reference_pipeline import ReferencePipeline  # noqa: E402

BENCH = ROOT / "outputs/reference_pipeline/bench"


def make_population_csv(n: int, path: Path, seed: int = 0) -> None:
    gen_cfg = load_yaml(ROOT / "configs" / "generation_v0_1.yaml")
    personas = PersonaGenerator(seed=seed, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=seed, config=gen_cfg).generate(n)
    columns = ["persona_id", "age_group", "income_group", "occupation",
               "household_size", "has_children", "car_ownership", "driving_license",
               "bike_ownership", "transit_pass", "habitual_mode",
               "schedule_flexibility", "mobility_limitation",
               "trip_id", "purpose", "origin_type", "destination_type",
               "distance_km", "desired_departure_min", "desired_arrival_min",
               "time_constraint"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for p, t in zip(personas, trips):
            w.writerow({
                "persona_id": p.persona_id, "age_group": p.age_group,
                "income_group": p.income_group, "occupation": p.occupation,
                "household_size": p.household_size, "has_children": p.has_children,
                "car_ownership": p.car_ownership, "driving_license": p.driving_license,
                "bike_ownership": p.bike_ownership, "transit_pass": p.transit_pass,
                "habitual_mode": p.habitual_mode,
                "schedule_flexibility": p.schedule_flexibility,
                "mobility_limitation": p.mobility_limitation,
                "trip_id": t.trip_id, "purpose": t.purpose,
                "origin_type": t.origin_type, "destination_type": t.destination_type,
                "distance_km": t.distance_km,
                "desired_departure_min": t.desired_departure_min,
                "desired_arrival_min": t.desired_arrival_min,
                "time_constraint": t.time_constraint,
            })


def write_bench_config(n: int, path: Path) -> None:
    text = f"""pipeline_name: reference_benchmark
log_level: INFO
paths:
  population_input: {BENCH / 'population.csv'}
  matsim_network: data/singapore/transit/network_with_transit.xml
  transit_schedule: data/singapore/transit/transitSchedule.xml
  transit_vehicles: data/singapore/transit/transitVehicles.xml
  transit_stops: data/singapore/transit/prep_stops.jsonl
  stop_snapping: data/singapore/transit/stop_snapping_report.json
  trips_by_stop: data/singapore/transit/trips_by_stop.json
  activity_nodes: data/singapore/transit/activity_nodes.json
  output_dir: {BENCH / f'run_{n}'}
  student_checkpoint: releases/s9_supply_aware_v2/checkpoint/model.pt
student:
  model_type: student_s8_v1
  batch_size: 256
  device: auto
scenario:
  context_id: bench
  weather: {{condition: clear, intensity: 0.0}}
  road_congestion: 0.3
  transit_delay_min: 0
  transit_disruption: false
  road_disruption: false
  fare_multiplier: 1.0
  parking_cost_multiplier: 1.0
  congestion_charge: 0.0
cache:
  enabled: true
  directory: {BENCH / f'cache_{n}'}
  reuse: true
  rebuild: false
feature_mapping: {{}}
matsim:
  run_after_build: false
  config_file: ""
  java: java
  java_xmx: 6g
  timeout_s: 3600
  matsim_dir: tools/matsim-2026.0-release/matsim-2026.0
  launcher_java: tools/java/RunMatsimPreloaded.java
  classpath_override: ""
  flow_capacity_factor: 0.3
  storage_capacity_factor: 0.3
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    n = args.n

    make_population_csv(n, BENCH / "population.csv")
    cfg_path = BENCH / "bench_config.yaml"
    write_bench_config(n, cfg_path)

    results = {}
    for run_i, label in ((1, "cold"), (2, "warm")):
        print(f"===== run {run_i} ({label}) =====", flush=True)
        t0 = time.perf_counter()
        pipeline = ReferencePipeline(cfg_path, ROOT)
        summary = pipeline.run(run_matsim_flag=False)
        wall = time.perf_counter() - t0
        results[label] = {
            "wall_clock_s": round(wall, 1),
            "timings": summary["timings"],
            "cache": summary["cache"],
            "mode_distribution": summary["mode_distribution"]["student_mode_counts"],
        }
        print(f"[{label}] wall={wall:.1f}s, cache_hit_rate="
              f"{summary['cache']['hit_rate'] if summary['cache'] else None}")
        print(f"[{label}] timings={summary['timings']}")

    speedup = results["cold"]["timings"]["total_build_s"] / max(
        1e-9, results["warm"]["timings"]["total_build_s"])
    report = {
        "n": n,
        "cold": results["cold"],
        "warm": results["warm"],
        "total_build_speedup_cold_vs_warm": round(speedup, 2),
        "cold_feature_saved_s": round(
            results["cold"]["timings"]["feature_preparation_s"]
            - results["warm"]["timings"]["feature_preparation_s"], 2),
    }
    (BENCH / f"benchmark_report_{n}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
