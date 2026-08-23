#!/usr/bin/env python
"""Phase 9: population-scale MATSim simulation with distilled students.

Builds a synthetic population, asks the distilled student for each person's
behavioral response under several dynamic contexts, runs MATSim per scenario
(lastIteration=0: the student's decisions are executed as-is), and collects
population-level mode shares / travel distances.

Scenarios (dynamic contexts):
    baseline      clear weather, fare x1.0
    rain          weather intensity 1.0 (heavy rain)
    fare_surge    fare multiplier 2.0
    combined      rain 1.0 + fare x1.5

Usage:
    python scripts/run_population_experiment.py \
        --checkpoint outputs/student_v0_3_c/checkpoints/best.pt \
        --num-personas 1000 --trips-per-persona 2 \
        --scenarios baseline,rain,fare_surge,combined \
        --output data/population_experiment
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.config import load_yaml
from traveler_distillation.generators import PersonaGenerator, TripGenerator
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.matsim import MATSimAdapter

# ASCII-only paths: PowerShell->java argv corrupts non-ASCII paths, so the
# MATSim release is reached via a Temp junction and the Java launcher lives in
# an ASCII Temp dir.
_TMP = "C:/Users/xuan1/AppData/Local/Temp"
_CP = f"{_TMP}/matsim_run;{_TMP}/matsim_rel/matsim-2026.0.jar;{_TMP}/matsim_rel/libs/*"

SCENARIOS = {
    "baseline": {"weather_intensity": 0.0, "fare_multiplier": 1.0},
    "rain": {"weather_intensity": 1.0, "fare_multiplier": 1.0},
    "fare_surge": {"weather_intensity": 0.0, "fare_multiplier": 2.0},
    "combined": {"weather_intensity": 1.0, "fare_multiplier": 1.5},
}


def make_context(name: str, gen_cfg: dict) -> DynamicContext:
    spec = SCENARIOS[name]
    baseline = gen_cfg.get("baseline_context", {})
    wi = spec["weather_intensity"]
    return DynamicContext(
        context_id=f"C_POP_{name}",
        weather=Weather(condition="clear" if wi <= 0 else "rain", intensity=wi),
        road_congestion=baseline.get("road_congestion", 0.3),
        transit_delay_min=baseline.get("transit_delay_min", 0),
        transit_disruption=baseline.get("transit_disruption", False),
        road_disruption=baseline.get("road_disruption", False),
        fare_multiplier=spec["fare_multiplier"],
        parking_cost_multiplier=baseline.get("parking_cost_multiplier", 1.0),
        congestion_charge=baseline.get("congestion_charge", 0.0),
    )


def run_matsim(scenario_dir: Path) -> tuple[int, str]:
    """Run the preloaded-scenario launcher; returns (exit code, log tail)."""
    proc = subprocess.run(
        ["java", "-Xmx2g", "-cp", _CP, "RunMatsimPreloaded", "config.xml"],
        cwd=str(scenario_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    log_path = scenario_dir / "output" / "logfileWarningsErrors.log"
    tail = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-6:])
    return proc.returncode, tail


def _parse_modestats(path: Path) -> dict:
    """MATSim modestats.csv: iteration;bike;car;pt;walk -> {mode: share}."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split(";")
    values = lines[1].split(";")
    return {h: float(v) for h, v in zip(header[1:], values[1:])}


def _parse_traveldist(path: Path) -> dict:
    """MATSim traveldistancestats.csv: avg leg / avg trip distance (meters)."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split(";")
    values = lines[1].split(";")
    out = {}
    for h, v in zip(header[1:], values[1:]):
        key = h.strip().lower().replace(" ", "_").replace(".", "")
        key = key.replace("avg_", "avg_", 1)
        if key.startswith("avg_average_leg"):
            out["avg_leg_distance_m"] = float(v)
        elif key.startswith("avg_average_trip"):
            out["avg_trip_distance_m"] = float(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--pop-seed", type=int, default=2026)
    ap.add_argument("--num-personas", type=int, default=1000)
    ap.add_argument("--trips-per-persona", type=int, default=2)
    ap.add_argument("--grid-n", type=int, default=20)
    ap.add_argument("--scenarios", default="baseline,rain,fare_surge,combined")
    ap.add_argument("--output", default="data/population_experiment")
    ap.add_argument("--skip-matsim", action="store_true")
    args = ap.parse_args()

    gen_cfg = load_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    n = args.num_personas
    tpp = args.trips_per_persona
    personas = PersonaGenerator(seed=args.pop_seed, config=gen_cfg).generate(n)
    all_trips = TripGenerator(seed=args.pop_seed, config=gen_cfg).generate(n * tpp)
    trips_per_persona = [
        all_trips[i * tpp : (i + 1) * tpp] for i in range(n)
    ]

    adapter = MATSimAdapter(args.checkpoint)
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    results = {}
    for name in scenario_names:
        if name not in SCENARIOS:
            print(f"unknown scenario {name}, skipping")
            continue
        context = make_context(name, gen_cfg)
        sdir = out / name
        print(f"\n===== scenario {name} ({n} personas x {tpp} trips) =====", flush=True)
        manifest = adapter.build_scenario(
            personas, all_trips, context, sdir, grid_n=args.grid_n,
            trips_per_persona=trips_per_persona,
        )
        student_modes = Counter(m["student_mode"] for m in manifest)
        total = sum(student_modes.values())
        student_share = {m: round(c / total, 4) for m, c in student_modes.items()}

        record = {
            "scenario": name,
            "context": SCENARIOS[name],
            "n_personas": n,
            "n_trips": len(manifest),
            "student_mode_counts": dict(student_modes),
            "student_mode_share": student_share,
            "matsim_exit_code": None,
            "matsim_mode_share": None,
            "avg_leg_distance_m": None,
            "avg_trip_distance_m": None,
            "log_tail": "",
        }
        if not args.skip_matsim:
            code, tail = run_matsim(sdir)
            record["matsim_exit_code"] = code
            record["log_tail"] = tail
            record["matsim_mode_share"] = _parse_modestats(sdir / "output" / "modestats.csv")
            record.update(_parse_traveldist(sdir / "output" / "traveldistancestats.csv"))
            if code != 0:
                print(f"  MATSim FAILED (exit {code}); log tail:\n{tail}", flush=True)
            else:
                print(f"  MATSim OK; executed mode share = {record['matsim_mode_share']}", flush=True)
        results[name] = record

    (out / "population_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # quick comparison table
    print("\n===== SUMMARY =====")
    print(f"{'scenario':<12} {'student bike/car/pt/walk':<35} {'MATSim executed share'}")
    for name, r in results.items():
        sm = r["student_mode_share"]
        mm = r["matsim_mode_share"] or {}
        s = f"b{sm.get('bike', 0):.2f}/c{sm.get('car', 0):.2f}/p{sm.get('pt', 0):.2f}/w{sm.get('walk', 0):.2f}"
        m = f"b{mm.get('bike', 0):.2f}/c{mm.get('car', 0):.2f}/p{mm.get('pt', 0):.2f}/w{mm.get('walk', 0):.2f}"
        print(f"{name:<12} {s:<35} {m}")
    print(f"\nwrote {out / 'population_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
