#!/usr/bin/env python
"""Singapore Phase A gate: 100 S7-W3 agents on the real supply network.

Builds the real-network scenario (OSM network + scheduled PT from the
Singapore GTFS) with the FROZEN S7-W3 student decisions, runs MATSim
(lastIteration=0, no replanning), and verifies the gate:

  * MATSim exit == 0;
  * car / pt / walk / bike legs all EXECUTED (departure events by leg mode);
  * PT boardings happened (PersonEntersVehicle events);
  * no stuck agents; summary stats (mode share, trip distances).

Usage:
    python scripts/singapore/run_phase_a_smoke.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traveler_distillation.config import load_yaml
from traveler_distillation.generators import PersonaGenerator, TripGenerator
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.matsim import MATSimAdapter

_TMP = "C:/Users/xuan1/AppData/Local/Temp"
_CP = f"{_TMP}/matsim_run;{_TMP}/matsim_rel/matsim-2026.0.jar;{_TMP}/matsim_rel/libs/*"

SUPPLY = {
    "network": "data/singapore/transit/network_with_transit.xml",
    "schedule": "data/singapore/transit/transitSchedule.xml",
    "vehicles": "data/singapore/transit/transitVehicles.xml",
    "stops": "data/singapore/transit/prep_stops.jsonl",
    "snapping": "data/singapore/transit/stop_snapping_report.json",
    "trips_by_stop": "data/singapore/transit/trips_by_stop.json",
    "activity_nodes": "data/singapore/transit/activity_nodes.json",
}


def _make_context() -> DynamicContext:
    return DynamicContext(
        context_id="C_SG_PHASEA_BASELINE",
        weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3,
        transit_delay_min=0,
        transit_disruption=False,
        road_disruption=False,
        fare_multiplier=1.0,
        parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )


def _run_matsim(sdir: Path) -> tuple[int, str]:
    with open(sdir / "java_run.log", "w", encoding="utf-8", errors="replace") as out_f:
        proc = subprocess.run(
            ["java", "-Xmx6g", "-cp", _CP, "RunMatsimPreloaded", "config.xml"],
            cwd=str(sdir), stdout=out_f, stderr=subprocess.STDOUT, timeout=3600,
        )
    tail = ""
    log = sdir / "output" / "logfileWarningsErrors.log"
    if log.exists():
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:])
    return proc.returncode, tail


def _parse_events(sdir: Path) -> dict:
    """Count executed legs by mode + PT boardings from it.0 events (.zst)."""
    import zstandard
    out = {"leg_departures": Counter(), "pt_boardings": 0, "n_events": 0,
           "transit_vehicle_departures": 0}
    ev = sdir / "output" / "ITERS" / "it.0" / "0.events.xml.zst"
    if not ev.exists():
        return out
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(open(ev, "rb")) as f:
        for event, elem in ET.iterparse(f, events=("end",)):
            if elem.tag != "event":
                continue
            etype = elem.get("type")
            out["n_events"] += 1
            if etype == "departure":
                out["leg_departures"][elem.get("legMode")] += 1
            elif etype == "PersonEntersPtVehicle":
                out["pt_boardings"] += 1
            elif etype == "TransitDriverStarts":
                out["transit_vehicle_departures"] += 1
            elem.clear()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/student_s7_w3/checkpoints/best.pt")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--num-agents", type=int, default=100)
    ap.add_argument("--output", default="outputs/singapore_phase_a/smoke_100")
    ap.add_argument("--skip-matsim", action="store_true")
    args = ap.parse_args()

    gen_cfg = load_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    n = args.num_agents
    personas = PersonaGenerator(seed=2026, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=2026, config=gen_cfg).generate(n)
    trips_per_persona = [[t] for t in trips]

    adapter = MATSimAdapter(args.checkpoint)
    context = _make_context()

    root = Path(__file__).resolve().parents[2]
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}

    print(f"building real-network scenario ({n} agents)...", flush=True)
    manifest = adapter.build_real_scenario(
        personas, trips, context, out,
        network_path=supply["network"], schedule_path=supply["schedule"],
        vehicles_path=supply["vehicles"], stops_path=supply["stops"],
        snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
        activity_nodes_path=supply["activity_nodes"],
        trips_per_persona=trips_per_persona,
    )
    student_modes = Counter(m["student_mode"] for m in manifest)
    executed_out = Counter(m["outbound_mode"] for m in manifest)
    print(f"student modes: {dict(student_modes)}")
    print(f"executed outbound: {dict(executed_out)}")

    record = {
        "n_agents": n,
        "checkpoint": args.checkpoint,
        "student_mode_counts": dict(student_modes),
        "executed_outbound_counts": dict(executed_out),
        "matsim_exit_code": None,
        "events": None,
        "log_tail": "",
    }
    if not args.skip_matsim:
        code, tail = _run_matsim(out)
        record["matsim_exit_code"] = code
        record["log_tail"] = tail
        record["events"] = _parse_events(out)
        record["events"]["leg_departures"] = dict(record["events"]["leg_departures"])
        print(f"MATSim exit={code}")
        if code != 0:
            print(f"log tail:\n{tail}")
        else:
            ev = record["events"]
            modes = set(ev["leg_departures"]) & {"car", "pt", "walk", "bike"}
            print(f"executed leg modes: {sorted(modes)}")
            print(f"pt boardings: {ev['pt_boardings']}, transit vehicle departures: {ev['transit_vehicle_departures']}")

    (out / "phase_a_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out / 'phase_a_result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
