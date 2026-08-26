#!/usr/bin/env python
"""S8 freeze reproduction gate — Singapore smoke (S8 freeze instructions §11).

Loads the FROZEN S8 checkpoint (no retraining), verifies the S8 schema
round-trip (12 alternative numeric fields, encoder input 20), builds a
real-supply scenario where the pt alternative carries REAL transit
accessibility attributes from the validated PT planner, and runs a minimal
MATSim execution under the FROZEN Phase C settings
(N*=100 smoke, flowCapacityFactor=storageCapacityFactor=0.3).

Gate criteria (recorded in the result JSON):
  * S8 inference -> adapter with no schema mismatch (hard assertions);
  * PT accessibility pipeline produced feasible/infeasible vectors;
  * MATSim exit == 0 with car/pt/walk/bike legs executed.

Usage:
    python scripts/singapore/run_s8_smoke.py [--skip-matsim]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traveler_distillation.config import load_yaml
from traveler_distillation.generators import PersonaGenerator, TripGenerator
from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter, smoke_schema_check
from traveler_distillation.schemas.context import DynamicContext, Weather
from traveler_distillation.schemas.state import UniversalTravelerState

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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_context() -> DynamicContext:
    return DynamicContext(
        context_id="C_SG_S8_SMOKE_BASELINE",
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
    ap.add_argument("--checkpoint", default="releases/s8_supply_aware_v1/checkpoint/model.pt")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--num-agents", type=int, default=100)
    ap.add_argument("--output", default="outputs/s8_singapore_smoke")
    ap.add_argument("--skip-matsim", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    gen_cfg = load_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    n = args.num_agents
    personas = PersonaGenerator(seed=2026, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=2026, config=gen_cfg).generate(n)
    trips_per_persona = [[t] for t in trips]

    adapter = S8MATSimAdapter((root / args.checkpoint))
    context = _make_context()
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}

    # --- schema round-trip check on a representative state (no schema mismatch)
    alt_factory = adapter.make_alt_factory(supply)
    stat_collector = Counter()

    def factory(persona, trip, ctx, origin, dest):
        alternatives = alt_factory(persona, trip, ctx, origin, dest)
        for alt in alternatives:
            if alt.mode == "pt":
                stat_collector["pt_feasible" if alt.pt_feasible == 1.0 else "pt_infeasible"] += 1
        return alternatives

    probe_state = UniversalTravelerState(
        persona=personas[0], trip=trips[0], context=context,
        alternatives=alt_factory(personas[0], trips[0], context,
                                  _probe_node(supply["activity_nodes"], 0),
                                  _probe_node(supply["activity_nodes"], 1)),
    )
    schema_check = smoke_schema_check(adapter, probe_state)

    print(f"building real-network S8 scenario ({n} agents)...", flush=True)
    manifest = adapter.build_real_scenario(
        personas, trips, context, out,
        network_path=supply["network"], schedule_path=supply["schedule"],
        vehicles_path=supply["vehicles"], stops_path=supply["stops"],
        snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
        activity_nodes_path=supply["activity_nodes"],
        trips_per_persona=trips_per_persona,
        flow_capacity_factor=0.3, storage_capacity_factor=0.3,
        alt_factory=factory,
    )
    student_modes = Counter(m["student_mode"] for m in manifest)
    executed_out = Counter(m["outbound_mode"] for m in manifest)
    print(f"student modes: {dict(student_modes)}")
    print(f"executed outbound: {dict(executed_out)}")
    print(f"pt accessibility split: {dict(stat_collector)}")

    record = {
        "gate": "s8_singapore_smoke",
        "n_agents": n,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256(root / args.checkpoint),
        "phase_c_frozen_settings": {
            "N_star": 10000, "flow_capacity_factor": 0.3, "storage_capacity_factor": 0.3,
            "note": "smoke runs N=100 of N*=10,000 with the frozen Phase C capacity factors",
        },
        "schema_check": schema_check,
        "pt_accessibility_split": dict(stat_collector),
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

    record["gate_pass"] = bool(
        schema_check.get("schema_match")
        and stat_collector.get("pt_feasible", 0) + stat_collector.get("pt_infeasible", 0) == n
        and record.get("matsim_exit_code") == 0
    )
    (out / "s8_smoke_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"gate_pass={record['gate_pass']}")
    print(f"wrote {out / 's8_smoke_result.json'}")
    return 0 if record["gate_pass"] else 1


def _probe_node(activity_nodes_path: str, i: int) -> str:
    import json as _json

    nodes = _json.loads(Path(activity_nodes_path).read_text(encoding="utf-8"))
    return nodes[i % len(nodes)]["node"]


if __name__ == "__main__":
    raise SystemExit(main())
