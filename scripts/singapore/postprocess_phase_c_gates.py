#!/usr/bin/env python
"""Post-process Phase C result JSONs with the amended gate semantics.

The Phase C run started before the stuck-split diagnosis; its result JSONs
record the TOTAL stuckAndAbort count without separating transit vehicles
(aborted at the 30:00 simulation end — a pre-existing property of the frozen
10k + 0.3 setting, see PHASE_C_SINGAPORE_SCENARIO_INSTRUCTIONS.md §4.1) from
human agents. This script streams each scenario's events file once, splits the
stuck counts, recomputes the gate with the amended criteria, and regenerates
the comparison report.

Usage:
    python scripts/singapore/postprocess_phase_c_gates.py \
        --output outputs/singapore_phase_c
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PERSON = re.compile(r"^P\d+$")


def split_stuck(events_path: Path) -> dict:
    import zstandard

    out = {"stuck_persons": 0, "stuck_transit_vehicles": 0, "vehicle_aborts": 0}
    if not events_path.exists():
        return out
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(open(events_path, "rb")) as f:
        for event, el in ET.iterparse(f, events=("end",)):
            if el.tag != "event":
                continue
            t = el.get("type")
            if t == "stuckAndAbort":
                person = el.get("person") or ""
                if person.startswith("pt_veh"):
                    out["stuck_transit_vehicles"] += 1
                elif _PERSON.match(person):
                    out["stuck_persons"] += 1
            elif t == "vehicle aborts":
                out["vehicle_aborts"] += 1
            el.clear()
    return out


def amended_gate(name: str, record: dict) -> dict:
    m = record.get("metrics") or {}
    checks = {}
    checks["matsim_exit_0"] = record.get("matsim_exit_code") == 0
    checks["stuck_persons_le_b5c_baseline"] = m.get("stuck_persons", 10 ** 9) <= 266
    checks["pt_alightings_le_boardings"] = m.get("pt_alightings", 0) <= m.get("pt_boardings", 0) or (
        not m.get("pt_boardings"))
    checks["legs_executed"] = bool(m.get("leg_departures"))
    if name == "C0_baseline":
        checks["four_modes_executed"] = {"car", "pt", "walk", "bike"} <= set(m.get("leg_departures", {}))
    checks["note"] = ("stuckAndAbort total includes transit vehicles aborted at sim end (30:00) "
                      f"({m.get('stuck_transit_vehicles', '?')} here; B.5C 0.3 baseline: 4,347 of 20,966 departures). "
                      f"Human stuck: {m.get('stuck_persons', '?')} (B.5C baseline: 266).")
    passed = all(v is not False for k, v in checks.items() if k != "note")
    summary = "; ".join(f"{k}={v}" for k, v in checks.items() if k != "note")
    return {"pass": passed, "checks": checks, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/singapore_phase_c")
    args = ap.parse_args()

    root = Path(args.output)
    records = []
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        res_path = sdir / "phase_c_result.json"
        if not res_path.exists():
            continue
        record = json.loads(res_path.read_text(encoding="utf-8"))
        events = sdir / "output" / "ITERS" / "it.0" / "0.events.xml.zst"
        split = split_stuck(events)
        m = record.setdefault("metrics", {})
        m.update(split)
        record["gate"] = amended_gate(record["scenario"], record)
        res_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{record['scenario']}: stuck_persons={split['stuck_persons']}, "
              f"stuck_transit_vehicles={split['stuck_transit_vehicles']}, "
              f"vehicle_aborts={split['vehicle_aborts']}, gate={'PASS' if record['gate']['pass'] else 'FAIL'}")
        records.append(record)

    if len(records) > 1:
        from run_phase_c import write_report

        write_report(root, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
