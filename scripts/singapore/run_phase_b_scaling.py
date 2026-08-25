#!/usr/bin/env python
"""Singapore Phase B: scale validation (100 -> 500 -> 1000 agents).

Same frozen S7-W3 checkpoint + the SAME real supply network; only the
population grows. Per scale the runner records:

  routing failures (planning-level fallbacks + MATSim-level failed trips)
  PT boardings            (PersonEntersPtVehicle events)
  mode share              (modestats.csv, executed outbound)
  mean trip time          (per-person departure->arrival pairs)
  road delay              (car link passages: actual - freeflow, seconds)
  runtime                 (MATSim wall time)
  failed trips            (departures without matching arrivals + stuckAndAbort)
  network congestion      (share of car passages slower than 1.5x freeflow)

and writes outputs/singapore_phase_b/scale_metrics.json +
scale_report.md with anomaly checks.

Usage:
    python scripts/singapore/run_phase_b_scaling.py --scales 100,500,1000
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
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

_CAR_VEHICLE = re.compile(r"^P\d+$")
_PERSON = re.compile(r"^P\d+$")


def _make_context() -> DynamicContext:
    return DynamicContext(
        context_id="C_SG_PHASEB_BASELINE",
        weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )


def _run_matsim(sdir: Path) -> tuple[int, float]:
    with open(sdir / "java_run.log", "w", encoding="utf-8", errors="replace") as out_f:
        t0 = time.time()
        proc = subprocess.run(
            ["java", "-Xmx8g", "-cp", _CP, "RunMatsimPreloaded", "config.xml"],
            cwd=str(sdir), stdout=out_f, stderr=subprocess.STDOUT, timeout=10800,
        )
        wall = time.time() - t0
    return proc.returncode, wall


def _load_link_attrs() -> dict[str, tuple[float, float]]:
    txt = open("data/singapore/transit/network_with_transit.xml", encoding="utf-8").read()
    out = {}
    for m in re.finditer(
        r'<link id="([^"]+)" from="[^"]+" to="[^"]+" length="([\d.]+)" freespeed="([\d.]+)"',
        txt,
    ):
        out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def _collect_metrics(sdir: Path, links: dict[str, tuple[float, float]]) -> dict:
    """Event-based metrics from it.0 events (.zst)."""
    import zstandard
    ev_path = sdir / "output" / "ITERS" / "it.0" / "0.events.xml.zst"
    if not ev_path.exists():
        return {"events_missing": True}

    dctx = zstandard.ZstdDecompressor()
    enter_t: dict[str, float] = {}
    car_dep: dict[str, float] = {}
    dep_stack: dict[str, list[float]] = defaultdict(list)
    n_dep = Counter()
    n_arr = Counter()
    pt_boardings = 0
    stuck = 0
    trip_time = 0.0
    n_trip_legs = 0
    car_time = 0.0
    n_car_legs = 0
    car_passages = 0
    car_delay_sum = 0.0
    car_slow_passages = 0
    car_vkt_m = 0.0
    link_delay: dict[str, list[float]] = defaultdict(list)  # link -> [delays per passage]

    with dctx.stream_reader(open(ev_path, "rb")) as f:
        for event, el in ET.iterparse(f, events=("end",)):
            if el.tag != "event":
                continue
            t = float(el.get("time"))
            etype = el.get("type")
            if etype == "departure":
                person = el.get("person")
                if person and _PERSON.match(person):
                    n_dep[person] += 1
                    dep_stack[person].append(t)
                    if el.get("legMode") == "car":
                        car_dep[person] = t
            elif etype == "arrival":
                person = el.get("person")
                if person and _PERSON.match(person):
                    n_arr[person] += 1
                    if dep_stack[person]:
                        trip_time += t - dep_stack[person].pop()
                        n_trip_legs += 1
                    if el.get("legMode") == "car" and person in car_dep:
                        car_time += t - car_dep[person]
                        n_car_legs += 1
                        del car_dep[person]
            elif etype == "PersonEntersPtVehicle":
                pt_boardings += 1
            elif etype == "stuckAndAbort":
                stuck += 1
            elif etype == "entered link":
                vid = el.get("vehicle")
                if vid and _CAR_VEHICLE.match(vid):
                    enter_t[vid] = t
            elif etype == "left link":
                vid = el.get("vehicle")
                if vid in enter_t:
                    lid = el.get("link")
                    length, free = links.get(lid, (0.0, 1.0))
                    dt = t - enter_t[vid]
                    if dt <= 300.0 and free > 0 and length > 0:
                        # dt > 300s = the car is PARKED on its activity link
                        # between arrival and departure (occupancy, not driving)
                        freeflow = length / free
                        delay = max(0.0, dt - freeflow)
                        car_passages += 1
                        car_delay_sum += delay
                        car_vkt_m += length
                        link_delay[lid].append(delay)
                        if dt > freeflow + 15.0:
                            car_slow_passages += 1
                    del enter_t[vid]
            el.clear()

    failed = sum(max(0, n_dep[p] - n_arr[p]) for p in set(n_dep) | set(n_arr))
    # congestion distribution over car links with >= 1 passage
    mean_link_delays = [sum(d) / len(d) for d in link_delay.values() if d]
    mean_link_delays.sort()
    n_links_used = len(mean_link_delays)

    def _share_above(threshold):
        return round(sum(1 for d in mean_link_delays if d > threshold) / max(1, n_links_used), 4)

    return {
        "pt_boardings": pt_boardings,
        "stuck_and_abort": stuck,
        "failed_trips": failed,
        "mean_trip_time_min": round(trip_time / max(1, n_trip_legs) / 60.0, 2),
        "n_trip_legs": n_trip_legs,
        "car_mean_travel_time_min": round(car_time / max(1, n_car_legs) / 60.0, 2),
        "n_car_legs": n_car_legs,
        "road_delay_mean_s_per_passage": round(car_delay_sum / max(1, car_passages), 2),
        "network_congestion_slow_share": round(car_slow_passages / max(1, car_passages), 4),
        "car_link_passages": car_passages,
        "car_vkt_km": round(car_vkt_m / 1000.0, 1),
        "car_links_used": n_links_used,
        "share_links_delay_gt_15s": _share_above(15.0),
        "share_links_delay_gt_30s": _share_above(30.0),
        "share_links_delay_gt_60s": _share_above(60.0),
        "mean_link_delay_p50_s": round(mean_link_delays[n_links_used // 2], 2) if n_links_used else 0.0,
        "mean_link_delay_p90_s": round(mean_link_delays[int(n_links_used * 0.9)], 2) if n_links_used else 0.0,
    }


def _csv(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        return {}
    header = lines[0].split(";")
    values = lines[1].split(";")
    return {h: float(v) for h, v in zip(header[1:], values[1:])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="1000,2000,5000,10000")
    ap.add_argument("--checkpoint", default="outputs/student_s7_w3/checkpoints/best.pt")
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--output", default="outputs/singapore_phase_b5/demand_sweep")
    ap.add_argument("--skip-matsim", action="store_true")
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute metrics from existing run outputs (no build/matsim)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}
    gen_cfg = load_yaml(args.config)
    links = _load_link_attrs()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    adapter = MATSimAdapter(args.checkpoint)
    context = _make_context()

    results = {}
    if args.metrics_only:
        results = json.loads((out / "scale_metrics.json").read_text(encoding="utf-8"))
        for scale_str, record in results.items():
            sdir = out / f"scale_{scale_str}"
            record["metrics"] = _collect_metrics(sdir, links)
            print(f"scale {scale_str}: metrics recomputed")
            print(json.dumps(record["metrics"], ensure_ascii=False, indent=2))
        (out / "scale_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    for scale_str in args.scales.split(","):
        n = int(scale_str.strip())
        if args.metrics_only:
            continue
        sdir = out / f"scale_{n}"
        print(f"\n===== scale {n} =====", flush=True)
        personas = PersonaGenerator(seed=2000 + n, config=gen_cfg).generate(n)
        trips = TripGenerator(seed=2000 + n, config=gen_cfg).generate(n)
        trips_per_persona = [[t] for t in trips]

        t0 = time.time()
        manifest = adapter.build_real_scenario(
            personas, trips, context, sdir,
            network_path=supply["network"], schedule_path=supply["schedule"],
            vehicles_path=supply["vehicles"], stops_path=supply["stops"],
            snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
            activity_nodes_path=supply["activity_nodes"],
            trips_per_persona=trips_per_persona,
        )
        build_s = time.time() - t0
        student_modes = Counter(m["student_mode"] for m in manifest)
        executed_out = Counter(m["outbound_mode"] for m in manifest)
        fallbacks = json.loads((sdir / "adapter_manifest.json").read_text(encoding="utf-8"))["fallback_counts"]

        record = {
            "n_agents": n,
            "build_seconds": round(build_s, 1),
            "student_mode_counts": dict(student_modes),
            "executed_outbound_counts": dict(executed_out),
            "planning_fallbacks": fallbacks,
            "matsim_exit_code": None,
            "runtime_seconds": None,
        }
        if not args.skip_matsim:
            code, wall = _run_matsim(sdir)
            record["matsim_exit_code"] = code
            record["runtime_seconds"] = round(wall, 1)
            record["mode_share_executed"] = _csv(sdir / "output" / "modestats.csv")
            record["travel_distance"] = _csv(sdir / "output" / "traveldistancestats.csv")
            record["metrics"] = _collect_metrics(sdir, links)
            print(json.dumps(record, ensure_ascii=False, indent=2)[:1500])
        results[str(n)] = record
        (out / "scale_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- report ----
    if args.metrics_only:
        results_for_report = json.loads((out / "scale_metrics.json").read_text(encoding="utf-8"))
    else:
        results_for_report = results
    keys = list(results_for_report.keys())
    lines = [
        "# Singapore Phase B.5B — Demand Loading Sweep\n",
        "## 需求加载扫描（1000 → 2000 → 5000 → 10000 agents，同一真实供给 + 冻结 S7-W3）\n",
        "| 指标 | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys),
    ]
    rows = {k: results_for_report.get(k, {}) for k in keys}
    for label, key_fn in [
        ("routing failures (planning fallbacks 合计)", lambda r: sum(r.get("planning_fallbacks", {}).values())),
        ("failed trips (MATSim)", lambda r: (r.get("metrics") or {}).get("failed_trips")),
        ("PT boardings", lambda r: (r.get("metrics") or {}).get("pt_boardings")),
        ("mean trip time (min)", lambda r: (r.get("metrics") or {}).get("mean_trip_time_min")),
        ("car travel time (min)", lambda r: (r.get("metrics") or {}).get("car_mean_travel_time_min")),
        ("road delay (s/passage)", lambda r: (r.get("metrics") or {}).get("road_delay_mean_s_per_passage")),
        ("congestion (慢行占比)", lambda r: (r.get("metrics") or {}).get("network_congestion_slow_share")),
        ("car VKT (km)", lambda r: (r.get("metrics") or {}).get("car_vkt_km")),
        ("links delay>15s / >30s / >60s", lambda r: _fmt_link_shares(r.get("metrics"))),
        ("runtime (s)", lambda r: r.get("runtime_seconds")),
        ("mode share (bike/car/pt/walk)", lambda r: _fmt_share(r.get("mode_share_executed"))),
        ("stuckAndAbort", lambda r: (r.get("metrics") or {}).get("stuck_and_abort")),
    ]:
        cells = []
        for k in keys:
            r = rows[k]
            v = key_fn(r) if r else None
            cells.append("—" if v is None else str(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### N* 选择规则与结论\n")
    # measurable congestion with headroom: slow-passage share >= 0.5% AND
    # mean trip time <= 1.25x the 1000-agent value
    base_trip = (rows[keys[0]].get("metrics") or {}).get("mean_trip_time_min", 1.0)
    n_star = None
    for k in keys:
        m = rows[k].get("metrics") or {}
        slow = m.get("network_congestion_slow_share", 0)
        trip = m.get("mean_trip_time_min", 0)
        if slow >= 0.005 and trip <= 1.25 * base_trip:
            n_star = k
            break
    if n_star:
        lines.append(f"- **N* = {n_star}**（首个 slow-passage 占比 ≥0.5% 且行程时间 ≤1.25×baseline 的规模）。\n")
    else:
        lines.append("- 扫描范围内未达到「轻度到中度拥堵」判据：baseline 仍接近 free-flow。"
                     "**不建议在无拥堵下进入 Phase C 网络级结论**——可选：提高每人出行次数、"
                     "降低容量标定，或向用户报告并请求决策。\n")
    lines.append(f"\n*基准判定指标：慢行占比（>自由流+15s）、mean trip time、car VKT、拥堵分布（链路平均延误分位数）。*\n")
    (out / "scale_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out / 'scale_report.md'}")
    return 0


def _fmt_share(d: dict) -> str:
    if not d:
        return "—"
    return "/".join(f"{d.get(m, 0):.2f}" for m in ("bike", "car", "pt", "walk"))


def _fmt_link_shares(m: dict | None) -> str:
    if not m:
        return "—"
    return f"{m.get('share_links_delay_gt_15s', 0):.3f}/{m.get('share_links_delay_gt_30s', 0):.3f}/{m.get('share_links_delay_gt_60s', 0):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
