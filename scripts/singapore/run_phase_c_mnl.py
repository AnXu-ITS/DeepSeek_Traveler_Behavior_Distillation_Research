#!/usr/bin/env python
"""E1 S4 — Phase C simulation mirror with MNL-B decisions (C0-C5, N=10,000).

Reuses the frozen Phase C pipeline (same SCENARIOS, SUPPLY, shared accessibility
cache, capacity factors 0.3/0.3, population seed 2026, lastIteration=0) and
replaces ONLY the decision step with the frozen MNL-B coefficients
(outputs/e1_mnl/mnl_b_coefs.json). S9 is read-only (frozen Phase C results).

G3 default-path gate runs FIRST (build-only): C0 is rebuilt with
decision_fn=None and its population.xml + adapter_manifest.json + config.xml
must be byte-identical to the frozen outputs/singapore_phase_c_s9/C0_baseline/,
proving the decision_fn injection left the S9 path untouched.

Usage:
    python scripts/singapore/run_phase_c_mnl.py --scenarios C0_baseline,C1_heavy_rain \
        --output outputs/singapore_phase_c_mnl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase_c import (  # noqa: E402
    SCENARIOS,
    SUPPLY,
    _evaluate_gate,
    _manifest_stats,
    _parse_events_full,
    _run_matsim,
    make_context,
    make_shared_factory,
    make_shared_idx,
    sha256,
)
from traveler_distillation.baselines.mnl import MNLModel  # noqa: E402
from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import PersonaGenerator, TripGenerator  # noqa: E402
from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter  # noqa: E402
from traveler_distillation.singapore.scenario_metrics import load_link_attrs  # noqa: E402
from traveler_distillation.student.release_guard import assert_not_frozen_output  # noqa: E402

S9_CKPT = "releases/s9_supply_aware_v2/checkpoint/model.pt"
FROZEN_C0 = "outputs/singapore_phase_c_s9/C0_baseline"
MNLS_COEFS = "outputs/e1_mnl/mnl_b_coefs.json"


def run_g3_default_path_check(root: Path) -> dict:
    """Build C0 with decision_fn=None; compare artifacts byte-for-byte."""
    tmp = assert_not_frozen_output(root / "outputs" / "_g3_default_path_check")
    tmp.mkdir(parents=True, exist_ok=True)
    adapter = S8MATSimAdapter(root / S9_CKPT)
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}
    gen_cfg = load_yaml(root / "configs" / "generation_v0_1.yaml")
    personas = PersonaGenerator(seed=2026, config=gen_cfg).generate(10000)
    trips = TripGenerator(seed=2026, config=gen_cfg).generate(10000)
    trips_per_persona = [[t] for t in trips]
    idx = make_shared_idx(supply)
    factory = make_shared_factory(idx, make_context("C0_baseline"))
    adapter.build_real_scenario(
        personas, trips, make_context("C0_baseline"), tmp,
        network_path=supply["network"], schedule_path=supply["schedule"],
        vehicles_path=supply["vehicles"], stops_path=supply["stops"],
        snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
        activity_nodes_path=supply["activity_nodes"],
        trips_per_persona=trips_per_persona,
        flow_capacity_factor=0.3, storage_capacity_factor=0.3,
        alt_factory=factory, decision_fn=None,
    )
    frozen = root / FROZEN_C0
    checks = {}
    for fname in ("population.xml", "adapter_manifest.json", "config.xml"):
        fresh = (tmp / fname).read_bytes()
        ref = (frozen / fname).read_bytes()
        checks[fname] = {"identical": fresh == ref,
                         "fresh_sha256": sha256(tmp / fname),
                         "frozen_sha256": sha256(frozen / fname)}
    return {"pass": all(c["identical"] for c in checks.values()), "checks": checks,
            "check_dir": str(tmp)}


def run_scenario_mnl(name: str, n: int, personas, trips, trips_per_persona,
                     adapter: S8MATSimAdapter, supply: dict, root: Path,
                     out_root: Path, mnl: MNLModel, skip_matsim: bool, idx) -> dict:
    context = make_context(name)
    factory = make_shared_factory(idx, context)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    def decision_fn(state):
        return mnl.decide(state)

    import time
    t0 = time.time()
    manifest = adapter.build_real_scenario(
        personas, trips, context, out,
        network_path=supply["network"], schedule_path=supply["schedule"],
        vehicles_path=supply["vehicles"], stops_path=supply["stops"],
        snap_report_path=supply["snapping"], trips_by_stop_path=supply["trips_by_stop"],
        activity_nodes_path=supply["activity_nodes"],
        trips_per_persona=trips_per_persona,
        flow_capacity_factor=0.3, storage_capacity_factor=0.3,
        alt_factory=factory, decision_fn=decision_fn,
    )
    build_s = time.time() - t0
    stats = _manifest_stats(manifest)
    print(f"[{name}] built in {build_s:.0f}s; modes {stats['student_mode_counts']}", flush=True)

    record = {
        "scenario": name,
        "label": SCENARIOS[name]["label"],
        "decision_model": "MNL-B",
        "mnl_coefs": str(root / MNLS_COEFS),
        "mnl_coefs_sha256": sha256(root / MNLS_COEFS),
        "n_agents": n,
        "population_seed": 2026,
        "frozen_settings": {"flow_capacity_factor": 0.3, "storage_capacity_factor": 0.3,
                            "last_iteration": 0, "network": supply["network"],
                            "schedule": supply["schedule"]},
        "matsim": "MATSim 2026.0 (matsim_rel/matsim-2026.0.jar)", "java": "OpenJDK 25.0.4",
        "build_seconds": round(build_s, 1),
        "decisions": stats,
        "matsim_exit_code": None,
        "metrics": None,
        "runtime_seconds": None,
        "log_tail": "",
    }

    links = load_link_attrs(str(root / supply["network"]))
    if not skip_matsim:
        t1 = time.time()
        code, tail = _run_matsim(out)
        record["runtime_seconds"] = round(time.time() - t1, 1)
        record["matsim_exit_code"] = code
        record["log_tail"] = tail
        metrics = _parse_events_full(out, links)
        metrics["leg_departures"] = dict(metrics.pop("leg_departures"))
        record["metrics"] = metrics
        if code != 0:
            print(f"[{name}] MATSim exit={code}\nlog tail:\n{tail}")

    gate = _evaluate_gate(name, record)
    record["gate"] = gate
    (out / "phase_c_mnl_result.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] gate={'PASS' if gate['pass'] else 'FAIL'} ({gate['summary']})", flush=True)
    return record


def write_report(out_root: Path, records: list[dict]) -> None:
    c0 = next((r for r in records if r["scenario"] == "C0_baseline"), None)
    s9 = json.loads((out_root.parent / "singapore_phase_c_s9" / "phase_c_records.json").read_text(encoding="utf-8"))
    s9_by = {r["scenario"]: r for r in s9}
    lines = [
        "# Phase C mirror — MNL-B decisions (E1 S4)",
        "",
        f"Population: N* = {records[0]['n_agents']} (seed 2026, identical across scenarios); "
        "MNL-B (frozen coefficients, spec S3); capacity factors 0.3/0.3; supply unchanged; "
        "lastIteration=0; S9 column = frozen Phase C (read-only reference).",
        "",
        "## Gates",
        "",
        "| scenario | exit | stuck persons | stuck transit veh | pt board/alight | gate |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        m = r.get("metrics") or {}
        g = r["gate"]
        lines.append(f"| {r['scenario']} | {r.get('matsim_exit_code')} | {m.get('stuck_persons', '—')} "
                     f"| {m.get('stuck_transit_vehicles', '—')} | {m.get('pt_boardings', '—')}/{m.get('pt_alightings', '—')} | "
                     f"{'PASS' if g['pass'] else 'FAIL'} |")
    lines += ["", "## Decisions (student mode share)", "",
              "| scenario | car | pt | bike | walk | shift mean (min) |",
              "|---|---|---|---|---|---|"]
    for r in records:
        d = r["decisions"]
        n = r["n_agents"]
        cells = " | ".join(f"{100 * d['student_mode_counts'].get(m, 0) / max(1, n):.1f}%"
                           for m in ("car", "pt", "bike", "walk"))
        sh = d["departure_shift"]
        lines.append(f"| {r['scenario']} | {cells} | {sh['mean_min']} (MNL fixed 0 by design) |")
    lines += ["", "## System metrics", "",
              "| scenario | PT boardings | mean trip time (min) | car VKT (km) | waiting pt | failed trips |",
              "|---|---|---|---|---|---|"]
    for r in records:
        m = r.get("metrics") or {}
        lines.append(f"| {r['scenario']} | {m.get('pt_boardings', '—')} | {m.get('mean_trip_time_min', '—')} | "
                     f"{m.get('car_vkt_km', '—')} | {m.get('waiting_for_pt', '—')} | {m.get('failed_trips', '—')} |")
    if c0 is not None:
        lines += ["", "## C1–C5 vs C0 (paired; MNL-B vs frozen S9)", "",
                  "| scenario | MNL Δ pt share | S9 Δ pt share | MNL Δ car share | S9 Δ car share | "
                  "MNL Δ boardings | S9 Δ boardings | MNL Δ VKT | S9 Δ VKT |",
                  "|---|---|---|---|---|---|---|---|---|"]
        m0 = c0.get("metrics") or {}
        d0 = c0["decisions"]
        n = c0["n_agents"]
        for r in records:
            if r["scenario"] == "C0_baseline":
                continue
            m = r.get("metrics") or {}
            d = r["decisions"]
            s = s9_by.get(r["scenario"])
            s_m = s.get("metrics") or {}
            s_d = s["decisions"]
            s_n = s["n_agents"]
            def dshare(dd, mode, nn):
                return 100 * (dd["student_mode_counts"].get(mode, 0) / nn - d0["student_mode_counts"].get(mode, 0) / n)
            def dshare_s9(dd, mode, nn):
                return 100 * (dd["student_mode_counts"].get(mode, 0) / nn - s9_by["C0_baseline"]["decisions"]["student_mode_counts"].get(mode, 0) / s9_by["C0_baseline"]["n_agents"])
            def dm(a, b):
                return f"{a - b:+.0f}" if a is not None and b is not None else "—"
            lines.append(f"| {r['scenario']} | {dshare(d,'pt',n):+.1f}pp | {dshare_s9(s_d,'pt',s_n):+.1f}pp | "
                         f"{dshare(d,'car',n):+.1f}pp | {dshare_s9(s_d,'car',s_n):+.1f}pp | "
                         f"{dm(m.get('pt_boardings'), m0.get('pt_boardings'))} | {dm(s_m.get('pt_boardings'), s9_by['C0_baseline'].get('metrics',{}).get('pt_boardings'))} | "
                         f"{dm(m.get('car_vkt_km'), m0.get('car_vkt_km'))} | {dm(s_m.get('car_vkt_km'), s9_by['C0_baseline'].get('metrics',{}).get('car_vkt_km'))} |")
    lines += ["", "## Honest boundaries", "",
              "- MNL-B: mode choice only, departure shift = 0; coefficients frozen from outputs/e1_mnl/mnl_b_coefs.json (spec S3).",
              "- MNL-B was estimated on the same Teacher supervision as S9's supply-aware adaptation (234 train states); "
              "it is NOT calibrated to revealed-preference data.",
              "- The frozen Teacher labels are largely cost-insensitive (see E1 report T5): MNL-B's cost coefficient is "
              "positive (z=1.28), so its C2 (fare) response direction may be inverted — reported as measured, not fixed.",
              "- Perturbations are demand-side context injections; network and schedule identical across scenarios.",
              "- MRT schedules are frequency-based/synthetic (community GTFS feed, not official LTA DataMall).",
              "- ~20% of transit vehicles are truncated at the 30:00 simulation end (pre-existing 10k + 0.3-factor property); "
              "deltas use the identical setting and remain informative.",
              "",
              f"*Generated from {len(records)} scenario result JSONs in this directory.*",
              ""]
    (out_root / "PHASE_C_MNL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_root / "phase_c_mnl_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {out_root / 'PHASE_C_MNL_REPORT.md'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="C0_baseline,C1_heavy_rain,C2_fare_increase,"
                                           "C3_transit_delay,C4_road_disruption,C5_joint_rain_delay")
    ap.add_argument("--num-agents", type=int, default=10000)
    ap.add_argument("--output", default="outputs/singapore_phase_c_mnl")
    ap.add_argument("--skip-matsim", action="store_true")
    ap.add_argument("--skip-g3-check", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}")

    # G3 default-path gate (build-only, byte-compare vs frozen S9 C0)
    if not args.skip_g3_check:
        print("=== G3 default-path check (build-only C0 with decision_fn=None) ===", flush=True)
        g3 = run_g3_default_path_check(root)
        print(json.dumps(g3, ensure_ascii=False, indent=2)[:1500], flush=True)
        if not g3["pass"]:
            print("G3 FAIL — decision_fn injection changed the default path; aborting", flush=True)
            return 1
        print("G3 PASS", flush=True)

    coefs = json.loads((root / MNLS_COEFS).read_text(encoding="utf-8"))
    mnl = MNLModel.from_json(coefs)
    print(f"MNL-B spec {mnl.spec}, k={len(mnl.theta)}", flush=True)

    gen_cfg = load_yaml(root / "configs" / "generation_v0_1.yaml")
    n = args.num_agents
    print(f"generating population (N={n}, seed=2026)...", flush=True)
    personas = PersonaGenerator(seed=2026, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=2026, config=gen_cfg).generate(n)
    trips_per_persona = [[t] for t in trips]

    adapter = S8MATSimAdapter(root / S9_CKPT)
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}
    shared_idx = make_shared_idx(supply)

    out_root = assert_not_frozen_output(root / args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    records = []
    for name in names:
        print(f"=== scenario {name} ({SCENARIOS[name]['label']}) ===", flush=True)
        rec = run_scenario_mnl(name, n, personas, trips, trips_per_persona,
                               adapter, supply, root, out_root, mnl,
                               args.skip_matsim, shared_idx)
        records.append(rec)

    write_report(out_root, records)
    gates_ok = all(r["gate"]["pass"] for r in records)
    print("E1 PHASE C MNL GATE:", "ALL PASS" if gates_ok else "SOME FAIL — review result JSONs")
    return 0 if gates_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
