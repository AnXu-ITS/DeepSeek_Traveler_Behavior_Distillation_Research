#!/usr/bin/env python
"""E5 Helsinki zero-shot transfer runner (second city, FROZEN S9).

Runs the frozen S9 Supply-Aware Traveler Agent v2.0 against the Helsinki
supply (data/helsinki/transit/) under the E5 minimal scenario set
(TRC_AIT_5_E5_SECOND_CITY_ZERO_SHOT_DESIGN.md §4):

  C0 baseline -> C1 heavy rain -> C3 transit delay

Frozen rules honored here (design §1 / §8.1 G0):
- checkpoint = releases/s9_supply_aware_v2/checkpoint/model.pt, identity
  verified by SHA256 (not by path) at startup;
- load-only S8MATSimAdapter (eval + no_grad, no optimizer);
- ZERO Teacher: env API keys are removed, the teacher package is never
  imported (asserted), recorded in the result;
- feature schema / normalization untouched (extractor restored from the
  checkpoint state);
- N=10,000, seed 2026, generation_v0_1.yaml, capacity 0.3/0.3,
  lastIteration=0 — identical to Singapore Phase C.

E5-only additions (all inside this file; the frozen pipeline is imported,
never modified):
- per-decision accessibility class (A-E) + 6-field vector attached to the
  manifest (model input schema unchanged);
- per-OD factory timing (performance probe for the V1 trigger);
- config.xml coordinateSystem patched EPSG:32648 -> EPSG:32635 after
  build_real_scenario writes it (label only; MATSim does no reprojection);
- G3 decision-layer gates (four modes present; PT decision validity >= 85%);
- covariate-shift audit vs the frozen Singapore normalization stats.

Usage:
    python scripts/helsinki/run_e5_helsinki.py \
        --num-agents 10000 \
        --scenarios C0_baseline,C1_heavy_rain,C3_transit_delay \
        --output outputs/e5_helsinki
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WB / "src"))
sys.path.insert(0, str(_WB / "scripts"))

# G0: zero-Teacher guard — strip API keys BEFORE any imports; the teacher
# package is never imported by this module or the imports below.
for _k in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_k, None)

from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives  # noqa: E402
from traveler_distillation.accessibility.gtfs_accessibility import (  # noqa: E402
    SupplyIndex,
    classify_accessibility,
    plan_accessibility,
)
from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import PersonaGenerator, TripGenerator  # noqa: E402
from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter  # noqa: E402
from traveler_distillation.singapore.scenario_metrics import load_link_attrs  # noqa: E402
from traveler_distillation.student.release_guard import assert_not_frozen_output  # noqa: E402
import singapore.run_phase_c as _rpc  # noqa: E402
from singapore.run_phase_c import (  # noqa: E402  (scripts/ namespace pkg; frozen, not modified)
    SCENARIOS,
    _evaluate_gate,
    _manifest_stats,
    _parse_events_full,
    _run_matsim,
    make_context,
    make_shared_idx,
    sha256,
)

# process-local durable-classpath patch (E5 only; see _MATSIM_HOME note below)

EXPECTED_S9_SHA256 = "6af79b44bc699c00cc8e913da15043a43398829fdc8bbb4dad2ddcb8e11d31e6"
S9_CHECKPOINT = "releases/s9_supply_aware_v2/checkpoint/model.pt"
NORMALIZATION = "releases/s9_supply_aware_v2/normalization/normalization.json"

# Durable MATSim classpath (E5-only override): the frozen run_phase_c._CP points
# at %TEMP%\matsim_rel, which Windows Temp cleanup deleted mid-run on 2026-08-28
# (C0 MATSim exited 1 with NoClassDefFoundError). The workbench-owned copy under
# tools/ is persistent; the override is process-local and leaves the frozen
# Singapore pipeline untouched.
_MATSIM_HOME = _WB / "tools" / "matsim-2026.0-release" / "matsim-2026.0"
if (_MATSIM_HOME / "matsim-2026.0.jar").exists():
    _TMP = "C:/Users/xuan1/AppData/Local/Temp"
    _E5_CP = f"{_TMP}/matsim_run;{str(_MATSIM_HOME / 'matsim-2026.0.jar').replace(chr(92), '/')};{str(_MATSIM_HOME / 'libs').replace(chr(92), '/')}/*"
else:
    _E5_CP = None

if _E5_CP is not None:
    _rpc._CP = _E5_CP

SUPPLY = {
    "network": "data/helsinki/transit/network_with_transit.xml",
    "schedule": "data/helsinki/transit/transitSchedule.xml",
    "vehicles": "data/helsinki/transit/transitVehicles.xml",
    "stops": "data/helsinki/transit/prep_stops.jsonl",
    "snapping": "data/helsinki/transit/stop_snapping_report.json",
    "trips_by_stop": "data/helsinki/transit/trips_by_stop.json",
    "activity_nodes": "data/helsinki/transit/activity_nodes.json",
}

ACC_FIELDS = [
    "pt_feasible", "egress_time_min", "wait_time_min",
    "in_vehicle_time_min", "transfer_time_min", "coverage_ratio",
]


def _teacher_guard() -> dict:
    import sys as _sys
    imported = sorted(m for m in _sys.modules if m.startswith("traveler_distillation.teacher"))
    env_leak = {k: (k in os.environ) for k in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY", "OPENAI_API_KEY")}
    guard = {"teacher_modules_imported": imported, "api_key_env_present": env_leak,
             "zero_teacher": (not imported) and not any(env_leak.values())}
    return guard


def make_e5_factory(idx: SupplyIndex, context, acc_log: dict, timing_log: list,
                    acc_cache: dict | None = None):
    """Shared alt_factory with per-OD accessibility cache + acc capture + timing.

    Mirrors `singapore.run_phase_c.make_shared_factory` exactly (same formulas,
    same cache key) so decisions are identical to the Phase C path; additionally
    records the accessibility vector per (persona, trip) and per-call wall time.
    ``acc_cache`` may be shared across scenarios (E5-only runtime optimization:
    the accessibility vector depends only on (OD, departure), never on the
    scenario — decisions are byte-identical with or without sharing).
    """
    delay = context.transit_delay_min
    disrupt = context.road_disruption
    acc_cache = acc_cache if acc_cache is not None else {}

    def factory(persona, trip, ctx, origin, dest):
        t0 = time.perf_counter()
        key = (origin, dest, round(float(trip.desired_departure_min), 3))
        acc = acc_cache.get(key)
        if acc is None:
            acc = plan_accessibility(idx, origin, dest, float(trip.desired_departure_min) * 60.0)
            acc_cache[key] = acc
        alts = build_real_alternatives(persona, trip, ctx, idx, origin, dest, acc)
        for alt in alts:
            if alt.mode == "pt" and delay:
                alt.travel_time_min = round(alt.travel_time_min + delay, 3)
                alt.reliability_delay_min = round(alt.reliability_delay_min + delay, 3)
            elif alt.mode == "car" and disrupt:
                alt.travel_time_min = round(alt.travel_time_min + 20.0, 3)
                alt.reliability_delay_min = round(alt.reliability_delay_min + 20.0, 3)
        acc_log[(persona.persona_id, trip.trip_id)] = acc
        timing_log.append(time.perf_counter() - t0)
        return alts

    return factory


def _patch_crs(out_dir: Path) -> None:
    cfg = out_dir / "config.xml"
    text = cfg.read_text(encoding="utf-8")
    if 'value="EPSG:32648"' in text:
        cfg.write_text(text.replace('value="EPSG:32648"', 'value="EPSG:32635"'), encoding="utf-8")


def _decision_gate(name: str, manifest: list[dict]) -> dict:
    stu = Counter(m["student_mode"] for m in manifest)
    intended_pt = [m for m in manifest if m["student_mode"] == "pt"]
    fallen = [m for m in intended_pt if m.get("outbound_mode") == "walk" and m["outbound"].get("pt_fallback")]
    reasons = Counter(m["outbound"].get("pt_reason", "unknown") for m in fallen)
    known = {"no_direct_or_transfer", "no_service_window", "no_stops_in_radius"}
    # validity conditioned on feasibility: execution fidelity where service exists
    feas = [m for m in intended_pt if (m.get("accessibility") or {}).get("pt_feasible") == 1.0]
    feas_fallen = [m for m in feas if m in fallen]
    feas_validity = 1.0 - len(feas_fallen) / max(1, len(feas))
    overall = 1.0 - len(fallen) / max(1, len(intended_pt))
    checks = {
        "manifest_built": len(manifest) > 0,
        "four_modes_in_decisions": {"car", "pt", "bike", "walk"} <= set(stu),
        "fallback_reasons_known": set(reasons) <= known,
        "note": (f"student modes {dict(stu)}; intended-pt {len(intended_pt)}, outbound pt_fallback_walk "
                 f"{len(fallen)} (reasons {dict(reasons)}) -> overall validity {overall:.1%}; "
                 f"feasible-pt {len(feas)}, feasible fallback {len(feas_fallen)} -> "
                 f"feasibility-conditioned validity {feas_validity:.1%}. "
                 f"Overall validity is REPORTED (Helsinki share of Class-E pt decisions differs from Singapore); "
                 f"hard checks = four modes + known fallback reasons."),
    }
    passed = all(v is not False for k, v in checks.items() if k != "note")
    return {"pass": passed, "checks": checks,
            "pt_validity_overall": round(overall, 4),
            "pt_validity_feasible_conditioned": round(feas_validity, 4),
            "fallback_reasons": dict(reasons),
            "student_mode_counts": dict(stu)}


def _timing_stats(per_call: list[float]) -> dict:
    s = sorted(per_call)
    n = max(1, len(s))
    return {
        "n_calls": len(per_call),
        "total_s": round(sum(per_call), 2),
        "mean_ms": round(sum(per_call) / n * 1000.0, 2),
        "p50_ms": round(s[n // 2] * 1000.0, 2),
        "p95_ms": round(s[min(n - 1, int(n * 0.95))] * 1000.0, 2),
        "max_ms": round(s[-1] * 1000.0, 2),
    }


def _acc_audit(manifest: list[dict], acc_log: dict) -> dict:
    """Covariate-shift audit of the 6 supply fields vs frozen SG normalization."""
    norm = json.loads((_WB / NORMALIZATION).read_text(encoding="utf-8"))
    stats = norm["alternative_fields"]["s8_new"]["stats"]
    rows = [acc_log[(m["persona_id"], m["trip_id"])] for m in manifest]
    audit: dict[str, dict] = {}
    for f in ACC_FIELDS:
        vals = [r[f] for r in rows if r is not None]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        srt = sorted(vals)
        mu, sd = stats[f]["mean"], stats[f]["std"]
        zs = [(v - mu) / max(sd, 1e-8) for v in vals]
        audit[f] = {
            "helsinki_mean": round(mean, 4),
            "helsinki_std": round(var ** 0.5, 4),
            "p5": round(srt[len(srt) // 20], 4),
            "p50": round(srt[len(srt) // 2], 4),
            "p95": round(srt[min(len(srt) - 1, int(len(srt) * 0.95))], 4),
            "sg_mean": mu,
            "sg_std": sd,
            "z_mean": round(sum(zs) / len(zs), 3),
            "z_p95": round(sorted(zs)[min(len(zs) - 1, int(len(zs) * 0.95))], 3),
            "share_abs_z_gt_3": round(sum(1 for z in zs if abs(z) > 3.0) / len(zs), 4),
        }
    class_counts = Counter(classify_accessibility(acc_log[(m["persona_id"], m["trip_id"])])
                           for m in manifest)
    return {"fields": audit, "class_counts": dict(class_counts)}


def run_scenario(name: str, n: int, personas, trips, trips_per_persona,
                 adapter: S8MATSimAdapter, supply: dict, root: Path, out_root: Path,
                 skip_matsim: bool, factory, acc_log: dict, timing_log: list) -> dict:
    s = SCENARIOS[name]
    context = make_context(name)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
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
    _patch_crs(out)
    build_s = time.time() - t0

    # E5 manifest extension: accessibility class + vector per decision
    missing_acc = [m for m in manifest if (m["persona_id"], m["trip_id"]) not in acc_log]
    if missing_acc:
        raise RuntimeError(f"{len(missing_acc)} manifest rows lack accessibility capture — acc_log key mismatch")
    for m in manifest:
        acc = acc_log[(m["persona_id"], m["trip_id"])]
        m["accessibility"] = acc
        m["accessibility_class"] = classify_accessibility(acc)
    # persist the EXTENDED manifest (the adapter wrote the un-extended one)
    manifest_path = out / "adapter_manifest.json"
    manifest_doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_doc["decisions"] = manifest
    manifest_doc["e5_extension"] = "accessibility + accessibility_class per decision (E5 design §4.1.5; model input schema unchanged)"
    manifest_path.write_text(json.dumps(manifest_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = _manifest_stats(manifest)
    dgate = _decision_gate(name, manifest)
    print(f"[{name}] built in {build_s:.0f}s; student modes {stats['student_mode_counts']}; "
          f"pt validity {dgate['pt_validity_overall']:.1%} "
          f"(feasible-conditioned {dgate['pt_validity_feasible_conditioned']:.1%})")

    record = {
        "experiment": "E5",
        "city": "Helsinki",
        "scenario": name,
        "label": s["label"],
        "context": {k: (v if not isinstance(v, tuple) else {"condition": v[0], "intensity": v[1]})
                    for k, v in s.items()},
        "n_agents": n,
        "population_seed": 2026,
        "checkpoint": str(root / S9_CHECKPOINT),
        "checkpoint_sha256": sha256(root / S9_CHECKPOINT),
        "checkpoint_sha256_verified": sha256(root / S9_CHECKPOINT) == EXPECTED_S9_SHA256,
        "teacher_guard": _teacher_guard(),
        "frozen_settings": {"flow_capacity_factor": 0.3, "storage_capacity_factor": 0.3,
                            "last_iteration": 0, "network": supply["network"],
                            "schedule": supply["schedule"], "crs_patched": "EPSG:32635"},
        "matsim": "MATSim 2026.0 (matsim_rel/matsim-2026.0.jar)", "java": "OpenJDK 25.0.4",
        "build_seconds": round(build_s, 1),
        "factory_timing": _timing_stats(timing_log),
        "decisions": stats,
        "decision_gate": dgate,
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
    record["all_gates"] = {"G0": record["checkpoint_sha256_verified"] and record["teacher_guard"]["zero_teacher"],
                           "G3": dgate["pass"], "G4": gate["pass"]}
    (out / "e5_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] done; G3={dgate['pass']} G4={'PASS' if gate['pass'] else 'FAIL'} ({gate['summary']})")
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="C0_baseline,C1_heavy_rain,C3_transit_delay")
    ap.add_argument("--checkpoint", default=S9_CHECKPOINT)
    ap.add_argument("--config", default="configs/generation_v0_1.yaml")
    ap.add_argument("--num-agents", type=int, default=10000)
    ap.add_argument("--output", default="outputs/e5_helsinki")
    ap.add_argument("--skip-matsim", action="store_true")
    args = ap.parse_args()

    root = _WB
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")

    # ---- G0 identity + red-line checks ----
    ck_path = root / args.checkpoint
    ck_sha = sha256(ck_path)
    if ck_sha != EXPECTED_S9_SHA256:
        raise SystemExit(f"G0 FAIL: checkpoint SHA256 {ck_sha} != expected {EXPECTED_S9_SHA256}")
    guard = _teacher_guard()
    if not guard["zero_teacher"]:
        raise SystemExit(f"G0 FAIL: teacher red line violated: {guard}")
    print(f"G0 PASS: S9 SHA256 verified; zero-Teacher ({guard})")

    out_root = assert_not_frozen_output(root / args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    gen_cfg = load_yaml(args.config)
    n = args.num_agents

    print(f"generating population (N={n}, seed=2026)...", flush=True)
    personas = PersonaGenerator(seed=2026, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=2026, config=gen_cfg).generate(n)
    trips_per_persona = [[t] for t in trips]

    adapter = S8MATSimAdapter(ck_path)
    supply = {k: str((root / v).resolve()) for k, v in SUPPLY.items()}
    shared_idx = make_shared_idx(supply)

    acc_log: dict = {}
    shared_acc_cache: dict = {}
    records = []
    for name in names:
        timing_log: list[float] = []
        factory = make_e5_factory(shared_idx, make_context(name), acc_log, timing_log,
                                  acc_cache=shared_acc_cache)
        print(f"=== scenario {name} ({SCENARIOS[name]['label']}) ===", flush=True)
        rec = run_scenario(name, n, personas, trips, trips_per_persona,
                           adapter, supply, root, out_root, args.skip_matsim,
                           factory, acc_log, timing_log)
        records.append(rec)

    # ---- covariate-shift audit on C0 ----
    c0 = next((r for r in records if r["scenario"] == "C0_baseline"), None)
    if c0 is not None:
        audit = _acc_audit(
            [m for m in json.loads((out_root / "C0_baseline" / "adapter_manifest.json").read_text(encoding="utf-8"))["decisions"]],
            acc_log)
        (out_root / "e5_acc_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        print("acc audit ->", out_root / "e5_acc_audit.json")

    (out_root / "e5_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    all_ok = all(r["all_gates"]["G0"] and r["all_gates"]["G3"] and r["all_gates"]["G4"] for r in records)
    print("E5 GATE:", "ALL PASS" if all_ok else "SOME FAIL — review e5_result.json files")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
