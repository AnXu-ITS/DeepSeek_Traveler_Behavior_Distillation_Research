#!/usr/bin/env python
"""Reference pipeline correctness validation (docs/REFERENCE_PIPELINE_VALIDATION.md).

Modes:
  fixture      dev-seed population (N=200): original production path vs
               reference path on the SAME input — decision parity, probability
               deltas, population.xml / config.xml byte equality, manifest
               equality. Plus batch vs per-state decide() on the same states.
  frozen-c0    READ-ONLY regression on the frozen Phase C benchmark: reference
               pipeline rebuilds the C0 population (N=10,000, seed 2026) and
               compares every decision field against the frozen manifest
               outputs/singapore_phase_c_s9/C0_baseline/adapter_manifest.json.
               Frozen inputs/results are never written (DATA_ISOLATION_CHECK).
  frozen-helsinki-c0
               Same regression on the second city: reference pipeline rebuilds
               E5 Helsinki C0 (N=10,000, seed 2026, data/helsinki/transit/) and
               compares against outputs/e5_helsinki/C0_baseline/
               adapter_manifest.json. The original-pipeline build time
               (9,507 s) is read from the frozen e5_result.json — the original
               pipeline is NOT re-run.

Usage:
    python scripts/reference/validate_reference_pipeline.py fixture
    python scripts/reference/validate_reference_pipeline.py frozen-c0
    python scripts/reference/validate_reference_pipeline.py frozen-helsinki-c0
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # reference_pipeline package
sys.path.insert(0, str(ROOT / "src"))  # production package

from traveler_distillation.accessibility.accessibility_dataset import build_real_alternatives  # noqa: E402
from traveler_distillation.accessibility.gtfs_accessibility import SupplyIndex, plan_accessibility  # noqa: E402
from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import PersonaGenerator, TripGenerator  # noqa: E402
from traveler_distillation.matsim.s8_adapter import S8MATSimAdapter  # noqa: E402
from traveler_distillation.schemas.context import DynamicContext, Weather  # noqa: E402

from reference_pipeline.feature_adapter import MemoizedEncoder, make_alt_factory  # noqa: E402
from reference_pipeline.matsim_adapter import build_scenario, build_supply_view  # noqa: E402
from reference_pipeline.route_cache import RouteCache  # noqa: E402
from reference_pipeline.student_adapter import StudentAdapter  # noqa: E402

SUPPLY = {
    "network": str((ROOT / "data/singapore/transit/network_with_transit.xml").resolve()),
    "schedule": str((ROOT / "data/singapore/transit/transitSchedule.xml").resolve()),
    "vehicles": str((ROOT / "data/singapore/transit/transitVehicles.xml").resolve()),
    "stops": str((ROOT / "data/singapore/transit/prep_stops.jsonl").resolve()),
    "snapping": str((ROOT / "data/singapore/transit/stop_snapping_report.json").resolve()),
    "trips_by_stop": str((ROOT / "data/singapore/transit/trips_by_stop.json").resolve()),
    "activity_nodes": str((ROOT / "data/singapore/transit/activity_nodes.json").resolve()),
}
# Helsinki zero-shot supply (E5, built with the same singapore/ toolchain)
SUPPLY_HEL = {
    "network": str((ROOT / "data/helsinki/transit/network_with_transit.xml").resolve()),
    "schedule": str((ROOT / "data/helsinki/transit/transitSchedule.xml").resolve()),
    "vehicles": str((ROOT / "data/helsinki/transit/transitVehicles.xml").resolve()),
    "stops": str((ROOT / "data/helsinki/transit/prep_stops.jsonl").resolve()),
    "snapping": str((ROOT / "data/helsinki/transit/stop_snapping_report.json").resolve()),
    "trips_by_stop": str((ROOT / "data/helsinki/transit/trips_by_stop.json").resolve()),
    "activity_nodes": str((ROOT / "data/helsinki/transit/activity_nodes.json").resolve()),
}
CHECKPOINT = ROOT / "releases/s9_supply_aware_v2/checkpoint/model.pt"
DEV_OUT = ROOT / "outputs/reference_pipeline/validation"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def c0_context() -> DynamicContext:
    return DynamicContext(
        context_id="C_SG_PHASEC_C0_BASELINE",
        weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.3, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )


# ---------------------------------------------------------------------------
# ORIGINAL production path — verbatim copy of run_phase_c.make_shared_idx /
# make_shared_factory (lines 146-187): one shared SupplyIndex + in-process
# per-OD accessibility cache + memoized mode travel times.
# ---------------------------------------------------------------------------
def production_shared_idx(supply: dict):
    idx = SupplyIndex(supply)
    _orig_mtt = idx.mode_travel_time
    _tt_cache: dict = {}
    _SENTINEL = object()

    def _mtt(mode, src, dst):
        key = (mode, src, dst)
        val = _tt_cache.get(key, _SENTINEL)
        if val is _SENTINEL:
            val = _orig_mtt(mode, src, dst)
            _tt_cache[key] = val
        return val

    idx.mode_travel_time = _mtt
    return idx


def production_factory(idx, context: DynamicContext):
    delay = context.transit_delay_min
    disrupt = context.road_disruption
    acc_cache: dict = {}

    def factory(persona, trip, ctx, origin, dest):
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
        return alts

    return factory


def population(seed: int, n: int):
    gen_cfg = load_yaml(ROOT / "configs" / "generation_v0_1.yaml")
    personas = PersonaGenerator(seed=seed, config=gen_cfg).generate(n)
    trips = TripGenerator(seed=seed, config=gen_cfg).generate(n)
    return personas, trips, [[t] for t in trips]


def compare_manifests(ma: list[dict], mb: list[dict]) -> dict:
    assert len(ma) == len(mb), f"manifest length {len(ma)} != {len(mb)}"
    n = len(ma)
    mode_match = sum(1 for a, b in zip(ma, mb) if a["student_mode"] == b["student_mode"])
    outbound_match = sum(1 for a, b in zip(ma, mb) if a["outbound_mode"] == b["outbound_mode"])
    return_match = sum(1 for a, b in zip(ma, mb) if a["return_mode"] == b["return_mode"])
    dep_match = sum(1 for a, b in zip(ma, mb)
                    if a["departure_min"] == b["departure_min"]
                    and a["departure_shift_min"] == b["departure_shift_min"])
    od_match = sum(1 for a, b in zip(ma, mb)
                   if a["home_node"] == b["home_node"] and a["dest_node"] == b["dest_node"])
    info_match = sum(1 for a, b in zip(ma, mb)
                     if a["outbound"] == b["outbound"] and a["return"] == b["return"])
    return {
        "n_decisions": n,
        "student_mode_exact_match": f"{mode_match}/{n}",
        "outbound_mode_exact_match": f"{outbound_match}/{n}",
        "return_mode_exact_match": f"{return_match}/{n}",
        "departure_fields_exact_match": f"{dep_match}/{n}",
        "od_exact_match": f"{od_match}/{n}",
        "leg_info_exact_match": f"{info_match}/{n}",
        "all_fields_identical": mode_match == n and outbound_match == n
        and return_match == n and dep_match == n and od_match == n and info_match == n,
    }


# ---------------------------------------------------------------------------
def mode_fixture() -> int:
    """N=200 dev-seed fixture: production path vs reference path."""
    out = DEV_OUT / "fixture"
    (out / "original").mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(parents=True, exist_ok=True)

    personas, trips, trips_per_persona = population(seed=0, n=200)
    context = c0_context()
    report = {"n": 200, "seed": 0, "checkpoint": str(CHECKPOINT)}

    # ---- Path A: ORIGINAL production (Phase C code path) ----
    t0 = time.perf_counter()
    adapter_a = S8MATSimAdapter(CHECKPOINT)
    idx_a = production_shared_idx(SUPPLY)
    factory_a = production_factory(idx_a, context)
    manifest_a = adapter_a.build_real_scenario(
        personas, trips, context, out / "original",
        network_path=SUPPLY["network"], schedule_path=SUPPLY["schedule"],
        vehicles_path=SUPPLY["vehicles"], stops_path=SUPPLY["stops"],
        snap_report_path=SUPPLY["snapping"], trips_by_stop_path=SUPPLY["trips_by_stop"],
        activity_nodes_path=SUPPLY["activity_nodes"],
        trips_per_persona=trips_per_persona,
        flow_capacity_factor=0.3, storage_capacity_factor=0.3,
        alt_factory=factory_a,
    )
    report["original_build_s"] = round(time.perf_counter() - t0, 1)

    # ---- Path B: REFERENCE (batch inference + persistent cache, cold) ----
    t0 = time.perf_counter()
    student = StudentAdapter(CHECKPOINT)
    supply_view = build_supply_view(SUPPLY["network"], SUPPLY["activity_nodes"])
    cache = RouteCache(out / "reference" / "cache" / "route_cache.pkl", SUPPLY,
                       reuse=True, rebuild=True)
    encoder = MemoizedEncoder(student.extractor)  # the real pipeline path
    result = build_scenario(
        student, personas, trips_per_persona, context, out / "reference", SUPPLY,
        supply_view, batch_size=256, cache=cache,
        flow_capacity_factor=0.3, storage_capacity_factor=0.3, encoder=encoder,
    )
    cache.flush()
    manifest_b = result["manifest"]
    decisions_b = result["decisions"]
    report["reference_build_s"] = round(time.perf_counter() - t0, 1)

    # ---- decision parity (reference batch vs original per-state) ----
    report["manifest_parity"] = compare_manifests(manifest_a, manifest_b)

    # probability / shift deltas (Path A manifest has no probs; use
    # per-state decide() on Path B states as the reference for the batch path)
    shift_deltas, prob_deltas = [], []
    # rebuild states deterministically the same way the reference build did
    from traveler_distillation.schemas.state import UniversalTravelerState
    from reference_pipeline.feature_adapter import derive_od
    idx_b = SupplyIndex(SUPPLY)
    factory_b = make_alt_factory(idx_b, context, cache=None)
    g_nodes = supply_view[3]
    states = []
    for persona, person_trips in zip(personas, trips_per_persona):
        for t_idx, trip in enumerate(person_trips):
            home, dest = derive_od(persona.persona_id, t_idx, g_nodes)
            alts = factory_b(persona, trip, context, home, dest)
            states.append(UniversalTravelerState(persona=persona, trip=trip,
                                                 context=context, alternatives=alts))
    single = [student.decide(s) for s in states]
    mode_match = sum(1 for a, b in zip(single, decisions_b) if a["mode"] == b["mode"])
    for a, b in zip(single, decisions_b):
        shift_deltas.append(abs(a["departure_time_shift_min"] - b["departure_time_shift_min"]))
        for m, p in a["mode_probabilities"].items():
            prob_deltas.append(abs(p - b["mode_probabilities"].get(m, 0.0)))
    report["batch_vs_single"] = {
        "mode_match": f"{mode_match}/{len(states)}",
        "max_abs_shift_delta": max(shift_deltas),
        "max_abs_prob_delta": max(prob_deltas),
    }

    # memoized encoder parity (static-part precompute vs production encode)
    memo_ok = sum(1 for s in states
                  if encoder.encode(s) == student.extractor.encode(s))
    report["memoized_encoder_parity"] = f"{memo_ok}/{len(states)}"

    # ---- artifact byte equality ----
    pop_equal = sha256(out / "original" / "population.xml") == sha256(out / "reference" / "population.xml")
    cfg_equal = sha256(out / "original" / "config.xml") == sha256(out / "reference" / "config.xml")
    man_equal = (json.loads((out / "original" / "adapter_manifest.json").read_text(encoding="utf-8"))
                 == json.loads((out / "reference" / "adapter_manifest.json").read_text(encoding="utf-8")))
    report["artifact_equality"] = {
        "population_xml_sha256_equal": pop_equal,
        "config_xml_sha256_equal": cfg_equal,
        "adapter_manifest_json_equal": man_equal,
        "original_population_sha256": sha256(out / "original" / "population.xml")[:16] + "…",
        "reference_population_sha256": sha256(out / "reference" / "population.xml")[:16] + "…",
    }
    report["cache_summary"] = cache.summary()

    ok = (report["manifest_parity"]["all_fields_identical"] and pop_equal
          and cfg_equal and man_equal
          and report["batch_vs_single"]["mode_match"].startswith(f"{len(states)}/"))
    report["PASS"] = ok
    (out / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Shared 10k frozen-benchmark regression (Singapore Phase C C0 / Helsinki E5 C0)
# ---------------------------------------------------------------------------
FROZEN_FIELDS = ["persona_id", "trip_id", "student_mode", "outbound_mode",
                 "return_mode", "departure_shift_min", "departure_min",
                 "home_node", "dest_node"]


def frozen_regression(
    supply: dict,
    frozen_manifest_path: Path,
    out_dir: Path,
    report_name: str,
) -> dict:
    """Rebuild the frozen C0 population (seed 2026, N=10,000) through the
    reference pipeline and compare every decision field against the frozen
    original-pipeline manifest. READ-ONLY: frozen inputs/results are never
    written; all outputs go to ``out_dir`` (DATA_ISOLATION_CHECK)."""
    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))["decisions"]
    out_dir.mkdir(parents=True, exist_ok=True)
    personas, trips, trips_per_persona = population(seed=2026, n=10000)
    context = c0_context()

    t0 = time.perf_counter()
    student = StudentAdapter(CHECKPOINT)
    supply_view = build_supply_view(supply["network"], supply["activity_nodes"])
    cache = RouteCache(out_dir / "cache" / "route_cache.pkl", supply, reuse=True, rebuild=False)
    result = build_scenario(
        student, personas, trips_per_persona, context, out_dir, supply, supply_view,
        batch_size=256, cache=cache, flow_capacity_factor=0.3,
        storage_capacity_factor=0.3,
    )
    cache.flush()
    build_s = time.perf_counter() - t0
    manifest_b = result["manifest"]

    assert len(manifest_b) == len(frozen), f"{len(manifest_b)} vs {len(frozen)}"
    n = len(manifest_b)
    matches = {f: sum(1 for a, b in zip(manifest_b, frozen) if a[f] == b[f])
               for f in FROZEN_FIELDS}

    # departure-shift disclosure: any row-level difference must be exactly the
    # 2-dp ROUNDING-BOUNDARY flip (0.01 min = 0.6 s). The raw shift noise is
    # <= ~4e-6 min (fixture D2), so a 0.01-min flip can only occur when the
    # raw value sits within noise of the 2-dp boundary.
    boundary_rows = [
        (i, a["persona_id"], a["trip_id"],
         a["departure_shift_min"], b["departure_shift_min"])
        for i, (a, b) in enumerate(zip(frozen, manifest_b))
        if a["departure_shift_min"] != b["departure_shift_min"]
    ]
    boundary_ok = all(abs(abs(a - b) - 0.01) < 1e-9 for _, _, _, a, b in boundary_rows)
    behavioral_identical = all(
        matches[f] == n for f in ("persona_id", "trip_id", "student_mode",
                                  "outbound_mode", "return_mode",
                                  "home_node", "dest_node")
    )
    report = {
        "n_decisions": n,
        "frozen_manifest": str(frozen_manifest_path),
        "reference_build_s": round(build_s, 1),
        "cache_loaded_from_disk": cache.loaded,
        "field_matches": matches,
        "behavioral_fields_identical": behavioral_identical,
        "departure_shift_boundary_rows": boundary_rows,
        "boundary_rows_are_0_01_rounding_flips": boundary_ok,
        "max_abs_departure_min_delta": max(
            (abs(a["departure_min"] - b["departure_min"]) for a, b in zip(frozen, manifest_b)),
            default=0.0,
        ),
        "mode_counts_reference": dict(Counter(m["student_mode"] for m in manifest_b)),
        "cache_summary": cache.summary(),
    }
    report["PASS"] = behavioral_identical and boundary_ok and len(boundary_rows) <= 5
    (out_dir / report_name).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def mode_frozen_c0() -> int:
    """Read-only 10k regression vs the frozen Phase C C0 manifest (Singapore)."""
    frozen_manifest_path = ROOT / "outputs/singapore_phase_c_s9/C0_baseline/adapter_manifest.json"
    if not frozen_manifest_path.exists():
        print("frozen C0 manifest missing — run fixture mode first", file=sys.stderr)
        return 2
    report = frozen_regression(SUPPLY, frozen_manifest_path,
                               DEV_OUT / "frozen_c0", "frozen_c0_regression.json")
    return 0 if report["PASS"] else 1


def mode_frozen_helsinki_c0() -> int:
    """Read-only 10k regression vs the frozen E5 Helsinki C0 manifest
    (second city, same toolchain — cross-city generalization evidence).

    The original-pipeline baseline build time (9,507 s) is the frozen value
    recorded in outputs/e5_helsinki/C0_baseline/e5_result.json — the original
    pipeline is NOT re-run."""
    frozen_manifest_path = ROOT / "outputs/e5_helsinki/C0_baseline/adapter_manifest.json"
    if not frozen_manifest_path.exists():
        print("frozen Helsinki C0 manifest missing", file=sys.stderr)
        return 2
    baseline_path = ROOT / "outputs/e5_helsinki/C0_baseline/e5_result.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    report = frozen_regression(SUPPLY_HEL, frozen_manifest_path,
                               DEV_OUT / "helsinki_c0", "frozen_helsinki_c0_regression.json")
    report["city"] = "Helsinki"
    report["original_pipeline_build_s"] = float(baseline["build_seconds"])
    report["original_pipeline_factory_timing_ms"] = baseline["factory_timing"]["mean_ms"]
    (DEV_OUT / "helsinki_c0" / "frozen_helsinki_c0_regression.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fixture"
    if mode == "fixture":
        raise SystemExit(mode_fixture())
    if mode == "frozen-c0":
        raise SystemExit(mode_frozen_c0())
    if mode == "frozen-helsinki-c0":
        raise SystemExit(mode_frozen_helsinki_c0())
    raise SystemExit(f"unknown mode {mode!r}; choose fixture | frozen-c0 | frozen-helsinki-c0")
