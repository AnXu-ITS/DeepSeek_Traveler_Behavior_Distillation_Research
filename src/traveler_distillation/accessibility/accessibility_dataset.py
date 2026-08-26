"""S8 — stratified real-supply accessibility dataset builder.

Builds the S8 training/evaluation dataset from the Singapore OSM + GTFS supply:

- OD pool stratified into accessibility Classes A-E (S8 instructions §9);
- three frozen holds (S8 instructions §14):
    Split A persona holdout  — S7/S3-convention 28/6/6 personas;
    Split B OD holdout       — train/val/test ODs completely disjoint;
    Split C accessibility    — high walking-burden feasible profiles
                               (access+egress >= 15 min) only in test;
- counterfactual accessibility curves: each (persona, trip) group spans several
  classes (§11), so accessibility sensitivity can be measured within-group;
- NO location identity anywhere in student/teacher-facing state: origin/dest
  stay as anonymous ``od_index`` entries in a separate routing-provenance file.

All records are city-independent: the state carries only persona, trip,
dynamic context and numeric alternative attributes (S8 instructions §3).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..generators import PersonaGenerator, TripGenerator
from ..schemas.alternative import TravelAlternative
from ..schemas.context import DynamicContext, Weather
from ..schemas.state import UniversalTravelerState
from ..schemas.trip import Trip
from .gtfs_accessibility import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    CLASS_D,
    CLASS_E,
    INFEASIBLE_TRAVEL_TIME_MIN,
    SupplyIndex,
    classify_accessibility,
    plan_accessibility,
)

ALL_CLASSES = [CLASS_A, CLASS_B, CLASS_C, CLASS_D, CLASS_E]
CURVE_CLASSES = [CLASS_A, CLASS_C, CLASS_D, CLASS_E]  # default per-group class set
WITH_B_FRACTION = 0.25  # share of groups that additionally get a Class B OD
PERSONA_SPLIT = {
    "train": ["P000004", "P000005", "P000010", "P000011", "P000012", "P000013",
              "P000014", "P000017", "P000019", "P000020", "P000021", "P000022",
              "P000023", "P000024", "P000025", "P000026", "P000027", "P000029",
              "P000030", "P000031", "P000032", "P000033", "P000034", "P000035",
              "P000036", "P000037", "P000038", "P000040"],
    "val": ["P000001", "P000003", "P000006", "P000007", "P000028", "P000039"],
    "test": ["P000002", "P000008", "P000009", "P000015", "P000016", "P000018"],
}


def make_context() -> DynamicContext:
    return DynamicContext(
        context_id="S8_BASELINE",
        weather=Weather(condition="clear", intensity=0.0),
        road_congestion=0.2, transit_delay_min=0, transit_disruption=False,
        road_disruption=False, fare_multiplier=1.0, parking_cost_multiplier=1.0,
        congestion_charge=0.0,
    )


def build_real_alternatives(
    persona, trip: Trip, context: DynamicContext,
    idx: SupplyIndex, origin: str, dest: str, acc: dict,
) -> list[TravelAlternative]:
    """Real-network alternatives: car/bike/walk from the OSM graph, pt from the
    real GTFS itinerary. Synthetic conventions (costs, exposure, congestion
    effect) mirror AlternativeGenerator so S7 semantics are preserved."""
    d = trip.distance_km
    cong = context.road_congestion
    weather_int = context.weather.intensity

    def _base(mode: str, tt_min: float) -> dict:
        out = {
            "available": True,
            "travel_time_min": round(tt_min, 3),
            "monetary_cost": 0.0,
            "access_time_min": 0.0,
            "transfers": 0,
            "reliability_delay_min": 0.0,
            "weather_exposure": 0.0,
        }
        return out

    alts = []
    # --- car ---
    car_tt = idx.mode_travel_time("car", origin, dest)
    car = _base("car", (car_tt if car_tt is not None else d / 32.0 * 60.0))
    car["available"] = bool(persona.driving_license and persona.car_ownership)
    car["travel_time_min"] = round(car["travel_time_min"] * (1.0 + cong * 0.5), 3)
    car["reliability_delay_min"] = round(cong * 15.0, 3)
    car["monetary_cost"] = round(d * 0.6 + 5.0 * context.parking_cost_multiplier
                                 + context.congestion_charge, 3)
    car["weather_exposure"] = 0.05
    car["travel_time_min"] = round(car["travel_time_min"] * (1.0 + weather_int * 0.15 * 0.05), 3)
    alts.append(TravelAlternative(mode="car", **car))

    # --- pt (real GTFS accessibility; stays in the choice set even if infeasible) ---
    pt = _base("pt", INFEASIBLE_TRAVEL_TIME_MIN)
    pt["available"] = True  # FVR is learnable behavior, not a mask (S8 §21)
    pt["weather_exposure"] = 0.4
    pt["monetary_cost"] = round((2.0 + d * 0.15) * context.fare_multiplier, 3)
    if acc["pt_feasible"] == 1.0:
        pt["travel_time_min"] = acc["door_to_door_min"]
        pt["access_time_min"] = acc["access_time_min"]
        pt["transfers"] = acc["transfer_count"]
        pt["pt_feasible"] = 1.0
        pt["egress_time_min"] = acc["egress_time_min"]
        pt["wait_time_min"] = acc["wait_time_min"]
        pt["in_vehicle_time_min"] = acc["in_vehicle_time_min"]
        pt["transfer_time_min"] = acc["transfer_time_min"]
        pt["coverage_ratio"] = acc["coverage_ratio"]
    else:
        pt["pt_feasible"] = 0.0
    alts.append(TravelAlternative(mode="pt", **pt))

    # --- bike ---
    bike_tt = idx.mode_travel_time("bike", origin, dest)
    bike = _base("bike", (bike_tt if bike_tt is not None else d / 14.0 * 60.0))
    bike["available"] = bool(persona.bike_ownership)
    bike["weather_exposure"] = 0.9
    bike["travel_time_min"] = round(bike["travel_time_min"] * (1.0 + weather_int * 0.15 * 0.9), 3)
    alts.append(TravelAlternative(mode="bike", **bike))

    # --- walk ---
    walk_tt = idx.mode_travel_time("walk", origin, dest)
    walk = _base("walk", (walk_tt if walk_tt is not None else d / 4.5 * 60.0))
    walk["available"] = True
    walk["weather_exposure"] = 1.0
    walk["travel_time_min"] = round(walk["travel_time_min"] * (1.0 + weather_int * 0.15), 3)
    alts.append(TravelAlternative(mode="walk", **walk))
    return alts


# ----------------------------------------------------------------------------
# candidate OD sampling + stratification
# ----------------------------------------------------------------------------
def _probe(idx: SupplyIndex, origin: str, dest: str, dep_sec: float) -> dict:
    return plan_accessibility(idx, origin, dest, dep_sec)


def _od_pair(rng: np.random.Generator, idx: SupplyIndex, from_band: str | None = None):
    nodes = (idx.band_nodes[from_band] if from_band else idx.candidate_nodes)
    i = int(rng.integers(0, len(nodes)))
    j = int(rng.integers(0, len(idx.candidate_nodes)))
    return nodes[i], idx.candidate_nodes[j]


def sample_od_candidates(
    idx: SupplyIndex, n_random: int, n_near_stops: int, n_mid_access: int,
    n_no_stops: int, dep_sec: float, seed: int = 0,
) -> list[dict]:
    """Candidate OD pairs across the accessibility spectrum:
    - random: uniform node pairs;
    - near_stops: both ends within 300 m of a stop (Class A/B pool);
    - mid_access: origin 300-700 m from the nearest stop (Class C/D pool);
    - no_stops: origin > 700 m from any stop (Class E pool)."""
    rng = np.random.default_rng(seed)
    ods: dict[tuple[str, str], dict] = {}

    def _add(origin: str, dest: str, tag: str):
        key = (origin, dest)
        if key in ods or origin == dest:
            return
        x0, y0 = idx.xy[origin]
        x1, y1 = idx.xy[dest]
        od_km = math.hypot(x0 - x1, y0 - y1) / 1000.0
        acc = _probe(idx, origin, dest, dep_sec)
        ods[key] = {
            "origin": origin, "dest": dest, "tag": tag,
            "od_km": round(od_km, 3), "dep_probe_sec": dep_sec,
            "accessibility": acc,
            "class": classify_accessibility(acc),
        }

    for _ in range(n_random):
        o, d = _od_pair(rng, idx)
        _add(o, d, "random")

    for _ in range(n_near_stops):
        o, d = _od_pair(rng, idx, from_band="near")
        _add(o, d, "near_stops")

    for _ in range(n_mid_access):
        o, d = _od_pair(rng, idx, from_band="mid")
        _add(o, d, "mid_access")

    for _ in range(n_no_stops):
        o, d = _od_pair(rng, idx, from_band="far")
        _add(o, d, "no_stops")
    return list(ods.values())


def build_od_pools(candidates: list[dict], rng: np.random.Generator,
                   class_targets: dict[str, int] | None = None,
                   train_frac: float = 0.7) -> dict:
    """Stratified OD pools with the three holds:
    - per class: train/val/test splits over DISJOINT OD sets (OD holdout),
    - 2-transfer feasible ODs excluded from train/val pools (accessibility holdout),
    - Class E always present in every split (FVR must be learnable AND testable)."""
    targets = class_targets or {
        CLASS_A: 24, CLASS_B: 12, CLASS_C: 28, CLASS_D: 28, CLASS_E: 32,
    }
    by_class = defaultdict(list)
    for c in candidates:
        by_class[c["class"]].append(c)
    for cls, rows in by_class.items():
        rng.shuffle(rows)
    pools: dict[str, dict[str, list[dict]]] = {split: defaultdict(list) for split in ("train", "val", "test")}
    for cls in ALL_CLASSES:
        rows = by_class[cls]
        # accessibility holdout: high walking-burden feasible profiles
        # (access + egress >= 15 min) appear only in the test pool; the planner
        # caps at 1 transfer (routing rule), so the holdout is defined on the
        # continuous walking-burden axis instead.
        heavy = [c for c in rows
                 if c["class"] != CLASS_E
                 and (c["accessibility"]["access_time_min"] + c["accessibility"]["egress_time_min"]) >= 15.0]
        light = [c for c in rows if c not in heavy]
        target = targets[cls]
        n_train = min(len(light), int(target * train_frac))
        n_val = min(len(light) - n_train, int(target * 0.15))
        n_test = int(target * 0.15)
        pools["train"][cls] = light[:n_train]
        pools["val"][cls] = light[n_train:n_train + n_val]
        pools["test"][cls] = (light[n_train + n_val:n_train + n_val + n_test]
                              + heavy[:n_test])
    return pools


# ----------------------------------------------------------------------------
# group assignment + record building
# ----------------------------------------------------------------------------
def build_groups(personas_by_split: dict[str, list], trips_by_persona: dict[str, list[dict]],
                 pools: dict, rng: np.random.Generator) -> list[dict]:
    """Assign each (persona, trip) group a class-spread set of ODs from its split pool."""
    groups = []
    for split, personas in personas_by_split.items():
        for persona in personas:
            pid = persona.persona_id
            for trip in trips_by_persona[pid]:
                classes = list(CURVE_CLASSES)
                if rng.random() < WITH_B_FRACTION:
                    classes.insert(1, CLASS_B)
                od_assign = []
                for cls in classes:
                    pool = pools[split][cls]
                    if not pool:
                        continue
                    od = dict(pool[int(rng.integers(0, len(pool)))])
                    od["split"] = split
                    od_assign.append(od)
                groups.append({
                    "split": split, "persona": persona, "trip": trip,
                    "od_assign": od_assign,
                    "group_id": f"{pid}::{trip.trip_id}",
                })
    return groups


def build_dataset(
    idx: SupplyIndex, groups: list[dict],
) -> tuple[list[dict], dict, list[dict]]:
    """Materialize records (states) for every group, at each trip's actual
    departure time. Returns (records, od_manifest, boundary_pairs)."""
    context = make_context()
    records: list[dict] = []
    od_index: dict[tuple[str, str], int] = {}
    od_manifest: list[dict] = []
    sample_seq = 0
    boundary_pairs = []
    n_classes = {cls: 0 for cls in ALL_CLASSES}

    def _od_key(od: dict) -> tuple[str, str]:
        return (od["origin"], od["dest"])

    for g in groups:
        persona = g["persona"]
        trip: Trip = g["trip"]
        dep_sec = trip.desired_departure_min * 60.0
        group_records = []
        for od in g["od_assign"]:
            key = _od_key(od)
            if key not in od_index:
                od_index[key] = len(od_manifest)
                od_manifest.append({
                    "od_index": len(od_manifest),
                    "origin": key[0], "dest": key[1],
                    "od_km": od["od_km"],
                    "accessibility_class_probe": od["class"],
                    "split": od["split"],
                })
            acc = plan_accessibility(idx, key[0], key[1], dep_sec)
            cls = classify_accessibility(acc)
            n_classes[cls] += 1
            trip_with_d = trip.model_copy(update={"distance_km": round(od["od_km"], 2)})
            alts = build_real_alternatives(persona, trip_with_d, context, idx, key[0], key[1], acc)
            state = UniversalTravelerState(persona=persona, trip=trip_with_d,
                                           context=context, alternatives=alts)
            sample_seq += 1
            rec = {
                "sample_id": f"S8_{sample_seq:06d}",
                "persona_id": persona.persona_id,
                "trip_id": trip.trip_id,
                "od_index": od_index[key],
                "accessibility_class": cls,
                "curve_group": g["group_id"],
                "split": g["split"],
                "departure_min": trip.desired_departure_min,
                "dep_sec": dep_sec,
                "accessibility": acc,
                "state": state.model_dump(),
            }
            records.append(rec)
            group_records.append(rec)

        # boundary pairs for K=5 (§13): good<->medium, medium<->poor,
        # poor<->infeasible transitions within the group curve
        by_class = defaultdict(list)
        for r in group_records:
            by_class[r["accessibility_class"]].append(r)
        order = [c for c in ALL_CLASSES if c in by_class]
        for c1, c2 in zip(order, order[1:]):
            if {c1, c2} in ({CLASS_B, CLASS_C}, {CLASS_C, CLASS_D}, {CLASS_D, CLASS_E}):
                boundary_pairs.append({"group": g["group_id"],
                                       "pair": [c1, c2],
                                       "samples": [by_class[c1][0]["sample_id"],
                                                   by_class[c2][0]["sample_id"]]})

    return records, od_manifest, boundary_pairs


def sanity_check(records: list[dict], od_manifest: list[dict],
                 personas_by_split: dict[str, list],
                 forbidden_strings: list[str] | None = None) -> dict:
    """Step 5 feature sanity checks. Returns a report dict; raises on violations."""
    report = {"n_states": len(records),
              "class_counts": dict(Counter(r["accessibility_class"] for r in records)),
              "split_counts": dict(Counter(r["split"] for r in records))}
    persona_split_sets = {s: {p.persona_id for p in ps} for s, ps in personas_by_split.items()}
    od_split_sets = defaultdict(set)
    for m in od_manifest:
        od_split_sets[m["split"]].add(m["od_index"])
    d2d_errs = []
    forbidden = forbidden_strings or []
    for r in records:
        acc = r["accessibility"]
        if acc["pt_feasible"] == 1.0:
            lhs = (acc["access_time_min"] + acc["egress_time_min"] + acc["wait_time_min"]
                   + acc["in_vehicle_time_min"] + acc["transfer_time_min"])
            d2d_errs.append(abs(lhs - acc["door_to_door_min"]))
        if not (0.0 <= acc["coverage_ratio"] <= 1.0):
            raise AssertionError(f"coverage out of range: {r['sample_id']}")
        if (acc["pt_feasible"] == 0.0) != (r["accessibility_class"] == CLASS_E):
            raise AssertionError(f"feasibility/class mismatch: {r['sample_id']}")
        st_json = json.dumps(r["state"], default=str)
        for bad in forbidden:
            # exact quoted-string match: ids are strings, so a leak appears as
            # `"<id>"` in JSON; substring matching on bare ids would false-positive
            # on numeric attribute values.
            if bad and f'"{bad}"' in st_json:
                raise AssertionError(f"location identity leak in {r['sample_id']}: {bad}")
    if d2d_errs:
        report["door_to_door_identity_max_err_min"] = round(max(d2d_errs), 4)
    if persona_split_sets["train"] & persona_split_sets["test"]:
        raise AssertionError("persona train/test overlap")
    if od_split_sets["train"] & od_split_sets["test"]:
        raise AssertionError("OD train/test overlap")
    report["od_pool_sizes"] = {s: len(v) for s, v in od_split_sets.items()}
    groups = defaultdict(set)
    for r in records:
        groups[r["curve_group"]].add(r["accessibility_class"])
    report["groups_with_lt3_classes"] = sum(1 for v in groups.values() if len(v) < 3)
    report["n_groups"] = len(groups)
    return report
