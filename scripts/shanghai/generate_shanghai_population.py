#!/usr/bin/env python
"""Generate a synthetic Shanghai-representative sample (100 residents) and
distance-matched OD, for running through the frozen S9 supply-aware model.

Produces:
  * examples/sample_population_shanghai_100.csv          (base, no OD)
  * examples/sample_population_shanghai_100_dmatched.csv (with origin_node/dest_node)

Distributions are researcher-assumed Shanghai urban-core priors (Xuhui), NOT
yet calibrated to the survey. They deliberately differ from the Singapore
generation_v0_1.yaml priors: lower car ownership (plate auction), lower period
transit-pass take-up (pay-per-ride dominates), PT-heavy habitual mode.

Deterministic (seed fixed). One trip per resident (their main weekday trip).
"""
from __future__ import annotations

import csv
import hashlib
import random
import sys
from pathlib import Path

_WB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WB))                       # reference_pipeline pkg
sys.path.insert(0, str(_WB / "src"))               # traveler_distillation pkg

from traveler_distillation.schemas.persona import Persona  # noqa: E402
from traveler_distillation.schemas.trip import Trip  # noqa: E402

SEED = 2026
N = 100
ID_OFFSET = 900100                      # personas P900101..P900200, trips T900101..
BASE_CSV = _WB / "examples/sample_population_shanghai_100.csv"
DMATCH_CSV = _WB / "examples/sample_population_shanghai_100_dmatched.csv"
NETWORK = _WB / "data/shanghai/transit/network_with_transit.xml"
NODES = _WB / "data/shanghai/transit/activity_nodes.json"
HOME_SAMPLES = 150


def _w(rng: random.Random, dist: dict) -> str:
    items = list(dist.items())
    total = sum(w for _, w in items)
    r = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += w
        if r < acc:
            return k
    return items[-1][0]


def _r(rng: random.Random, lo: float, hi: float, nd: int = 1) -> float:
    return round(rng.uniform(lo, hi), nd)


# ---------------------------------------------------------------------------
# Shanghai urban-core priors (assumed; document as uncalibrated)
# ---------------------------------------------------------------------------
PERSONA_PRIORS = {
    "age_group": {"18-24": 0.10, "25-34": 0.25, "35-44": 0.22, "45-64": 0.28, "65+": 0.15},
    "income_group": {"low": 0.30, "medium": 0.45, "high": 0.25},
    "occupation": {"student": 0.08, "office_worker": 0.38, "service_worker": 0.18,
                   "manual_worker": 0.08, "retired": 0.14, "unemployed": 0.06, "other": 0.08},
    "car_ownership_prob": 0.42,      # Shanghai real value (household private car, ~42/100 households)
    "driving_license_prob": 0.55,
    "bike_ownership_prob": 0.10,      # Shanghai real value (conventional pedal bicycle; excl. e-bike/shared)
    "transit_pass_prob": 0.15,       # period passes uncommon; pay-per-ride dominates
    "habitual_mode": {"car": 0.12, "pt": 0.38, "bike": 0.15, "walk": 0.15, "mixed": 0.20},
    "schedule_flexibility": {"low": 0.45, "medium": 0.40, "high": 0.15},
    "mobility_limitation": {"none": 0.85, "mild": 0.12, "significant": 0.03},
}

# purpose -> (distance_km range, departure window min, arrival window min, time_constraint)
TRIP_TEMPLATES = {
    "commute":   ((2.0, 9.0),  (420, 540), (450, 570), "hard",   "work"),
    "education": ((1.0, 8.0),  (420, 510), (450, 555), "hard",   "school"),
    "shopping":  ((1.0, 6.0),  (540, 1200), (600, 1260), "soft", "shop"),
    "leisure":   ((1.0, 8.0),  (600, 1260), (660, 1320), "soft", "leisure"),
    "healthcare": ((1.0, 8.0), (480, 960),  (540, 1020), "medium", "healthcare"),
    "escort":    ((1.0, 6.0),  (420, 540),  (480, 600),  "medium", "school"),
    "other":     ((1.0, 8.0),  (480, 1080), (540, 1140), "soft",  "other"),
}

# occupation-conditional purpose priors (consistency: students educate, retired/unemployed no commute)
_NONCOMMUTE = {"shopping": 0.35, "leisure": 0.25, "healthcare": 0.20, "other": 0.20}
PURPOSE_BY_OCCUPATION = {
    "student": {"education": 0.60, "leisure": 0.20, "shopping": 0.10, "other": 0.10},
    "retired": _NONCOMMUTE,
    "unemployed": _NONCOMMUTE,
    "other": {"commute": 0.20, "shopping": 0.30, "leisure": 0.20, "healthcare": 0.15, "escort": 0.15},
    "office_worker": {"commute": 0.70, "shopping": 0.08, "leisure": 0.08, "escort": 0.08, "healthcare": 0.03, "other": 0.03},
    "service_worker": {"commute": 0.70, "shopping": 0.08, "leisure": 0.08, "escort": 0.08, "healthcare": 0.03, "other": 0.03},
    "manual_worker": {"commute": 0.75, "shopping": 0.08, "leisure": 0.06, "escort": 0.06, "healthcare": 0.03, "other": 0.02},
}


def generate() -> tuple[list[Persona], list[Trip]]:
    rng = random.Random(SEED)
    personas, trips = [], []
    for i in range(N):
        pid = f"P{ID_OFFSET + i + 1:06d}"
        occ = _w(rng, PERSONA_PRIORS["occupation"])
        hh = rng.randint(1, 6)
        has_children = rng.random() < 0.30 and hh >= 2
        # age-income-occupation plausibility: students -> 18-24, retired -> 45+/65+
        if occ == "student":
            age = "18-24"
        elif occ == "retired":
            age = rng.choice(["45-64", "65+"])
        else:
            age = _w(rng, PERSONA_PRIORS["age_group"])
            if age == "65+" and occ in ("office_worker", "service_worker", "manual_worker"):
                age = "45-64"
        income = "low" if occ in ("student", "unemployed") else _w(rng, PERSONA_PRIORS["income_group"])
        car_own = rng.random() < PERSONA_PRIORS["car_ownership_prob"]
        lic_draw = rng.random()  # always consumed -> keeps the RNG stream identical to before
        lic_has = True if car_own else (lic_draw < PERSONA_PRIORS["driving_license_prob"])
        personas.append(Persona(
            persona_id=pid,
            age_group=age,
            income_group=income,
            occupation=occ,
            household_size=hh,
            has_children=has_children,
            car_ownership=car_own,
            driving_license=lic_has,
            bike_ownership=rng.random() < PERSONA_PRIORS["bike_ownership_prob"],
            transit_pass=rng.random() < PERSONA_PRIORS["transit_pass_prob"],
            habitual_mode=_w(rng, PERSONA_PRIORS["habitual_mode"]),
            schedule_flexibility=_w(rng, PERSONA_PRIORS["schedule_flexibility"]),
            mobility_limitation=_w(rng, PERSONA_PRIORS["mobility_limitation"]),
        ))
        purpose = _w(rng, PURPOSE_BY_OCCUPATION[occ])
        (dlo, dhi), (d_lo, d_hi), (a_lo, a_hi), tc, dest = TRIP_TEMPLATES[purpose]
        dep = rng.randint(d_lo, d_hi)
        arr = rng.randint(a_lo, a_hi)
        if arr <= dep:
            arr = dep + rng.randint(20, 60)
        trips.append(Trip(
            trip_id=f"T{ID_OFFSET + i + 1:06d}",
            purpose=purpose,
            origin_type="home",
            destination_type=dest,
            distance_km=_r(rng, dlo, dhi),
            desired_departure_min=dep,
            desired_arrival_min=arr,
            time_constraint=tc,
        ))
    return personas, trips


def write_base(personas: list[Persona], trips: list[Trip]) -> None:
    cols = ["persona_id", "age_group", "income_group", "occupation", "household_size",
            "has_children", "car_ownership", "driving_license", "bike_ownership",
            "transit_pass", "habitual_mode", "schedule_flexibility", "mobility_limitation",
            "trip_id", "purpose", "origin_type", "destination_type", "distance_km",
            "desired_departure_min", "desired_arrival_min", "time_constraint"]
    with open(BASE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p, t in zip(personas, trips):
            w.writerow([p.persona_id, p.age_group, p.income_group, p.occupation,
                        p.household_size, str(p.has_children).lower(),
                        str(p.car_ownership).lower(), str(p.driving_license).lower(),
                        str(p.bike_ownership).lower(), str(p.transit_pass).lower(),
                        p.habitual_mode, p.schedule_flexibility, p.mobility_limitation,
                        t.trip_id, t.purpose, t.origin_type, t.destination_type,
                        t.distance_km, t.desired_departure_min, t.desired_arrival_min,
                        t.time_constraint])


def assign_distance_matched(personas: list[Persona], trips: list[Trip]) -> None:
    import numpy as np
    from reference_pipeline.matsim_adapter import build_supply_view

    _g, _xy, _tt, activity_nodes = build_supply_view(NETWORK, NODES)
    ids = [str(n["node"]) for n in activity_nodes]
    xy = np.array([[float(n["x"]), float(n["y"])] for n in activity_nodes], dtype=np.float64)
    n = len(xy)
    extreme_idx = [int(np.argmin(xy[:, 0])), int(np.argmin(xy[:, 1])),
                   int(np.argmax(xy[:, 0])), int(np.argmax(xy[:, 1]))]
    max_d = max(float(np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1]).max()) for i in extreme_idx)
    print(f"activity nodes={n}, max achievable straight-line distance={max_d / 1000:.2f} km")

    def assign(trip_id: str, target_km: float) -> tuple[str, str, float]:
        rng = random.Random(int(hashlib.sha256(("sh100:" + trip_id).encode()).hexdigest()[:16], 16))
        target_m = target_km * 1000.0
        best = None
        for h in rng.sample(range(n), HOME_SAMPLES):
            d = np.hypot(xy[:, 0] - xy[h, 0], xy[:, 1] - xy[h, 1])
            j = int(np.argmin(np.abs(d - target_m)))
            err = abs(float(d[j]) - target_m)
            if best is None or err < best[0]:
                best = (err, h, j)
        _, h, j = best
        achieved = float(np.hypot(xy[j, 0] - xy[h, 0], xy[j, 1] - xy[h, 1])) / 1000.0
        return ids[h], ids[j], achieved

    cols = None
    rows = []
    n_clamp = 0
    with open(BASE_CSV, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames + ["origin_node", "dest_node"]
        for row in rd:
            o, d, achieved = assign(row["trip_id"], float(row["distance_km"]))
            if achieved < float(row["distance_km"]) * 0.9:
                n_clamp += 1
            row["origin_node"] = o
            row["dest_node"] = d
            rows.append(row)
    with open(DMATCH_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"distance-matched OD written; trips clamped (>10% short of target): {n_clamp}/{len(rows)}")


def main() -> int:
    personas, trips = generate()
    write_base(personas, trips)
    print(f"wrote {BASE_CSV} ({len(personas)} personas, {len(trips)} trips)")
    assign_distance_matched(personas, trips)
    print(f"wrote {DMATCH_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
