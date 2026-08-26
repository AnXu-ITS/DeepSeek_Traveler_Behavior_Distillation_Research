# S8 Singapore Accessibility Dataset — Data Notes

**Built by** `scripts/build_singapore_accessibility_dataset.py` (S8 instructions §4–7).
**Location**: `data/singapore_accessibility/`

## Supply sources (§8 record)

| Item | Value |
|---|---|
| OSM snapshot | `data/singapore/osm/tampines_pasir_ris.osm`, Overpass bbox 2026-08-25, lat 1.330–1.400 × lon 103.900–104.015 (ODbL) |
| GTFS source | community-constructed `singapore-gtfs-2025` (NOT official LTA DataMall); sha256 `1fcc6f5f…b3cd`; feed dates 2025-01-01–2030-12-31 |
| Study area | Tampines + Pasir Ris (same bbox as Phase A/B supply) |
| Network | `transit/network_with_transit.xml` (OSM-derived, UTM 48N) |
| Stops | 859 served stops (full-day supply 05:00–23:00, 20,966 trips) |
| OD candidate nodes | 50,241 (car-accessible ∩ walk largest component) |

## Routing rules (frozen for S8 features)

| Rule | Value |
|---|---|
| access radius | boarding stop within **700 m** of origin (no wider fallback — a longer walk would be temporally impossible) |
| egress radius | alighting stop within **700 m** of destination; extended egress ≤ **1.5 km** allowed |
| max transfers | **1** vehicle-to-vehicle (same rule as `MATSimAdapter._plan_pt`) |
| boarding buffer | access-aware: `max(300 s, access_walk + 60 s)` |
| connection window | 180–2700 s between alight and next board |
| departure window | first boarding within 1 h of the trip departure |
| walk time | network-derived `length / freespeed` |

## Feature vector (city-independent — no location identity)

Core S7 fields (travel_time, monetary_cost, access_time, transfers,
reliability_delay, weather_exposure) + 6 S8 fields: `pt_feasible`,
`egress_time_min`, `wait_time_min`, `in_vehicle_time_min`,
`transfer_time_min`, `coverage_ratio` (= D_PT,vehicle / D_OD, clip [0,1]).

`pt_feasible = 0` states keep pt in the choice set with a documented sentinel
travel time (120 min) — feasibility is a LEARNED behavior (FVR metric, §21),
not an availability mask.

## Splits (frozen; see `split_manifest.json`)

- **Persona holdout**: 28 train / 6 val / 6 test (S7/S3-C convention personas).
- **OD holdout**: train 79 / val 16 / test 19 ODs, pairwise disjoint; every
  test state uses only test ODs and test personas (double holdout).
- **Accessibility holdout**: feasible ODs with walking burden
  (access + egress) ≥ 15 min appear only in the test pool.
- **Curves**: every (persona, trip) group spans ≥ 3 accessibility classes
  (A excellent / B good / C moderate / D poor / E infeasible); 80 groups.
- **K=5 boundary samples** (§13): good↔medium, medium↔poor, poor↔infeasible
  transitions within a group (`boundary_pairs.json`).

## Files

| File | Content |
|---|---|
| `records.jsonl` | 338 states (persona/trip/context/alternatives + accessibility vector; NO location ids) |
| `od_manifest.json` | anonymous `od_index` → routing provenance (origin/dest node ids; internal only) |
| `split_manifest.json` | frozen splits + sanity report |
| `boundary_pairs.json` | K=5 teacher sampling list |
| `candidate_probe.jsonl` | stratification evidence for every probed OD pair |
| `states_with_teacher.jsonl` | teacher-labeled aggregated targets (K=3/5) |
| `repeat_records.jsonl` | raw teacher repeats with usage/completion evidence |
| `generation_manifest.json` | labeling stats (prompt_version `teacher_s8_accessibility_v0.1`) |

## Honesty notes

- The GTFS feed is community-constructed; MRT schedules are frequency-based.
  S8 consumes only **accessibility attributes derived from this supply**, never
  schedule identities. Publication wording must follow
  `data/singapore/gtfs/README_snapshot.md` and S8 §28–29.
- Teacher labels are synthetic-persona LLM inferences (no real human choice
  data) — S8 §29 prohibits claiming "real Singapore traveler behavior".
