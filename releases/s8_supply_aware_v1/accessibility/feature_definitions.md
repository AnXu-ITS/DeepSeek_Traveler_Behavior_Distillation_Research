# S8 Transit Accessibility Feature Definitions (frozen)

Source of truth: `configs/accessibility_features.yaml`, `data/singapore_accessibility/DATA_README.md`, `src/traveler_distillation/accessibility/gtfs_accessibility.py` (planner `plan_accessibility`), `accessibility/feature_schema.json`.

## Field definitions

| Field | Unit | Range | Computation | Normalization | Source |
|---|---|---|---|---|---|
| pt_feasible () | binary 0/1 | [0,1] | 1.0 = a feasible itinerary exists in the departure window under the routing rules; 0.0 otherwise | z-score (S8 train fit) | plan_accessibility (GTFS + OSM network) |
| pt_access_time_min (access_time_min (S7 field reused)) | minutes | >= 0 | network walk time origin -> first boarding stop (access radius 700 m) | z-score (FROZEN S7-W3 stats) | OSM network walk graph |
| pt_egress_time_min (egress_time_min) | minutes | >= 0 | network walk time last alighting stop -> destination (egress radius 700 m, extended 1.5 km allowed) | z-score (S8 train fit) | OSM network walk graph |
| pt_wait_time_min (wait_time_min) | minutes | >= 0 | first boarding departure - trip departure - access walk | z-score (S8 train fit) | GTFS departure times |
| pt_in_vehicle_time_min (in_vehicle_time_min) | minutes | >= 0 | total time aboard transit vehicles (direct or 1-transfer itinerary) | z-score (S8 train fit) | GTFS stop times |
| pt_transfer_count (transfers (S7 field reused)) | count | 0 or 1 (planner cap) | vehicle-to-vehicle transfer count of the itinerary | z-score (FROZEN S7-W3 stats) | GTFS itinerary search |
| pt_transfer_time_min (transfer_time_min) | minutes | >= 0 | alight vehicle -> board next (walk + wait at transfer; connection window 180-2700 s) | z-score (S8 train fit) | GTFS itinerary search |
| pt_total_door_to_door_time_min (travel_time_min of the pt alternative) | minutes | > 0 | T_PT = T_access + T_wait + T_invehicle + T_transfer + T_egress (infeasible sentinel = 120 min) | z-score (FROZEN S7-W3 stats) | plan_accessibility |
| pt_coverage_ratio (coverage_ratio) | ratio | [0,1] | R_coverage = D_PT,vehicle / D_OD (straight-line in-vehicle distance / OD distance), clipped [0,1] | z-score (S8 train fit) | GTFS stop geometry |
| pt_generalized_cost () | n/a |  | NOT DEFINED — S8 §7.10: no invented value-of-time parameters; raw features only | n/a | n/a |

## Routing rules (frozen for S8 features)

| Rule | Value |
|---|---|
| access radius | boarding stop within 700 m of origin (no wider fallback) |
| egress radius | alighting stop within 700 m of destination; extended egress <= 1.5 km |
| max transfers | 1 vehicle-to-vehicle |
| boarding buffer | access-aware: max(300 s, access_walk + 60 s) |
| connection window | 180-2700 s between alight and next board |
| departure window | first boarding within 1 h of trip departure |
| walk time | network-derived length / freespeed (1.2 m/s reference) |

## Feasibility convention

`pt_feasible = 0` states keep pt in the choice set with a documented sentinel travel time (120 min) — feasibility is a LEARNED behavior (FVR metric), not an availability mask.

## City-independence guarantee

All fields are numeric accessibility attributes with no location identity. `pt_generalized_cost` is deliberately NOT defined (no invented value-of-time).
