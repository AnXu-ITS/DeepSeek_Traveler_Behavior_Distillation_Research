# E5 — Helsinki Zero-Shot Transfer Report

Frozen S9 (SHA256 6af79b44bc699c00…) · N=10000 · seed 2026 · capacity 0.3/0.3 · lastIteration=0 · scenarios C0–C5 (C2/C4/C5 run under design V4 conditional extension: C0/C1/C3 gates passed).
Supply: HSL GTFS 2026-08-27 04:51:22 (2026-09-09) + hsl.osm.pbf subregion; reference: `TRC_AIT_5_E5_SECOND_CITY_ZERO_SHOT_DESIGN.md` v0.1.

## T1 — Supply parity (left = Singapore frozen)

| item | Singapore (frozen) | Helsinki (measured) |
|---|---|---|
| bbox / area | 1.330-1.400 x 103.900-104.015 (≈100 km2) | 60.145-60.23 x 24.875-25.06 (≈97 km2) |
| network nodes / links / km | 100867 / 191617 / 3520.58 | 290204 / 638328 / 7494.86 |
| connectivity largest comp (car/bike/walk) | 0.9515 / 0.9316 / 0.951 | 0.9745 / 0.9711 / 0.981 |
| GTFS trips (bus/rail) / stops | 21286 (20382/904) / 859 | 13655 (9567/4088) / 1269 |
| routing failures (unique seq) | 0 / 216 | 0 / 262 |
| snapping mean/p50/p90/max (m) | 61.3 / 24.1 / 191.1 / 687.5 | 11.5 / 8.4 / 21.7 / 223.6 |
| activity nodes | (Singapore frozen) | 246020 |
| reference day / service | WD | 2026-09-09 (wednesday; 507 active ids) |

## T2 — C0 baseline execution and decisions

| city | decision share (car/pt/bike/walk) | executed share | PT decision validity (overall / feasible-cond.) | boardings | stuck persons (rate) | failed trips | car VKT | slow share |
|---|---|---|---|---|---|---|---|---|
| Singapore (frozen) | 28.3%/25.3%/34.0%/12.4% | 25.7%/24.6%/34.0%/15.8% | ≈93.5% (164/2,530, frozen manifest) | 5613 | 211 (3.8%) | 938 | 33939.0 | 0.0014 |
| Helsinki C0 | 28.6%/23.0%/34.6%/13.8% | 27.0%/17.5%/34.6%/20.9% | 75.9% (556/2304) / 100.0% (0/1748) | 3824 | 1 (0.03%) | 1 | 39170.9 | 0.0035 |

30:00 transit truncation (stuck transit vehicles at sim end): Singapore frozen 4287 / 20,966 vehicle departures; Helsinki C0 0 / 13,655 vehicle departures. (Pre-existing artifact of the single-departure-per-trip supply; reported, not gated.)

Departure shift (C0): mean 2.048 min; earlier/later/unchanged 0.1325/0.7352/0.1323.

PT fallback reasons (C0, outbound): {'no_direct_or_transfer': 556} — all fallbacks are learned Class-E pt choices (frozen S9 P(pt|E)=0.213 in Singapore); feasibility-conditioned validity is the execution-fidelity metric.

## T3 — Supply-aware zero-shot response: P(PT) by accessibility class

| city | A | B | C | D | E | E-A | n per class (A/B/C/D/E) |
|---|---|---|---|---|---|---|---|
| Singapore (frozen S9 accessibility eval; file models.B1_S8 = S9 checkpoint) | 0.366 | 0.452 | 0.326 | 0.215 | 0.213 | -0.153 | 12/2/13/12/12 |
| Helsinki (S9 zero-shot, C0 decisions) | 0.236 | 0.254 | 0.198 | 0.254 | 0.189 | -0.047 | 954/846/445/4806/2949 |

> Helsinki A vs E direction reproduces (E-A = -0.047); full A→E monotonicity: NO (non-monotonic middle classes — reported as measured). Singapore frozen E-A = -0.153. The zero-shot gradient is present in direction but flatter/less monotonic than in-domain — an honest cross-city observation, not a claim of equal strength.

> Helsinki P(PT) = argmax share among C0 decisions grouped by plan_accessibility class (classify_accessibility, frozen S8 §9 rule); Singapore row = frozen S9 accessibility eval (pt_prob_by_class means, bootstrap CI in source file). Both are S9 outputs on different supplies — the E5 claim is the qualitative A→E gradient, not numeric equality.

## T4 — Scenario response direction (paired, same population)

| scenario | city | Δcar | Δpt | Δbike | Δwalk | Δboardings | ΔVKT | Δstuck | direction |
|---|---|---|---|---|---|---|---|---|---|
| C1-C0 | Singapore (frozen) | +9.3pp | +20.0pp | -20.7pp | -8.6pp | +4148.0 | +8838.5 | +155.0 | frozen direction |
| C1-C0 | Helsinki | +9.0pp | +17.6pp | -17.3pp | -9.3pp | +3291.0 | +9500.1 | +1.0 | consistent (7/7 sign matches) |
| C2-C0 | Singapore (frozen) | +1.6pp | +0.4pp | -1.4pp | -0.6pp | +61.0 | +1858.8 | +10.0 | frozen direction |
| C2-C0 | Helsinki | +1.6pp | +1.5pp | -1.3pp | -1.7pp | +86.0 | +1830.6 | +0.0 | consistent (6/7 sign matches) |
| C3-C0 | Singapore (frozen) | +0.8pp | -15.1pp | +1.8pp | +12.5pp | -3139.0 | +745.6 | -109.0 | frozen direction |
| C3-C0 | Helsinki | +0.4pp | -13.8pp | +1.4pp | +12.0pp | -2809.0 | +305.7 | +0.0 | consistent (6/7 sign matches) |
| C4-C0 | Singapore (frozen) | -26.5pp | +12.1pp | +7.8pp | +6.6pp | +2924.0 | -30995.6 | +74.0 | frozen direction |
| C4-C0 | Helsinki | -23.7pp | +7.5pp | +8.3pp | +7.9pp | +1775.0 | -29478.7 | +0.0 | consistent (6/7 sign matches) |
| C5-C0 | Singapore (frozen) | +9.4pp | -6.7pp | -9.5pp | +6.8pp | -1312.0 | +8883.8 | -28.0 | frozen direction |
| C5-C0 | Helsinki | +9.1pp | -4.1pp | -9.1pp | +4.1pp | -1419.0 | +9412.4 | +0.0 | consistent (6/7 sign matches) |

> Expected directions (frozen Singapore Phase C): C1 rain → away from bike/walk; C3 delay → away from pt. Helsinki direction consistency is QUALITATIVE (no cross-city statistical test; no Helsinki ground-truth labels).

## T5 — Gates

| gate | result |
|---|---|
| G0 identity & red lines (S9 SHA256 6af79b44…31e6, zero Teacher) | PASS |
| G1 supply QA (revised parity band 0.5-4x SG per design revision-log #2 + connectivity>=90% + snap p90<=250m + routing failures=0 + trips2 cross-check) | PASS |
| G2 determinism (1k pilot manifest SHA256 identical) | PASS (identical SHA256) |
| G3 decision validity (four modes + known fallback reasons; PT validity REPORTED) | PASS (C0 PT validity 75.9% overall / 100.0% feasible-conditioned; reasons {'no_direct_or_transfer': 556}) |
| G4 simulation (exit 0, four modes, alightings≤boardings, stuck≤5%) | PASS |
| G5 migration audit (covariate audit produced; ≥4 classes with ≥50 decisions) | PASS |

## Covariate-shift audit (Helsinki C0 vs frozen Singapore normalization)

| field | HEL mean | HEL std | HEL p50 | HEL p95 | SG mean | SG std | z mean | z p95 | share \|z\|>3 |
|---|---|---|---|---|---|---|---|---|---|
| pt_feasible | 0.7051 | 0.456 | 1.0 | 1.0 | 0.19017094017094016 | 0.39243592303131586 | 1.312 | 2.064 | 0.0 |
| egress_time_min | 10.6003 | 9.6708 | 8.883 | 23.568 | 1.2976495726495725 | 3.3477045636001677 | 2.779 | 6.652 | 0.4778 |
| wait_time_min | 12.017 | 63.6105 | 4.259 | 15.744 | 3.479190170940171 | 34.694515955907065 | 0.246 | 0.354 | 0.0142 |
| in_vehicle_time_min | 11.8137 | 11.9706 | 9.0 | 35.0 | 2.72170405982906 | 7.506830903568419 | 1.211 | 4.3 | 0.1554 |
| transfer_time_min | 0.7955 | 2.5613 | 0.0 | 7.0 | 0.0819423076923077 | 0.6665041475661496 | 1.071 | 10.38 | 0.1147 |
| coverage_ratio | 0.6154 | 0.4367 | 0.8669 | 1.0 | 0.17631207264957263 | 0.3730806910921411 | 1.177 | 2.208 | 0.0 |

Class counts (C0 decisions): {'D_poor': 4806, 'A_excellent': 954, 'E_infeasible': 2949, 'C_moderate': 445, 'B_good': 846}

## Honest boundaries

- No Helsinki behavioral labels: Teacher calls are prohibited (G0); cross-city consistency is QUALITATIVE.
- Normalization stats are Singapore-fitted and frozen; covariate shift reported, never refit.
- Scenario injection identical to Singapore (C1 rain 0.75 / C2 fare ×1.5 / C3 delay 15 min / C4 road disruption +20 min / C5 joint rain+delay via context/alternatives only).
- C2/C4/C5 ran under the design V4 conditional extension (C0/C1/C3 all gates passed + time budget allowed); C2 is the first scenario of a fresh process and repays the full accessibility pass (~2.5-3 h) — runtime only, decisions identical.
- City-specific supply notes: rail-on-road for metro/commuter rail/tram; night departures outside [05:00, 23:00] dropped; 30:00 truncation measured; capacity factors 0.3/0.3 carried over ('frozen effective-capacity setting carried over', not a Helsinki calibration).
- Population is synthetic personas/trips (seed 2026); OD from Helsinki activity_nodes hashing; not real Helsinki travel.
- Zero-shot supply response is QUALITATIVELY reproduced but WEAKER than in-domain: T3 E-A = -0.047 (measured) vs Singapore -0.153 and the middle classes are non-monotonic — reported as measured, not smoothed.
- Environment incident (2026-08-28): Windows Temp cleanup deleted matsim_rel mid-run; C0 MATSim exited 1 (NoClassDefFoundError) and was rerun from the durable workbench copy (exit 0); C1/C3 ran normally after the Temp restore. The E5 runner now pins the durable classpath (process-local override).
- Machine contention: an E4 multi-seed run (run_e4_multiseed.py --seed 42) was started by the user at 23:58 and ran concurrently with the E5 C3 build — decisions are deterministic and unaffected; C3 wall-clock timings carry contention (timing is a performance observation only).
- Timing is single-machine (24C / 31.4 GB / CPU-only torch) — performance observation only.
- Licenses: HSL GTFS (CC BY 4.0, attribution HSL), OSM (ODbL).
