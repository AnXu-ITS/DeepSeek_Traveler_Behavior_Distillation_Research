# Phase C mirror — MNL-B decisions (E1 S4)

Population: N* = 10000 (seed 2026, identical across scenarios); MNL-B (frozen coefficients, spec S3); capacity factors 0.3/0.3; supply unchanged; lastIteration=0; S9 column = frozen Phase C (read-only reference).

## Gates

| scenario | exit | stuck persons | stuck transit veh | pt board/alight | gate |
|---|---|---|---|---|---|
| C0_baseline | 0 | 265 | 4299 | 7038/6178 | PASS |
| C1_heavy_rain | 0 | 272 | 4306 | 7355/6443 | PASS |
| C2_fare_increase | 0 | 321 | 4318 | 8572/7526 | PASS |
| C3_transit_delay | 0 | 214 | 4284 | 5575/4904 | PASS |
| C4_road_disruption | 0 | 275 | 4302 | 7289/6403 | PASS |
| C5_joint_rain_delay | 0 | 232 | 4287 | 5975/5248 | PASS |

## Decisions (student mode share)

| scenario | car | pt | bike | walk | shift mean (min) |
|---|---|---|---|---|---|
| C0_baseline | 36.9% | 31.7% | 21.8% | 9.6% | 0.0 (MNL fixed 0 by design) |
| C1_heavy_rain | 37.0% | 33.4% | 21.3% | 8.4% | 0.0 (MNL fixed 0 by design) |
| C2_fare_increase | 36.3% | 39.1% | 17.9% | 6.7% | 0.0 (MNL fixed 0 by design) |
| C3_transit_delay | 37.2% | 24.7% | 24.8% | 13.3% | 0.0 (MNL fixed 0 by design) |
| C4_road_disruption | 33.9% | 32.9% | 23.3% | 9.8% | 0.0 (MNL fixed 0 by design) |
| C5_joint_rain_delay | 37.4% | 26.8% | 24.4% | 11.5% | 0.0 (MNL fixed 0 by design) |

## System metrics

| scenario | PT boardings | mean trip time (min) | car VKT (km) | waiting pt | failed trips |
|---|---|---|---|---|---|
| C0_baseline | 7038 | 45.79 | 41945.2 | 7266 | 1125 |
| C1_heavy_rain | 7355 | 46.55 | 42069.4 | 7588 | 1184 |
| C2_fare_increase | 8572 | 48.8 | 41438.0 | 8834 | 1367 |
| C3_transit_delay | 5575 | 42.51 | 42314.9 | 5760 | 885 |
| C4_road_disruption | 7289 | 46.57 | 39295.9 | 7525 | 1161 |
| C5_joint_rain_delay | 5975 | 43.34 | 42448.7 | 6178 | 959 |

## C1–C5 vs C0 (paired; MNL-B vs frozen S9)

| scenario | MNL Δ pt share | S9 Δ pt share | MNL Δ car share | S9 Δ car share | MNL Δ boardings | S9 Δ boardings | MNL Δ VKT | S9 Δ VKT |
|---|---|---|---|---|---|---|---|---|
| C1_heavy_rain | +1.6pp | +20.0pp | +0.1pp | +9.3pp | +317 | +4148 | +124 | +8838 |
| C2_fare_increase | +7.4pp | +0.4pp | -0.5pp | +1.6pp | +1534 | +61 | -507 | +1859 |
| C3_transit_delay | -7.0pp | -15.1pp | +0.4pp | +0.8pp | -1463 | -3139 | +370 | +746 |
| C4_road_disruption | +1.2pp | +12.1pp | -2.9pp | -26.5pp | +251 | +2924 | -2649 | -30996 |
| C5_joint_rain_delay | -5.0pp | -6.7pp | +0.5pp | +9.4pp | -1063 | -1312 | +504 | +8884 |

## Honest boundaries

- MNL-B: mode choice only, departure shift = 0; coefficients frozen from outputs/e1_mnl/mnl_b_coefs.json (spec S3).
- MNL-B was estimated on the same Teacher supervision as S9's supply-aware adaptation (234 train states); it is NOT calibrated to revealed-preference data.
- The frozen Teacher labels are largely cost-insensitive (see E1 report T5): MNL-B's cost coefficient is positive (z=1.28), so its C2 (fare) response direction may be inverted — reported as measured, not fixed.
- Perturbations are demand-side context injections; network and schedule identical across scenarios.
- MRT schedules are frequency-based/synthetic (community GTFS feed, not official LTA DataMall).
- ~20% of transit vehicles are truncated at the 30:00 simulation end (pre-existing 10k + 0.3-factor property); deltas use the identical setting and remain informative.

*Generated from 6 scenario result JSONs in this directory.*
