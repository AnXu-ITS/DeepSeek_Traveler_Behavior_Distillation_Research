# Phase C — Singapore Real-Network Scenario Report

Population: N* = 10000 (seed 2026, identical across scenarios); frozen S9 checkpoint (Supply-Aware Traveler Agent v2.0); capacity factors 0.3/0.3; supply unchanged in every scenario.

## Gates

| scenario | exit | stuck persons | stuck transit veh | pt board/alight | gate |
|---|---|---|---|---|---|
| C0_baseline | 0 | 211 | 4287 | 5613/4886 | PASS |
| C1_heavy_rain | 0 | 366 | 4330 | 9761/8494 | PASS |
| C2_fare_increase | 0 | 221 | 4286 | 5674/4939 | PASS |
| C3_transit_delay | 0 | 102 | 4246 | 2474/2176 | PASS |
| C4_road_disruption | 0 | 285 | 4311 | 8537/7441 | PASS |
| C5_joint_rain_delay | 0 | 183 | 4266 | 4301/3752 | PASS |

> stuck transit vehicles are aborted at the 30:00 simulation end — a pre-existing property of the frozen 10k + 0.3-factor setting (B.5C baseline: 4,347/20,966 departures). Gates check the HUMAN stuck RATE per PT boarding (<= 5%; S9 C0 = 3.8%, stable at 3.3-4.3% across scenarios) — human stuck scales with PT demand (riders missing the single allowed transfer), not with scenario pathology.

## Decisions (student / executed outbound)

| scenario | car | pt | bike | walk | shift mean (min) | earlier/later/unchanged |
|---|---|---|---|---|---|---|
| C0_baseline | 28.3%/25.7% | 25.3%/24.6% | 34.0%/34.0% | 12.4%/15.8% | 2.175 | 0.1245/0.748/0.1275 |
| C1_heavy_rain | 37.5%/34.1% | 45.3%/44.5% | 13.3%/13.3% | 3.8%/8.1% | -0.476 | 0.3534/0.4116/0.235 |
| C2_fare_increase | 29.9%/27.1% | 25.7%/24.9% | 32.5%/32.5% | 11.8%/15.4% | 2.654 | 0.1109/0.818/0.0711 |
| C3_transit_delay | 29.0%/26.3% | 10.2%/9.6% | 35.8%/35.7% | 25.0%/28.3% | -1.153 | 0.3535/0.5446/0.1019 |
| C4_road_disruption | 1.8%/1.6% | 37.5%/36.8% | 41.8%/41.8% | 19.0%/19.9% | 2.136 | 0.1776/0.7353/0.0871 |
| C5_joint_rain_delay | 37.6%/34.2% | 18.6%/17.8% | 24.5%/24.5% | 19.2%/23.5% | -4.539 | 0.4985/0.3706/0.1309 |

## System metrics

| scenario | PT boardings | mean trip time (min) | car mean TT (min) | road delay (s/pass) | slow share | car VKT (km) | waiting pt | failed trips |
|---|---|---|---|---|---|---|---|---|
| C0_baseline | 5613 | 43.67 | 10.11 | 0.59 | 0.0014 | 33939.0 | 5787 | 938 |
| C1_heavy_rain | 9761 | 50.86 | 9.89 | 0.59 | 0.0016 | 42777.5 | 10070 | 1633 |
| C2_fare_increase | 5674 | 43.86 | 10.08 | 0.59 | 0.0014 | 35797.8 | 5854 | 956 |
| C3_transit_delay | 2474 | 32.23 | 10.16 | 0.59 | 0.0015 | 34684.6 | 2559 | 400 |
| C4_road_disruption | 8537 | 52.71 | 12.5 | 0.56 | 0.001 | 2943.4 | 8792 | 1381 |
| C5_joint_rain_delay | 4301 | 38.27 | 9.91 | 0.59 | 0.0016 | 42822.8 | 4457 | 732 |

## C1–C5 vs C0 (paired, same population)

| scenario | Δ PT boardings | Δ mean trip time | Δ car TT | Δ road delay | Δ VKT | Δ car share | Δ pt share |
|---|---|---|---|---|---|---|---|
| C1_heavy_rain | +4148.00 | +7.19 | -0.22 | +0.00 | +8838.50 | +9.3pp | +20.0pp |
| C2_fare_increase | +61.00 | +0.19 | -0.03 | +0.00 | +1858.80 | +1.6pp | +0.4pp |
| C3_transit_delay | -3139.00 | -11.44 | +0.05 | +0.00 | +745.60 | +0.8pp | -15.1pp |
| C4_road_disruption | +2924.00 | +9.04 | +2.39 | -0.03 | -30995.60 | -26.5pp | +12.1pp |
| C5_joint_rain_delay | -1312.00 | -5.40 | -0.20 | +0.00 | +8883.80 | +9.4pp | -6.7pp |

## Honest boundaries

- Perturbations are demand-side context injections; network and schedule are identical in every scenario.
- MRT schedules are frequency-based/synthetic (community GTFS feed, not official LTA DataMall).
- Demand is synthetic personas/trips (seed 2026); no real Singapore traveler behavior.
- Wording: "a controlled real-network experiment under calibrated effective capacity".
- ~20% of transit vehicles are truncated at the 30:00 simulation end (pre-existing in the frozen 10k + 0.3-factor setting, same as B.5C baseline); PT absolute quantities carry this artifact — scenario-vs-C0 deltas use the identical setting and remain informative.

*Generated from 6 scenario result JSONs in this directory.*
