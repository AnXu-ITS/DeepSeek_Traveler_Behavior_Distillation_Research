# Historical population-seed robustness experiment

This record examines repeated Singapore scenario populations under fixed model and supply settings. Population seeds change synthetic people/trips and are distinct from the training seeds of the later controlled neural comparison. Paired scenarios share their population within a seed. Reported intervals and variation retain the historical population, event and scenario definitions.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/evidence/e4_multiseed/E4_REPORT.md).

[Current research](<../../docs/RESEARCH_DESIGN.md>) · [Training](<../../docs/TRAINING.md>) · [Results](<../../docs/RESULTS.md>) · [Data and access](<../../docs/DATA_SOURCES.md>) · [Model use](<../../docs/MODEL_USE.md>)

## Scenario outcomes by population seed

| seed | scenario | decision share car/pt/bike/walk(%) | PT boardings | car VKT(km) | stuck people(rate) | failed trips | mean trip time(min) |
|---|---|---|---|---|---|---|---|
| 2026(frozen) | C0_baseline | 28.3/25.3/34.0/12.4 | 5613 | 33939.0 | 211(3.8%) | 938 | 43.67 |
| 2026(frozen) | C1_heavy_rain | 37.5/45.3/13.3/3.8 | 9761 | 42777.5 | 366(3.7%) | 1633 | 50.86 |
| 2026(frozen) | C2_fare_increase | 29.9/25.7/32.5/11.8 | 5674 | 35797.8 | 221(3.9%) | 956 | 43.86 |
| 2026(frozen) | C3_transit_delay | 29.0/10.2/35.8/25.0 | 2474 | 34684.6 | 102(4.1%) | 400 | 32.23 |
| 2026(frozen) | C4_road_disruption | 1.8/37.5/41.8/19.0 | 8537 | 2943.4 | 285(3.3%) | 1381 | 52.71 |
| 2026(frozen) | C5_joint_rain_delay | 37.6/18.6/24.5/19.2 | 4301 | 42822.8 | 183(4.3%) | 732 | 38.27 |
| 42 | C0_baseline | 27.7/26.1/33.3/12.9 | 5819 | 32667.2 | 214(3.7%) | 945 | 44.42 |
| 42 | C1_heavy_rain | 36.9/45.8/13.6/3.7 | 9927 | 41366.3 | 345(3.5%) | 1602 | 50.94 |
| 42 | C2_fare_increase | 29.5/26.6/31.8/12.1 | 5918 | 34600.7 | 222(3.8%) | 968 | 44.32 |
| 42 | C3_transit_delay | 28.2/10.8/34.9/26.0 | 2587 | 33055.5 | 96(3.7%) | 428 | 32.84 |
| 42 | C4_road_disruption | 1.6/37.8/41.0/19.6 | 8587 | 2546.2 | 290(3.4%) | 1386 | 52.81 |
| 42 | C5_joint_rain_delay | 37.0/19.7/24.2/19.1 | 4516 | 41408.5 | 186(4.1%) | 812 | 39.17 |
| 7 | C0_baseline | 28.2/25.9/33.3/12.6 | 5860 | 33324.2 | 207(3.5%) | 910 | 44.44 |
| 7 | C1_heavy_rain | 37.4/46.0/13.2/3.5 | 10011 | 41973.2 | 336(3.4%) | 1644 | 51.06 |
| 7 | C2_fare_increase | 29.7/26.4/32.1/11.8 | 5940 | 34827.1 | 223(3.8%) | 951 | 44.66 |
| 7 | C3_transit_delay | 28.9/10.2/35.3/25.6 | 2552 | 33896.5 | 87(3.4%) | 383 | 32.53 |
| 7 | C4_road_disruption | 1.7/38.2/41.3/18.8 | 8708 | 2640.4 | 275(3.2%) | 1385 | 53.05 |
| 7 | C5_joint_rain_delay | 37.4/18.9/24.4/19.3 | 4417 | 41967.2 | 181(4.1%) | 764 | 39.34 |

## Paired scenario changes from baseline by population seed

| seed | scenario | Δcar share(pp) | Δpt share(pp) | ΔPT boardings | Δcar VKT(km) | Δstuck people | Δmean trip time(min) |
|---|---|---|---|---|---|---|---|
| 2026(frozen) | C1_heavy_rain | +9.3 | +20.0 | +4148 | +8838.5 | +155 | +7.19 |
| 2026(frozen) | C2_fare_increase | +1.6 | +0.4 | +61 | +1858.8 | +10 | +0.19 |
| 2026(frozen) | C3_transit_delay | +0.8 | -15.1 | -3139 | +745.6 | -109 | -11.44 |
| 2026(frozen) | C4_road_disruption | -26.5 | +12.1 | +2924 | -30995.6 | +74 | +9.04 |
| 2026(frozen) | C5_joint_rain_delay | +9.4 | -6.7 | -1312 | +8883.8 | -28 | -5.40 |
| 42 | C1_heavy_rain | +9.2 | +19.7 | +4108 | +8699.1 | +131 | +6.52 |
| 42 | C2_fare_increase | +1.8 | +0.5 | +99 | +1933.5 | +8 | -0.10 |
| 42 | C3_transit_delay | +0.6 | -15.3 | -3232 | +388.3 | -118 | -11.58 |
| 42 | C4_road_disruption | -26.1 | +11.7 | +2768 | -30121.0 | +76 | +8.39 |
| 42 | C5_joint_rain_delay | +9.3 | -6.4 | -1303 | +8741.3 | -28 | -5.25 |
| 7 | C1_heavy_rain | +9.2 | +20.1 | +4151 | +8649.0 | +129 | +6.62 |
| 7 | C2_fare_increase | +1.5 | +0.5 | +80 | +1502.9 | +16 | +0.22 |
| 7 | C3_transit_delay | +0.6 | -15.7 | -3308 | +572.3 | -120 | -11.91 |
| 7 | C4_road_disruption | -26.5 | +12.3 | +2848 | -30683.8 | +68 | +8.61 |
| 7 | C5_joint_rain_delay | +9.1 | -6.9 | -1443 | +8643.0 | -26 | -5.10 |

## Cross-seed summaries of scenario effects

| scenario | Δpt share mean ± std(min–max) | ΔPT boardings mean ± std(min–max) | Δcar share mean ± std | ΔVKT mean ± std | Δstuck mean ± std | sign agreement(Δpt/Δboardings) | \|mean\|/std(Δpt) |
|---|---|---|---|---|---|---|
| C1_heavy_rain | 19.9 ± 0.2(19.7–20.1) | 4136 ± 24(4108–4151) | 9.2 ± 0.1(9.2–9.3) | 8728.9 ± 98.2(8649.0–8838.5) | 138 ± 14(129–155) | 3/3 / 3/3 | 95.76 |
| C2_fare_increase | 0.5 ± 0.1(0.4–0.5) | 80 ± 19(61–99) | 1.6 ± 0.2(1.5–1.8) | 1765.1 ± 230.1(1502.9–1933.5) | 11 ± 4(8–16) | 3/3 / 3/3 | 8.08 |
| C3_transit_delay | -15.4 ± 0.3(-15.7–-15.1) | -3226 ± 85(-3308–-3139) | 0.7 ± 0.1(0.6–0.8) | 568.7 ± 178.7(388.3–745.6) | -116 ± 6(-120–-109) | 3/3 / 3/3 | 50.30 |
| C4_road_disruption | 12.0 ± 0.3(11.7–12.3) | 2847 ± 78(2768–2924) | -26.4 ± 0.2(-26.5–-26.1) | -30600.1 ± 443.3(-30995.6–-30121.0) | 73 ± 4(68–76) | 3/3 / 3/3 | 39.39 |
| C5_joint_rain_delay | -6.7 ± 0.3(-6.9–-6.4) | -1353 ± 78(-1443–-1303) | 9.3 ± 0.2(9.1–9.4) | 8756.0 ± 121.1(8643.0–8883.8) | -27 ± 1(-28–-26) | 3/3 / 3/3 | 26.49 |

## Fare-increase response stability across population seeds

|primary metric| 3-seed Δ value(2026/42/7) |level|
|---|---|---|
| Δpt share(pp) | +0.4 / +0.5 / +0.5 | **stable** |
| ΔPT boardings | +61 / +99 / +80 | **stable** |

## Artifact checksums and runtime by seed and scenario

| seed / scenario | population.xml SHA256(before 12) | manifest SHA256(before 12) | checkpoint SHA256(before 12) | build / matsim / parse(s) |
|---|---|---|---|---|
| 42/C0_baseline | 5e6f55156719 | 9f6297bff65c | 6af79b44bc69 | 2557.7 / 125.9 / 61.8 |
| 42/C1_heavy_rain | f88845089d2f | 74645c43cfed | 6af79b44bc69 | 1583.6 / 122.4 / 63.8 |
| 42/C2_fare_increase | 730979551ae2 | a46c425c1051 | 6af79b44bc69 | 1610.9 / 127.7 / 66.7 |
| 42/C3_transit_delay | 2d9681e9dbd8 | ececa393e9c2 | 6af79b44bc69 | 1591.4 / 126.6 / 70.4 |
| 42/C4_road_disruption | a4cb438fac33 | ba9d86392b5a | 6af79b44bc69 | 1970.2 / 128.9 / 63.1 |
| 42/C5_joint_rain_delay | 2dfa0050f43c | ff8eaddc6ff3 | 6af79b44bc69 | 1674.4 / 115.1 / 66.4 |
| 7/C0_baseline | a127034b1e59 | f010b3249595 | 6af79b44bc69 | 2517.3 / 126.3 / 63.7 |
| 7/C1_heavy_rain | d5d3ec0c3faa | b425f9c0269d | 6af79b44bc69 | 8998.7 / 124.4 / 61.3 |
| 7/C2_fare_increase | 20874a1ceea5 | 299ae0fbc35e | 6af79b44bc69 | 1630.4 / 126.3 / 65.4 |
| 7/C3_transit_delay | 28c3e0160ff0 | aeb2558cebf1 | 6af79b44bc69 | 1637.5 / 128.5 / 68.5 |
| 7/C4_road_disruption | d176b21d25c8 | c60553d25f3b | 6af79b44bc69 | 2181.3 / 132.8 / 69.4 |
| 7/C5_joint_rain_delay | 57bf19ed9f94 | 1a0dbe7563a8 | 6af79b44bc69 | 2159.4 / 136.7 / 77.8 |
