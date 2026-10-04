# Historical population-scale experiment

This experiment constructs nested synthetic populations under the frozen Singapore supply/model settings, reaching 50,000 agents. It separates inference from supply/route preparation and MATSim execution. The fixed 10,000-person baseline provides a consistency reference. Scaling a synthetic population demonstrates computational/execution behavior under those conditions, not validation of population-representative travel demand.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/evidence/e3_scale/E3_REPORT.md).

[Current research](<../../docs/RESEARCH_DESIGN.md>) · [Training](<../../docs/TRAINING.md>) · [Results](<../../docs/RESULTS.md>) · [Data and access](<../../docs/DATA_SOURCES.md>) · [Model use](<../../docs/MODEL_USE.md>)

## Runtime, memory and artifact size by population scale

| N | T_setup | T_factory | T_decide(mean/P50/P95 ms · total) | T_residual | T_build | T_matsim | T_parse | peak RAM py(P1/Δ, MB) | peak RAM jvm(MB) | population.xml | events.zst | states/s | agents/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,000 | 6.1 s | 3.7 min | 1.0195/0.7959/1.9455 ms · 1.0 s | 2.2 min | 5.9 min | 2.4 min | 55.2 s | 1,966/746 | 6,564 | 9 | 200 | 2.83 | 6.89 |
| 10,000 | 5.8 s | 31.6 min | 1.0309/0.7284/1.8277 ms · 10.3 s | 19.3 min | 51.1 min | 2.1 min | 1.2 min | 2,742/1,497 | 6,569 | 94 | 273 | 3.26 | 81.26 |
| 20,000 | 5.8 s | 1.01 h | 0.9417/0.7476/1.815 ms · 18.8 s | 38.1 min | 1.65 h | 2.4 min | 1.6 min | 3,272/2,004 | 6,577 | 189 | 352 | 3.37 | 138.82 |
| 50,000 | 5.9 s | 2.22 h | 0.9759/0.7463/1.7967 ms · 48.8 s | 1.63 h | 3.86 h | 3.4 min | 2.7 min | 5,271/3,924 | 6,559 | 472 | 589 | 3.6 | 247.43 |

## Behavioral decisions and simulation outcomes by population scale

| N | decision share(car/pt/bike/walk) | execution share | PT validity | boardings | stuck people(rate) | failed trips | mean trip time(min) | car VKT(km) | road delay(s/pass) | slow share | drift vs 10k(pp, behavioral decision stage) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10k(frozen reference) | 28.3/25.3/34.0/12.4 | 25.7/24.6/34.0/15.8 | 93.5%(164/2531) | 5613 | 211(3.8%) | 938 | 43.67 | 33939.0 | 0.59 | 0.0014 | 0/0/0/0 |
| 1,000 | 27.3/24.7/34.6/13.4 | 25.4/24.2/34.6/15.8 | 95.1%(12/247) | 560 | 27(4.8%) | 98 | 43.82 | 3308.2 | 0.58 | 0.0014 | -1.0/-0.6/+0.6/+1.0 |
| 10,000(E3 reproduction) | 28.3/25.3/34.0/12.4 | 25.7/24.6/34.0/15.8 | 93.5%(164/2531) | 5613 | 211(3.8%) | 938 | 43.67 | 33939.0 | 0.59 | 0.0014 | +0.0/+0.0/+0.0/+0.0 |
| 20,000 | 28.5/25.3/33.9/12.3 | 25.7/24.7/33.8/15.8 | 93.9%(307/5068) | 11357 | 426(3.8%) | 1956 | 43.9 | 67599.6 | 0.62 | 0.0021 | +0.2/+0.0/-0.1/-0.1 |
| 50,000 | 28.8/25.5/33.4/12.3 | 26.0/24.9/33.3/15.7 | 94.2%(738/12766) | 28440 | 1153(4.1%) | 5586 | 48.92 | 171193.1 | 1.45 | 0.02 | +0.5/+0.2/-0.6/-0.1 |

## Runtime and memory scaling ratios

|transition| T_factory | T_decide | T_build | T_matsim | peak RAM py | peak RAM jvm |linear expectation|
|---|---|---|---|---|---|---|---|
| 1,000→10,000 | 8.56 | 10.11 | 8.67 | 0.85 | 1.39 | 1.00 | ×10 |
| 10,000→20,000 | 1.92 | 1.83 | 1.94 | 1.17 | 1.19 | 1.00 | ×2 |
| 20,000→50,000 | 2.20 | 2.59 | 2.34 | 1.40 | 1.61 | 1.00 | ×2.5 |

## Runtime bottlenecks and resource-limit checks

| N | wall-clock share build/matsim/parse | resource caution(timeout/OOM/≥24 GB) | congestion measure(slow share / stuck rate) |interpretation|
|---|---|---|---|---|
| 1,000 | 64%/26%/10% |none| 0.0014 / 4.8% | bottleneck = build; ceiling = not triggered |
| 10,000 | 94%/4%/2% |none| 0.0014 / 3.8% | bottleneck = build; ceiling = not triggered |
| 20,000 | 96%/2%/2% |none| 0.0021 / 3.8% | bottleneck = build; ceiling = not triggered |
| 50,000 | 97%/1%/1% |none| 0.02 / 4.1% | bottleneck = build; ceiling = not triggered |
