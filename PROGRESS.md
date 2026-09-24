# Research development record

This record traces the development from early pointwise distillation through multi-context and mechanism supervision, S7-W3 freezing, the deprecated S8 adaptation, and corrected S9 deployment. Earlier prototype feedback experiments and operational gates have their own scope. The current paper uses the later controlled-response, human-response and fixed-behavior execution comparisons. Historical numbers below remain associated with their original stages.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/PROGRESS.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<docs/RESEARCH_DESIGN.md>) · [Training](<docs/TRAINING.md>) · [Results](<docs/RESULTS.md>) · [Data and access](<docs/DATA_SOURCES.md>) · [Model use](<docs/MODEL_USE.md>)

## Historical numerical table 1

|Metric (test)| v0.2-A | v0.2-B | v0.2-C |
|---|---|---|---|
| mode_accuracy | 0.7917 | 0.7917 | 0.7917 |
| KL | 0.0984 | 0.1160 | **0.0787** |
| probability L1 | 0.3031 | 0.3503 | **0.2891** |
| counterfactual \|ΔP_T−ΔP_S\| | **0.1269** | 0.1564 | 0.1461 |
| heterogeneity \|ΔP_T−ΔP_S\| | — | — | 0.1601 (8pairs) |

## Historical numerical table 2

|Metric (test)| v0.3-A | v0.3-B (+decomposed response supervision) | v0.3-C (+heterogeneity) |
|---|---|---|---|
| mode_accuracy | 0.7708 | **0.8125** | **0.8125** |
| KL(P_T‖P_S) | 0.3816 | 0.1440 | **0.1092** |
| probability L1 | 0.7605 | 0.3260 | **0.2860** |
| counterfactual \|dP_T−dP_S\| | 0.0661 | 0.0596 | **0.0539** |
| sign agreement | 0.6508 | **0.7143** | 0.6825 |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0986 (48pairs) |

## Historical numerical table 3

  |scenario| bike | car | pt | walk |
  |---|---|---|---|---|
  | rain | −19.2 | +15.0 | +5.7 | −1.5 |
  | fare ×2 | −2.5 | +7.0 | −4.3 | −0.2 |
  | combined | −19.2 | +15.0 | +5.4 | −1.2 |

## Historical numerical table 4

  |scenario| equilibrium c* | car share |iterations to convergence|
  |---|---|---|---|
  | baseline | 0.38 | 25.0% | 3 |
  | rain | **0.44** | 38.5% | 5 |
  | fare ×2 | 0.40 | 31.6% | 4 |

## Historical numerical table 5

  | checkpoint | |ΔP_S| | |ΔP_T−ΔP_S| | sign |
  |---|---|---|---|---|
  | v0.3-C(without congestion training) | 0.0067 | 0.0637 | 0.3611 |
  | **v0.3-S1-C(after additional training)** | 0.0215 | 0.0604 | **0.7361** |

## Historical numerical table 6

  |axis| sign before→after | \|ΔP_T−ΔP_S\| before→after |
  |---|---|---|
  | parking_cost | 0.315 → **0.778** | 0.090 → **0.059** |
  | transit_delay | 0.685 → **0.759** | 0.067 → **0.047** |

## Historical numerical table 7

  | checkpoint | \|ΔP_S\| | \|ΔP_T−ΔP_S\| | sign |
  |---|---|---|---|
  | v0.3-S2-C(without disruption training) | 0.1056 | 0.0738 | 0.7778 |
  | **v0.3-S4-C(after additional training)** | **0.1408** | 0.0787 | **0.8333** |

## Historical numerical table 8

|metric| M0 single-axis | M1 joint(K=5) | M2 joint(+K=7) |
|---|---|---|---|
| seen joint KL | 0.0538 | **0.0440** | 0.0441 |
| unseen joint KL (fare×cong) | 0.0954 | 0.0890 | **0.0880** |
| interaction L1 error | 0.0469 | 0.0464 | **0.0453** |
| legacy acc(no regression) | 0.8496 | 0.8496 | 0.8451 |
| legacy KL | 0.0776 | 0.0745 | **0.0735** |

## Historical numerical table 9

|axis| Teacher | C0 pre-S5 | C1 S5 |interpretation|
|---|---|---|---|---|
| transit_delay | 0.73 / **0.97** | 0.37 / **0.90** | 0.35 / **0.84** | Case 3 Teacher and Student mechanisms agree |
| congestion | 0.52 / **0.89** | 0.81 / 0.75 | 1.02 / 0.67 | Case 1 distillation degradation |
| parking_cost | 0.69 / 0.56 | 1.03 / **0.05** | 1.02 / 0.06 | Case 1 mechanism loss |

## Historical numerical table 10

|metric| 100 | 500 | 1000 |
|---|---|---|---|
| planning fallbacks | 49 | 245 | 501(linear in population size, ≈50% is pt→walk) |
| failed trips / stuckAndAbort | 0 / 0 | 0 / 0 | 0 / 0 |
| PT boardings | 12 | 55 | 127(linear) |
| mean trip time (min) | 8.96 | 9.67 | 9.62 |
| road delay (s/passage) | 0.51 | 0.51 | 0.51 |
| congestion(free flow+15s active-mode share) | 0.0 | 0.0002 | 0.0 |
| runtime (s) | 29.4 | 26.5 | 27.6 |
| executed mode share | 0.18/0.33/0.06/0.43 | 0.18/0.29/0.06/0.48 | 0.19/0.29/0.06/0.47 |

## Historical numerical table 11

|metric| S7-W3 (B0) | S8 (R2) | Teacher |
|---|---|---|---|
| PT prob MAE | 0.1336 | **0.1184**(Δ -0.0152, CI excluding 0) | — |
| mean P(PT\|infeasible) | 0.0978 | **0.0599**(Δ -0.0379*) | 0.000 |
| monotonicity pair/triplet | 0.657/0.261 | **0.686/0.304** | 0.800/0.522 |
| sensitivity ΔP_PT | 0.163 | 0.156 | 0.212 |

## Historical numerical table 12

|scenario| car | pt | bike | walk | PT boardings | VKT |
|---|---|---|---|---|---|---|
| C0 baseline | 18.1% | 2.8% | 38.2% | 40.8% | 608 | 21,145 km |
| C1 heavy rain | **36.0%** | **32.4%** | 14.4% | 17.2% | **6,715** | 40,855 km |
| C2 fare ×1.5 | 20.3% | 2.6% | 37.1% | 39.9% | 550 | 23,700 km |
| C3 delay 15min | 19.5% | **0.0%** | 38.1% | 42.4% | 3 | 22,500 km |
| C4 road disruption | **0.2%** | 5.5% | 43.1% | **51.3%** | 1,230 | **168 km** |
| C5 rain+delay | 36.5% | 0.2% | 13.4% | 49.9% | 43 | 41,475 km |

## Historical numerical table 13

|scenario| car | pt | bike | walk | PT boardings | VKT |
|---|---|---|---|---|---|---|
| C0 baseline | 28.3% | 25.3% | 34.0% | 12.4% | 5,613 | 33,939 km |
| C1 heavy rain | **37.5%** | **45.3%** | 13.3% | 3.8% | **9,761** | 42,778 km |
| C2 fare ×1.5 | 29.9% | 25.7% | 32.5% | 11.8% | 5,674 | 35,798 km |
| C3 delay 15min | 29.0% | **10.2%** | 35.8% | **25.0%** | 2,474 | 34,685 km |
| C4 road disruption | **1.8%** | **37.5%** | **41.8%** | 19.0% | 8,537 | **2,943 km** |
| C5 rain+delay | **37.6%** | 18.6% | 24.5% | 19.2% | 4,301 | 42,823 km |
