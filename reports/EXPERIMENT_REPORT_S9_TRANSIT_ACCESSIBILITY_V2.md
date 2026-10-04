# S9 accessibility adaptation results

S9, called SA-Student in the current manuscript, initializes from frozen S7-W3 and adapts to corrected Singapore PT accessibility inputs. It has 24,562 parameters, 336 states split 234/51/51, and 1,502 valid Teacher queries. Training selects epoch 17. This historical adaptation comparison includes inherited generic and auxiliary supervision, so it is not the later matched objective ablation. The 51-state accessibility test and 226-state single-context test are distinct evaluation sets.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/reports/EXPERIMENT_REPORT_S9_TRANSIT_ACCESSIBILITY_V2.md).

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Teacher transit probability by corrected accessibility class

| class | n | Teacher P(PT) |
|---|---|---|
| A_excellent | 79 | 0.448 |
| B_good | 16 | 0.429 |
| C_moderate | 77 | 0.372 |
| D_poor | 84 | 0.189 |
| E_infeasible | 80 | 0.000 |

## Corrected supply-aware model performance

|metric| B0 S7-W3 | S9 | Teacher |
|---|---|---|---|
| PT prob MAE | 0.1750 | **0.1346** | — |
| mean P(PT\|infeasible) | 0.2924 | **0.2130** | 0.000 |
| FVR rate | 0.3333 | **0.0833** | 0.000 |
| pair monotonicity | 0.6316 | **0.6579** | 0.7895 |
| sensitivity ΔP_PT | 0.0845 | **0.1527** | 0.4480 |

## Generic-capability regression checks

|validation check|value|interpretation|
|---|---|---|
| legacy accuracy drop | -4.43 pp | ✅ ≤ 1 pp |
| legacy KL change | -26.04% | ✅ ≤ +10% |
| seen joint KL change | -3.33% | ✅ ≤ +10% |
| unseen joint KL change | -36.45% | ✅ ≤ +10% |
