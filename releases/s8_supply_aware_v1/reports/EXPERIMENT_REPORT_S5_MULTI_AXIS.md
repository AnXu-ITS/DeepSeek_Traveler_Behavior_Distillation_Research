# S5 joint-context response study

S5 extends single-context supervision to joint perturbations and tests whether probability responses combine as the Teacher predicts. It distinguishes seen joint combinations from held-out combinations and evaluates both endpoint fidelity and interaction behavior. This historical stage contributes supervision sources to the later controlled benchmark, but its own training recipe and comparisons differ from that matched experiment.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/releases/s8_supply_aware_v1/reports/EXPERIMENT_REPORT_S5_MULTI_AXIS.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Historical numerical table 1

| model | seen joint KL | seen joint L1 | unseen joint KL | unseen joint L1 |
|---|---|---|---|---|
| M0 | 0.0538 | 0.2065 | 0.0954 | 0.2725 |
| M1 | 0.0440 | 0.1860 | 0.0890 | 0.2558 |
| M2 | 0.0441 | 0.1847 | 0.0880 | 0.2538 |

## Historical numerical table 2

| combo | seen | M0 KL / L1 | M1 KL / L1 | M2 KL / L1 |
|---|---|---|---|---|
| rain_x_congestion | yes | 0.0518 / 0.2278 | 0.0442 / 0.2150 | 0.0442 / 0.2137 |
| fare_x_transit_delay | yes | 0.0580 / 0.1852 | 0.0575 / 0.1815 | 0.0574 / 0.1802 |
| road_disruption_x_congestion | yes | 0.0515 / 0.2066 | 0.0302 / 0.1614 | 0.0308 / 0.1602 |
| fare_x_congestion | **no (holdout)** | 0.0954 / 0.2725 | 0.0890 / 0.2558 | 0.0880 / 0.2538 |

## Historical numerical table 3

| model | mean interaction L1 error | linked states |
|---|---|---|
| M0 | 0.0469 | 94/96 |
| M1 | 0.0464 | 94/96 |
| M2 | 0.0453 | 94/96 |

## Historical numerical table 4

| combo | M0 | M1 | M2 |
|---|---|---|---|
| rain_x_congestion | 0.0563 | 0.0553 | 0.0539 |
| fare_x_transit_delay | 0.0361 | 0.0348 | 0.0363 |
| road_disruption_x_congestion | 0.0560 | 0.0570 | 0.0555 |
| fare_x_congestion | 0.0385 | 0.0376 | 0.0346 |

## Historical numerical table 5

| model | mode acc | KL | prob L1 |
|---|---|---|---|
| M0 | 0.8496 | 0.0776 | 0.2646 |
| M1 | 0.8496 | 0.0745 | 0.2575 |
| M2 | 0.8451 | 0.0735 | 0.2560 |
