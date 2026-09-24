# v0_3_S4 historical training report

This development-stage report evaluates Teacher imitation, probability distributions, departure adjustment and scenario-response behavior under the model/data version named in the title. Additional supervision stages must be interpreted using their own splits and selection rules. Historical pointwise or synthetic-context results are not interchangeable with the later 437-state, 398-pair controlled benchmark. The retained numerical tables below preserve the original stage-specific results.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/EXPERIMENT_REPORT_v0_3_S4.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Historical numerical table 1

|perturbation axis| Teacher response \|ΔP_T\| | Teacher variability(pairwise L1) |signal-to-noise ratio|
|---|---|---|---|
| **road_disruption** | **0.1756** | 0.1388 | **1.27** |
| parking_cost_multiplier | 0.0806 | 0.1289 | 0.63 |
| transit_delay | 0.0757 | 0.1483 | 0.51 |
| weather_intensity | 0.0797 | 0.1710 | 0.47 |
| road_congestion | 0.0572 | 0.1325 | 0.43 |
| fare_multiplier | 0.0493 | 0.1451 | 0.34 |

## Historical numerical table 2

| checkpoint | \|ΔP_T\| | \|ΔP_S\| | \|ΔP_T−ΔP_S\| | sign agreement |
|---|---|---|---|---|---|
| v0.3-S2-C(without disruption training) | 0.1756 | 0.1056 | 0.0738 | 0.7778 |
| **v0.3-S4-C(after additional training)** | 0.1756 | **0.1408** | 0.0787 | **0.8333** |

## Historical numerical table 3

|Metric (test)| v0.3-S4-A | v0.3-S4-B | v0.3-S4-C |
|---|---|---|---|
| mode_accuracy | 0.8509 | 0.8509 | **0.8684** |
| KL(P_T‖P_S) | 0.1233 | 0.1373 | **0.0519** |
| probability L1 | 0.3037 | 0.3143 | **0.2303** |
| counterfactual \|dP_T−dP_S\| | 0.0696 | 0.0675 | **0.0551** |
| sign agreement | 0.6852 | 0.6790 | 0.6636 |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0705 |
