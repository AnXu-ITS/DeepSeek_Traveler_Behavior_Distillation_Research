# v0_3_S3 historical training report

This development-stage report evaluates Teacher imitation, probability distributions, departure adjustment and scenario-response behavior under the model/data version named in the title. Additional supervision stages must be interpreted using their own splits and selection rules. Historical pointwise or synthetic-context results are not interchangeable with the later 437-state, 398-pair controlled benchmark. The retained numerical tables below preserve the original stage-specific results.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/EXPERIMENT_REPORT_v0_3_S3.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Historical numerical table 1

|perturbation axis| Teacher response \|ΔP_T\| |Teacher variability|signal-to-noise ratio|
|---|---|---|---|
| road_disruption | 0.1422 | 0.1472 | **0.97** |
| weather_intensity | 0.1023 | 0.1386 | 0.74 |
| parking_cost_multiplier | 0.0660 | 0.1476 | 0.45 |
| road_congestion | 0.0434 | 0.1305 | 0.33 |
| transit_delay | 0.0453 | 0.1414 | 0.32 |
| fare_multiplier | 0.0358 | 0.1345 | 0.27 |

## Historical numerical table 2

|axis|metric| before(S4-C) | after(S3-C) |
|---|---|---|---|
| road_congestion | sign | 0.609 | **0.719** |
| transit_delay | sign | 0.831 | **0.868** |
| road_disruption | sign | 0.972 | 0.764 |
| fare_multiplier | sign | 0.727 | 0.556 |
| parking_cost | sign | 0.771 | 0.678 |
| weather | sign | 0.707 | 0.693 |

## Historical numerical table 3

|Metric (test)| v0.3-S3-A | v0.3-S3-B | v0.3-S3-C |
|---|---|---|---|
| mode_accuracy | 0.8673 | **0.8850** | 0.8496 |
| KL(P_T‖P_S) | 0.0880 | **0.0497** | 0.0776 |
| probability L1 | 0.2897 | **0.1872** | 0.2646 |
| counterfactual \|dP_T−dP_S\| | 0.0627 | 0.0598 | **0.0516** |
| sign agreement | 0.6301 | 0.6745 | **0.7048** |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0955 (561pairs) |

## Historical numerical table 4

| persona | n | mode_acc | prob_L1 |
|---|---|---|---|
| P000002 | 38 | 0.9211 | 0.2543 |
| P000008 | 37 | 1.0000 | 0.1200 |
| P000009 | 38 | 0.7632 | 0.2517 |
| P000015 | 37 | 0.9459 | 0.2975 |
| P000016 | 38 | 0.8421 | 0.3001 |
| P000018 | 38 | 0.6316 | 0.3611 |
