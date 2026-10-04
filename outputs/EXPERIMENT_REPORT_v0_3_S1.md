# v0_3_S1 historical training report

This development-stage report evaluates Teacher imitation, probability distributions, departure adjustment and scenario-response behavior under the model/data version named in the title. Additional supervision stages must be interpreted using their own splits and selection rules. Historical pointwise or synthetic-context results are not interchangeable with the later 437-state, 398-pair controlled benchmark. The retained numerical tables below preserve the original stage-specific results.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/EXPERIMENT_REPORT_v0_3_S1.md).

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Teacher perturbation response and sampling variability

|perturbation axis| Teacher response \|ΔP_T\| | Teacher sampling variability(pairwise L1, K=3) |
|---|---|---|
| weather_intensity | 0.0797 | 0.1710 |
| fare_multiplier | 0.0493 | 0.1451 |
| road_congestion | **0.0572** | 0.1325 |

## Congestion response before and after additional supervision

| checkpoint | \|ΔP_T\| | \|ΔP_S\| | \|ΔP_T−ΔP_S\| | sign agreement |
|---|---|---|---|---|
| v0.3-A(without congestion training) | 0.0572 | 0.0017 | 0.0578 | 0.4167 |
| v0.3-C(without congestion training) | 0.0572 | 0.0067 | 0.0637 | 0.3611 |
| **v0.3-S1-A(after additional training)** | 0.0572 | 0.0186 | 0.0465 | 0.6250 |
| **v0.3-S1-C(after additional training)** | 0.0572 | 0.0215 | 0.0604 | **0.7361** |

## Student test performance after congestion supervision

|Metric (test)| v0.3-S1-A | v0.3-S1-B | v0.3-S1-C |
|---|---|---|---|
| mode_accuracy | 0.8472 | 0.8056 | **0.8750** |
| KL(P_T‖P_S) | 0.1233 | 0.1288 | **0.0659** |
| probability L1 | 0.3043 | 0.3121 | **0.2511** |
| counterfactual \|dP_T−dP_S\| | 0.0563 | 0.0547 | 0.0548 |
| sign agreement | 0.6667 | 0.6869 | **0.7424** |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0720 |
