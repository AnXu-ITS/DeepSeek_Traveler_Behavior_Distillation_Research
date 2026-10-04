# v0_2 historical training report

This development-stage report evaluates Teacher imitation, probability distributions, departure adjustment and scenario-response behavior under the model/data version named in the title. Additional supervision stages must be interpreted using their own splits and selection rules. Historical pointwise or synthetic-context results are not interchangeable with the later 437-state, 398-pair controlled benchmark. The retained numerical tables below preserve the original stage-specific results.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/EXPERIMENT_REPORT_v0_2.md).

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Student prediction and departure-time performance

|metric| v0.2-A | v0.2-B | v0.2-C |
|---|---|---|---|
| mode_accuracy | 0.7917 | 0.7917 | 0.7917 |
| cross_entropy | 0.575 | 0.625 | 0.592 |
| KL(P_T‖P_S) | 0.0984 | 0.1160 | **0.0787** |
| probability L1 | 0.3031 | 0.3503 | **0.2891** |
| departure MAE (min) | 3.11 | 2.90 | 3.21 |
| departure sign agreement | 0.833 | 0.875 | 0.792 |

## Counterfactual probability-response error

|version| mean \|ΔP_T−ΔP_S\| |pair count|
|---|---|---|
| v0.2-A | **0.1269** | 21 |
| v0.2-B | 0.1564 | 21 |
| v0.2-C | 0.1461 | 21 |
