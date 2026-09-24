# v0_3 historical training report

This development-stage report evaluates Teacher imitation, probability distributions, departure adjustment and scenario-response behavior under the model/data version named in the title. Additional supervision stages must be interpreted using their own splits and selection rules. Historical pointwise or synthetic-context results are not interchangeable with the later 437-state, 398-pair controlled benchmark. The retained numerical tables below preserve the original stage-specific results.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/EXPERIMENT_REPORT_v0_3.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Historical numerical table 1

| mode | mean P | std | n |
|---|---|---|---|
| walk | 0.2725 | 0.3222 | 40 |
| bike | 0.336 | 0.2627 | 16 |
| pt | 0.3912 | 0.3818 | 40 |
| car | 0.5767 | 0.2958 | 14 |

## Historical numerical table 2

| metric | A | B | C |
|---|---|---|---|
| mode accuracy | 0.7708 | 0.8125 | 0.8125 |
| KL(P_T || P_S) | 0.3816 | 0.1440 | 0.1092 |
| probability L1 | 0.7605 | 0.3260 | 0.2860 |
| departure MAE (min) | 9.2489 | 3.2117 | 2.0945 |
| departure sign agreement | 0.4792 | 0.6875 | 0.7500 |

## Historical numerical table 3

| metric | A | B | C |
|---|---|---|---|
| counterfactual \|dP_T - dP_S\| | 0.0661 | 0.0596 | 0.0539 |
| counterfactual sign agreement | 0.6508 | 0.7143 | 0.6825 |
| heterogeneity \|dP_T - dP_S\| | n/a | n/a | 0.0986 |

## Historical numerical table 4

| persona | n | A acc | A L1 | B acc | B L1 | C acc | C L1 |
|---|---|---|---|---|---|---|---|
| P000001 | 16 | 0.4375 | 0.6995 | 0.9375 | 0.3178 | 0.9375 | 0.2626 |
| P000004 | 16 | 0.9375 | 0.7766 | 0.8125 | 0.3013 | 0.75 | 0.2776 |
| P000009 | 16 | 0.9375 | 0.8054 | 0.6875 | 0.3589 | 0.75 | 0.3179 |

## Retained English technical listings

```powershell
.venv\Scripts\python.exe scripts\run_v0_3_experiment.py --dataset data/student_v0_3/aggregated_teacher_dataset.jsonl --repeats data/student_v0_3/repeat_records.jsonl --outputs outputs
```
