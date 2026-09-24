# v0_3_S2 historical training report

This development-stage report evaluates Teacher imitation, probability distributions, departure adjustment and scenario-response behavior under the model/data version named in the title. Additional supervision stages must be interpreted using their own splits and selection rules. Historical pointwise or synthetic-context results are not interchangeable with the later 437-state, 398-pair controlled benchmark. The retained numerical tables below preserve the original stage-specific results.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/EXPERIMENT_REPORT_v0_3_S2.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../docs/RESEARCH_DESIGN.md>) · [Training](<../docs/TRAINING.md>) · [Results](<../docs/RESULTS.md>) · [Data and access](<../docs/DATA_SOURCES.md>) · [Model use](<../docs/MODEL_USE.md>)

## Historical numerical table 1

|perturbation axis| Teacher response \|ΔP_T\| | Teacher variability(pairwise L1) |signal-to-noise ratio|
|---|---|---|---|
| parking_cost_multiplier | 0.0806 | 0.1289 | **0.63** |
| transit_delay | 0.0757 | 0.1483 | 0.51 |
| weather_intensity | 0.0797 | 0.1710 | 0.47 |
| road_congestion | 0.0572 | 0.1325 | 0.43 |
| fare_multiplier | 0.0493 | 0.1451 | 0.34 |

## Historical numerical table 2

|axis|metric| before(S1-C) | after(S2-C) |
|---|---|---|---|
| parking_cost | \|ΔP_S\| | 0.0098 | **0.0577** (Teacher 0.0806) |
| parking_cost | \|ΔP_T−ΔP_S\| | 0.0901 | **0.0587** |
| parking_cost | sign | 0.3148 | **0.7778** |
| transit_delay | \|ΔP_S\| | 0.0982 | 0.0954 (Teacher 0.0757) |
| transit_delay | \|ΔP_T−ΔP_S\| | 0.0665 | **0.0466** |
| transit_delay | sign | 0.6852 | **0.7593** |

## Historical numerical table 3

|Metric (test)| v0.3-s2-A | v0.3-s2-B | v0.3-s2-C |
|---|---|---|---|
| mode_accuracy | 0.8519 | 0.8519 | **0.8611** |
| KL(P_T‖P_S) | 0.1202 | 0.1231 | **0.0529** |
| probability L1 | 0.2897 | 0.2945 | **0.2357** |
| counterfactual \|dP_T−dP_S\| | 0.0642 | 0.0617 | **0.0460** |
| sign agreement | 0.6863 | 0.6078 | **0.6928** |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0699 |

## Retained English technical listings

```powershell
.\.venv\Scripts\python.exe scripts\extend_teacher_dataset.py `
  --dataset data\student_v0_3_s1\aggregated_teacher_dataset.jsonl `
  --axes transit_delay,parking_cost_multiplier --output data\student_v0_3_s2 --workers 6

.\.venv\Scripts\python.exe scripts\run_v0_3_s1_experiment.py --tag s2 --outputs outputs

.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s2\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s1_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml   # before
.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s2\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s2_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml   # after
```
