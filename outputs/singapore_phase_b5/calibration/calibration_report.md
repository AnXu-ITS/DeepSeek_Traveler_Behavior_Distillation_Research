# Effective-capacity calibration record

This historical study examines effective flow/storage settings for the synthetic demand placed on the Singapore network. The resulting factors are scenario configuration choices. They do not establish a calibrated reproduction of observed Singapore congestion. Demand, stuck events, unfinished legs and travel times should be interpreted together under the stated population and mode representation.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/singapore_phase_b5/calibration/calibration_report.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Historical numerical table 1

|configuration| slow-passage | links>15s | mean delay (s) | car travel (min) | car failure rate |interpretation|
|---|---|---|---|---|---|---|
| gc35(approach×0.35) | 0.02% | 0.02% | 0.52 | 8.6 | 0.2% | free-flow(current dominant bottleneck 507 items endTime queueing) |
| gc45 | 0.01% | 0.03% | 0.52 | 8.42 | 0.2% | free-flow |
| gc55 | 0.04% | 0.06% | 0.53 | 8.64 | 0.2% | free-flow |
| sample ×0.12 | 0.53% | 0.88% | 0.83 | 17.2 | 0.2% | heavy congestion(in-vehicle time×2, transit slows down→pt sharp increase in missed connections) |
| sample ×0.2 | 0.30% | 0.54% | 0.69 | 10.7 | 0.2% | in-heavy |
| **sample ×0.3** | **0.17%** | **0.36%** | **0.60** | **9.68(+13%)** | **0.2%** | **✅ light-moderate, sufficient dynamic capacity margin** |
| sample ×0.4 | 0.11% | 0.22% | 0.56 | 9.09 | 0.2% |light|
| sample ×0.5 | 0.08% | 0.14% | 0.54 | 8.7 | 0.2% |light|
| 0.6 / 0.7 + gc45 | ≤0.05% | ≤0.07% | ~0.53 | 8.7–8.9 | 0.2% |light|
