# Historical demand-scale experiment

This record varies synthetic demand size under a fixed scenario configuration and reports runtime/loading outcomes. Population counts, completion metrics and event counts retain their original denominators. This development-stage scale check should not be merged with later experiments using different capacity factors, populations or Student versions.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/singapore_phase_b/scale_report.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../docs/RESEARCH_DESIGN.md>) · [Training](<../../docs/TRAINING.md>) · [Results](<../../docs/RESULTS.md>) · [Data and access](<../../docs/DATA_SOURCES.md>) · [Model use](<../../docs/MODEL_USE.md>)

## Historical numerical table 1

|metric| 100 | 500 | 1000 |anomaly classification|
|---|---|---|---|---|
| routing failures (planning fallbacks total) | 49 | 245 | 501 | — |
| failed trips (MATSim) | 0 | 0 | 0 | — |
| PT boardings | 12 | 55 | 127 | — |
| mean trip time (min) | 8.96 | 9.67 | 9.62 | — |
| road delay (s/passage) | 0.51 | 0.51 | 0.51 | — |
| congestion (active-mode share) | 0.0 | 0.0002 | 0.0 | — |
| runtime (s) | 29.4 | 26.5 | 27.6 | — |
| mode share (bike/car/pt/walk) | 0.18/0.33/0.06/0.43 | 0.18/0.29/0.06/0.48 | 0.19/0.29/0.06/0.47 | — |
| stuckAndAbort | 0 | 0 | 0 | — |
