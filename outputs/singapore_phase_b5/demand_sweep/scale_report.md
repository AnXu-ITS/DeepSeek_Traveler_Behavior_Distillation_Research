# Historical demand-scale experiment

This record varies synthetic demand size under a fixed scenario configuration and reports runtime/loading outcomes. Population counts, completion metrics and event counts retain their original denominators. This development-stage scale check should not be merged with later experiments using different capacity factors, populations or Student versions.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/outputs/singapore_phase_b5/demand_sweep/scale_report.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Historical numerical table 1

|metric| 1000 | 2000 | 5000 | 10000 | 40000 |
|---|---|---|---|---|---|
| car VKT (km) | 3,674 | 7,052 | 18,206 | 36,196 | 142,538 |
| road delay (s/passage) | 0.52 | 0.53 | 0.52 | 0.52 | 0.54 |
| slow-passage share | 0.0001 | 0.0004 | 0.0003 | 0.0004 | 0.0005 |
| links delay>15s share | 0.02% | 0.08% | 0.09% | 0.08% | 0.07% |
| car travel time (min) | 8.26 | 8.20 | 8.49 | 8.59 | 8.61 |
| failed trips | 3 | 2 | 10 | 18 | 276(0.7%, missed single service; wait until endTime) |
| stuckAndAbort | 3 | 2 | 10 | 18 | 384(~1%, as above, awaiting cleanup) |
