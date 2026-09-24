# Historical cost and time estimates

This document records planning estimates for Teacher requests, dataset expansion and experiment time. Historical endpoint/provider assumptions and projected throughput must not be interpreted as current prices or measured billed amounts. The current results distinguish successful-request latency, projected Teacher population costs, local Student inference and end-to-end simulation construction.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/archive/legacy_20260821/root/COST_AND_TIME_ESTIMATE.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Historical numerical table 1

|configuration|population size| states |request count| time(4 workers, gateway available) | time(gateway variability +50%) | cost(v3level → high-end) |
|---|---|---|---|---|---|---|
| B. minimal sufficient setting(recommended next step): 20 personas × 3 trips × {weather, fare} | 60 pairs × 8 states | 480 | 1440 | ~4.5 h | ~7 h | $0.35 → $3.0 |
| A. development verification complete: 24 personas × 4 trips × {weather, fare, congestion, delay} | 96 pairs × 15 states | 1440 | 4320 | ~13 h | ~20 h | $1.0 → $9.0 |
| C. Phase 0–7 all axes: 20 personas × 3 trips × all 6 axis | 60 pairs × 19 states | 1140 | 3420 | ~10.5 h | ~16 h | $0.8 → $7.1 |

## Historical numerical table 2

|stage|description|estimated time|
|---|---|---|
| Phase 8 | MATSimAdapter(Student output → plans/attributes) | 2–5 days |
| Phase 9 | Population-scale simulation(public example scenario + distillation agents) | 1–3 days(+simulation execution, several minutes per run–several hours CPU) |
| Phase 10 |network feedback loop| 2–5 days |
