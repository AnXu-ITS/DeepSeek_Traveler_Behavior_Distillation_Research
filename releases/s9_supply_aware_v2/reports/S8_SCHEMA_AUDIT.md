# Supply-aware schema audit

This audit checks the six-field PT-supply extension of the generic Student. The extended architecture has 12 alternative numerical inputs, a 20-wide alternative-encoder input after the mode embedding, and 24,562 parameters. Existing weights and normalizers are preserved at initialization; new input columns start at zero. Schema compatibility does not validate the speed assumptions in the underlying data. S8 data were subsequently deprecated; the compatible S9 checkpoint uses corrected inputs.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/releases/s9_supply_aware_v2/reports/S8_SCHEMA_AUDIT.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Retained English technical listings

```text
pt_feasible            (0/1)
pt_egress_time_min
pt_wait_time_min
pt_in_vehicle_time_min
pt_transfer_time_min
pt_coverage_ratio      [0,1]
```
