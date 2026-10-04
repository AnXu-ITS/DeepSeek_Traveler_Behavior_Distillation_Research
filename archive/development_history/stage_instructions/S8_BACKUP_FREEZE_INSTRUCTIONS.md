# Historical S8 freeze procedure — deprecated

This procedure describes packaging the first supply-aware checkpoint and its feature extension, provenance and gates. The data were later found to use incorrect active-mode speeds, so the S8 experiment is deprecated. The original tag remains an audit record. New applications should use the corrected S9 model and current guide rather than rerun these historical instructions against frozen output paths.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/stage_instructions/S8_BACKUP_FREEZE_INSTRUCTIONS.md).

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Supply-aware release layout and freeze requirements

```text
releases/
  s8_supply_aware_v1/
    checkpoint/
    config/
    schema/
    normalization/
    accessibility/
    metrics/
    reports/
    data_manifest/
    code_manifest/
    provenance/
    checksums/
    README.md
    FINAL_S8_FREEZE.md
```


```text
releases/s7_w3_generic_core_v1/
```


```text
releases/s8_supply_aware_v1/checkpoint/
```


```text
config/student_s8.yaml
config/training_s8.yaml
config/accessibility_features.yaml
```


```text
schema/input_schema_s8.json
schema/output_schema_s8.json
schema/feature_order_s8.txt
schema/schema_diff_vs_s7.json
```


```text
accessibility/feature_definitions.md
accessibility/feature_schema.json
```


```text
provenance/singapore_supply.json
provenance/osm_manifest.json
provenance/gtfs_manifest.json
```


```text
data_manifest/s8_accessibility_dataset.json
data_manifest/s8_split_manifest.json
```


```text
provenance/teacher_s8.json
```


```text
metrics/final_metrics.json
metrics/comparison_s7_vs_s8.csv
```


```text
reports/S8_SCHEMA_AUDIT.md
reports/EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md
```


```text
reports/S8_EVIDENCE_INDEX.md
```


```text
65e505f
```


```text
freeze: S8 supply-aware traveler agent v1.0
```


```bash
git tag -a s8-supply-aware-v1.0 -m "Freeze S8 supply-aware traveler agent v1.0"
```


```text
checksums/SHA256SUMS.txt
```


```text
Model: S8
Release: Supply-Aware Traveler Agent v1.0
Base: S7-W3 Generic Behavioral Core v1.0
Parameters: 24,562
Schema evolution: Case B
Status: FROZEN
```


```text
Student = frozen S8
N* = 10,000
flowCapacityFactor = 0.3
storageCapacityFactor = 0.3
Singapore supply = frozen Phase B.5 version
PT planner = frozen validated version
```


```text
S7-W3 = Generic Behavioral Core v1.0
S8 = Supply-Aware Traveler Agent v1.0
```


```text
C0 baseline
-> C1 heavy rain
-> C2 PT fare increase
-> C3 transit delay
-> C4 road disruption
-> C5 joint scenario
```
