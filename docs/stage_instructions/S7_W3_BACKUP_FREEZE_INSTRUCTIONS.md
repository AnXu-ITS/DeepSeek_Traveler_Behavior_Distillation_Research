# S7-W3 model freeze procedure

This historical procedure collects the selected generic Student checkpoint, configuration, schema, vocabularies, normalization, supervision lineage, test metrics and reproducibility checks into a frozen release. It requires output isolation and checksum checks and separates archived results from new experiments. S7-W3 is the predecessor used for the later S9 supply adaptation; it is not a replacement for the current controlled models.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/stage_instructions/S7_W3_BACKUP_FREEZE_INSTRUCTIONS.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../RESEARCH_DESIGN.md>) · [Training](<../TRAINING.md>) · [Results](<../RESULTS.md>) · [Data and access](<../DATA_SOURCES.md>) · [Model use](<../MODEL_USE.md>)

## Retained English technical listings

```text
S8-...
S9-...
```


```text
releases/
  s7_w3_generic_core_v1/
    checkpoint/
    config/
    schema/
    normalization/
    metrics/
    reports/
    data_manifest/
    code_manifest/
    checksums/
    README.md
    FINAL_S7_W3_FREEZE.md
```


```text
releases/s7_w3_generic_core_v1/checkpoint/model.pt
```


```text
optimizer state
scheduler state
training state
```


```text
student config
training config
loss weights
replay ratio
learning rate
batch size
seed
early stopping config
split config
feature config
```


```text
config/student_s7_w3.yaml
config/training_s7_w3.yaml
```


```text
schema/input_schema.json
schema/output_schema.json
schema/feature_order.txt
```


```text
normalization/normalization.json
normalization/mode_mapping.json
normalization/category_mapping.json
```


```text
S3 legacy dataset
S5 joint dataset
S6 causal audit dataset
S7 mechanism dataset
```


```text
data_manifest/data_manifest.json
```


```text
git commit hash
git branch
git status
python version
pytorch version
java version
MATSim version
```


```text
code_manifest/git_commit.txt
code_manifest/git_status.txt
code_manifest/environment.txt
code_manifest/uncommitted.patch
```


```text
freeze: S7-W3 generic behavioral core v1.0
```


```text
s7-w3-generic-core-v1.0
```


```text
metrics/final_metrics.json
```


```text
42
7
123
2024
```


```text
seed=42
seed=7
seed=123
seed=2024
```


```text
metrics/seed_robustness.csv
```


```text
EXPERIMENT_REPORT_S5_MULTI_AXIS.md
EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md
EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md
```


```text
reports/
```


```text
checksums/SHA256SUMS.txt
```


```text
Model: S7-W3
Release: Generic Behavioral Core v1.0
Status: FROZEN
```


```bash
git tag -a s7-w3-generic-core-v1.0 -m "Freeze S7-W3 generic behavioral core v1.0"
```


```text
S8 scripts must reject output paths inside releases/s7_w3_generic_core_v1/
```


```text
S7-W3
=
Generic Behavioral Core v1.0
=
FROZEN
```


```text
load S7-W3
→ initialize new S8 experiment
→ save to separate output/release
```
