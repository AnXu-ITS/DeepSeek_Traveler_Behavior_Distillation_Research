# Historical S8 accessibility experiment — deprecated

This record describes the first supply-aware accessibility adaptation, including data splits, inherited generic weights, added PT features, loss settings and validation gates. Its active-mode travel-time inputs were later found to be incorrect. Numerical results below are preserved for historical inspection only. They must not be used as final S9 or current-paper evidence. The corrected S9 release and current training guide supersede this experiment for model use.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/stage_instructions/S8_TRANSIT_ACCESSIBILITY_TRAINING_INSTRUCTIONS.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../RESEARCH_DESIGN.md>) · [Training](<../TRAINING.md>) · [Results](<../RESULTS.md>) · [Data and access](<../DATA_SOURCES.md>) · [Model use](<../MODEL_USE.md>)

## Retained English technical listings

```text
transit_accessibility = low/medium/high
```


```text
OSM + GTFS
    ↓
PT itinerary
    ↓
Transit Accessibility features
    ↓
Teacher response
    ↓
Student adaptation
```


```text
reports/S8_SCHEMA_AUDIT.md
```


```text
pt_feasible
pt_access_time_min
pt_egress_time_min
pt_wait_time_min
pt_in_vehicle_time_min
pt_transfer_count
pt_transfer_time_min
pt_total_door_to_door_time_min
pt_coverage_ratio
pt_generalized_cost
```


```text
pt_direct
pt_service_frequency
pt_reliability
pt_walk_distance
```


```text
[0,1]
```


```text
300–800 unique OD-context states
```


```text
access = 3
egress = 4
wait = 4
transfer = 0
coverage = 0.92
door_to_door = 28
```


```text
access = 8
egress = 7
wait = 8
transfer = 1
coverage = 0.75
door_to_door = 42
```


```text
access = 15
egress = 12
wait = 12
transfer = 2
coverage = 0.48
door_to_door = 61
```


```text
pt_feasible = 0
```


```text
K = 3
```


```text
K = 5
```


```text
train: OD set A
test: completely unseen OD set B
```


```text
legacy S3
+
S5 joint
+
S7 mechanism
+
S8 accessibility
```


```text
2 : 1 : 1 : 1
```


```text
2 : 1 : 1 : 2
```


```text
PT good
→ PT medium
→ PT poor
```


```text
student_intended_mode
pt_feasible
pt_access_time
pt_egress_time
pt_wait_time
pt_transfer_count
pt_coverage_ratio
routing_success
executed_mode
fallback_reason
```


```text
legacy accuracy drop ≤ 1 pp
legacy KL increase ≤ 10%
seen joint KL increase ≤ 10%
unseen joint KL increase ≤ 10%
```


```text
50–200 OD
```


```text
GTFS
→ accessibility vector
→ S8 inference
```


```text
src/traveler_distillation/accessibility/
  gtfs_accessibility.py
  accessibility_features.py
  accessibility_dataset.py

scripts/
  audit_s8_schema.py
  build_singapore_accessibility_dataset.py
  label_s8_teacher.py
  train_student_s8.py
  eval_s8_accessibility.py
  eval_s8_unseen_od.py
  eval_s8_regression.py
  make_s8_report.py

configs/
  student_s8.yaml
  accessibility_features.yaml

data/
  singapore_accessibility/

outputs/
  student_s8/
  s8_accessibility_eval/
  s8_regression/

reports/
  S8_SCHEMA_AUDIT.md
  EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md
```


```text
S7-W3
Generic Behavioral Core v1.0
        │
        └────── initialization ──────→ S8
                                      │
                                      ↓
                        Supply-Aware Traveler Agent
```
