# S7 mechanism-aware training

S7 supplements the generic Student with mechanism and between-person response supervision while checking retention of earlier endpoint and context behavior. The W3 model was frozen as the generic behavioral predecessor. Its inherited supervision matters when interpreting S9: adaptation from this checkpoint is not a matched comparison against a randomly initialized baseline. Current controlled objectives deliberately exclude these auxiliary updates.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/stage_instructions/S7_MECHANISM_AWARE_FINETUNING_INSTRUCTIONS.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Retained English technical listings

```text
parking_cost label ↑
    ↓
car probability ↓
```


```text
outputs/
  student_s3_c/
  student_s5_joint/
  student_s7_mechanism/
```


```text
A = baseline
B = natural intervention
C = broken-path intervention
D = mediator-only intervention
```


```text
road_congestion
    ↓
car travel time / reliability
    ↓
car generalized attractiveness
    ↓
P(car)
```


```yaml
road_congestion: low
car_travel_time: baseline
car_reliability: baseline
```


```yaml
road_congestion: high
car_travel_time: increased
car_reliability: decreased
```


```yaml
road_congestion: high
car_travel_time: baseline
car_reliability: baseline
```


```yaml
road_congestion: low
car_travel_time: increased
car_reliability: decreased
```


```text
parking_cost
    ↓
car monetary / generalized cost
    ↓
car attractiveness
    ↓
P(car)
```


```yaml
parking_cost_multiplier: 1.0
car_monetary_cost: baseline
```


```yaml
parking_cost_multiplier: high
car_monetary_cost: increased
```


```yaml
parking_cost_multiplier: high
car_monetary_cost: baseline
```


```yaml
parking_cost_multiplier: 1.0
car_monetary_cost: increased
```


```text
broken effect = 0
```


```text
Student broken effect ≈ Teacher broken effect
```


```text
E_natural ≈ 0
```


```text
R_shortcut
R_mediator
```


```text
src/traveler_distillation/student/mechanism_dataset.py
```


```text
MechanismQuadrupletDataset
```


```python
{
    "baseline": A,
    "natural": B,
    "broken": C,
    "mediator": D,
    "teacher_probs_A": ...,
    "teacher_probs_B": ...,
    "teacher_probs_C": ...,
    "teacher_probs_D": ...,
    "axis": "congestion" | "parking_cost",
    "persona_id": ...,
    "trip_id": ...
}
```


```yaml
lambda_mechanism: 0.5
lambda_broken: 0.5
```


```text
λm = 0
λb = 0
```


```text
λm = 0.25
λb = 0.25
```


```text
λm = 0.5
λb = 0.5
```


```text
λm = 1.0
λb = 1.0
```


```text
legacy single-axis
+
S5 seen joint
+
S7 mechanism quadruplets
```


```text
legacy single-axis : seen joint : mechanism
= 2 : 1 : 1
```


```text
1 : 1 : 1
```


```text
S5 LR × 0.25
```


```text
S5 LR × 0.5
```


```text
validation total loss
+
mechanism effect gap
+
legacy KL
```


```text
C0 pre-S5
C1 S5
C2 S7
Teacher
```


```text
Gap ↓
```


```text
accuracy drop ≤ 1.0 percentage point
KL increase ≤ 10%
probability L1 increase ≤ 10%
```


```text
seen joint KL increase ≤ 10%
unseen joint KL increase ≤ 10%
```


```text
R_shortcut ≈ 1.02
R_mediator ≈ 0.67
```


```text
R_shortcut ≈ 0.52
R_mediator ≈ 0.89
```


```text
R_shortcut: 0.55–0.80
R_mediator: 0.75–0.95
```


```text
R_shortcut ≈ 1.02
R_mediator ≈ 0.06
```


```text
R_shortcut ≈ 0.69
R_mediator ≈ 0.56
```


```text
R_shortcut: 0.65–0.85
R_mediator: 0.35–0.65
```


```text
A = mechanism only
B = broken only
C = both
```


```text
Persona + Context + Supply
        ↓
Perceived mode attributes
        ↓
Generalized utility / latent preference
        ↓
Mode distribution
```


```text
src/traveler_distillation/student/
  mechanism_dataset.py
  mechanism_losses.py

scripts/
  build_s7_mechanism_dataset.py
  train_student_s7.py
  eval_s7_causal_repair.py
  eval_s7_regression.py
  make_s7_report.py

configs/
  student_s7_w1.yaml
  student_s7_w2.yaml
  student_s7_w3.yaml

data/
  student_s7_mechanism/

outputs/
  student_s7_w1/
  student_s7_w2/
  student_s7_w3/
  s7_causal_eval/
  s7_regression/

reports/
  EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md
```


```text
EXPERIMENT_REPORT_S5_MULTI_AXIS.md
EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md
```


```text
L_mechanism
L_broken
```


```text
all tests pass
training exits normally
loss finite
quadruplet linking correct
```


```text
EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md
```
