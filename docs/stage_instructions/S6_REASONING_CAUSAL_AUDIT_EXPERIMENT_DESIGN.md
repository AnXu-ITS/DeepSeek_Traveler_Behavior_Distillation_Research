# S6 mechanism diagnostic study

S6 compares natural perturbations with states that change context labels, numerical travel attributes or both. These comparisons test what information Teacher and Student responses follow, and whether distillation preserves the observed mechanism. The contrasts are diagnostic model experiments; they are not field estimates of causal effects. Mechanism states later contribute a separate source to the controlled response benchmark.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/stage_instructions/S6_REASONING_CAUSAL_AUDIT_EXPERIMENT_DESIGN.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../RESEARCH_DESIGN.md>) · [Training](<../TRAINING.md>) · [Results](<../RESULTS.md>) · [Data and access](<../DATA_SOURCES.md>) · [Model use](<../MODEL_USE.md>)

## Retained English technical listings

```text
congestion=high → car↓
```


```text
congestion=high
car travel time unchanged
```


```text
congestion
→ car travel time / reliability deterioration
→ car utility decline
→ car probability decline
```


```text
road_congestion = low
car_travel_time = 20 min
```


```text
road_congestion = high
car_travel_time = 40 min
```


```text
road_congestion = high
car_travel_time = 20 min
```


```text
road_congestion = low
car_travel_time = 40 min
```


```text
rain = true
congestion = high
car travel time = baseline
```


```text
Rain
  ↓
Student: car↑
  ↓
MATSim
  ↓
Congestion↑
  ↓
Student re-decision
  ↓
car partially↓ / PT↑ / departure shift
```


```text
rain = true
congestion = observed_high
```


```text
Context
  ↓
Mediator
  ↓
Behavior
```


```text
Baseline
Natural
Broken
Mediator-only
```


```text
Single-axis Student
        ↓
Multi-axis Student
        ↓
Causal-aware Student (if needed)
```


```text
scripts/
  generate_causal_audit_states.py
  run_teacher_causal_audit.py
  eval_causal_mechanism.py
  eval_shortcut_ratio.py
  eval_sequential_reasoning.py

configs/
  causal_audit.yaml

data/
  causal_audit/

outputs/
  causal_audit/
  reasoning_generalization/

reports/
  EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md
```
