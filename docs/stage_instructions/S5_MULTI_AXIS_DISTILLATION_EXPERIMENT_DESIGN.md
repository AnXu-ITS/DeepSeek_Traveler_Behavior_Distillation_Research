# S5 joint-context response study

S5 extends single-context supervision to joint perturbations and tests whether probability responses combine as the Teacher predicts. It distinguishes seen joint combinations from held-out combinations and evaluates both endpoint fidelity and interaction behavior. This historical stage contributes supervision sources to the later controlled benchmark, but its own training recipe and comparisons differ from that matched experiment.

This is an English editorial consolidation of the historical document. The [complete original version](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/docs/stage_instructions/S5_MULTI_AXIS_DISTILLATION_EXPERIMENT_DESIGN.md) remains available in Git history. The numerical tables and English code listings retained below are historical records, not newly executed results.

[Current research](<../RESEARCH_DESIGN.md>) · [Training](<../TRAINING.md>) · [Results](<../RESULTS.md>) · [Data and access](<../DATA_SOURCES.md>) · [Model use](<../MODEL_USE.md>)

## Retained English technical listings

```text
rain ∈ {0, 1}
road_congestion ∈ {low, medium, high}
```


```text
fare_multiplier ∈ {1.0, 1.5, 2.0}
road_congestion ∈ {low, medium, high}
```


```text
fare_multiplier ∈ {1.0, 1.5, 2.0}
transit_delay_min ∈ {0, 5, 15, 30}
```


```text
40 personas × 2 trips × 2 joint levels = 160 states / pair
```


```text
160 × 4 = 640 joint states
```


```text
single-axis legacy data: K=3
joint low-SNR data: K=5
causal / benchmark subset: K=7
```


```yaml
sample_id:
persona_id:
trip_id:
split_group_id:
persona_group_id:
baseline_sample_id:
context:
  weather:
  fare_multiplier:
  road_congestion:
  transit_delay:
  parking_cost_multiplier:
  road_disruption:
joint_axes:
  - axis_1
  - axis_2
joint_level_id:
teacher_repeats:
teacher_probability_mean:
teacher_probability_std:
teacher_action:
departure_time_shift:
teacher_noise:
teacher_pairwise_l1:
source:
  single_or_joint: joint
combination_id:
combination_seen_in_training:
```


```text
Student-S3 → Student-S5-Joint
```


```text
before S5 vs after S5
```


```text
scripts/
  generate_joint_teacher_dataset.py
  validate_joint_dataset.py
  train_student_s5_joint.py
  eval_joint_behavior.py
  eval_interaction_effect.py
  eval_unseen_combinations.py

configs/
  student_s5_joint.yaml
  joint_sampling.yaml

data/
  student_s5_joint/

outputs/
  student_s5_joint/
  joint_eval/
  unseen_combination_eval/

reports/
  EXPERIMENT_REPORT_S5_MULTI_AXIS.md
```
