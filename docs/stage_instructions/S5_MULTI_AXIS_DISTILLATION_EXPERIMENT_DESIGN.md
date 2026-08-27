# Multi-Axis Behavioral Distillation Experiment Design
## 多轴联合补训实验设计

**项目主线**：DeepSeek V4 Pro Teacher → Lightweight Traveler Agent → MATSim  
**实验阶段建议名称**：S5 — Multi-Axis & Low-SNR Refinement  
**目标**：从“单轴行为弹性蒸馏”升级为“多条件联合行为响应蒸馏”，同时避免继续无效扩展 persona 数量。  
**建议执行顺序**：Teacher 联合条件数据 → 数据审计 → Student 补训 → seen/unseen combination evaluation → 与单轴模型对比

---

# 1. 当前问题定义

现有 Student 已覆盖 6 个单轴扰动：

1. weather
2. fare
3. road_congestion
4. transit_delay
5. parking_cost
6. road_disruption

当前实验已经证明：

- Student 可以学习单轴 counterfactual response；
- elasticity / heterogeneity supervision 对未见 persona 有效；
- 不同扰动轴的可蒸馏质量与 Teacher signal-to-noise ratio（SNR）存在稳定关联；
- road_disruption / weather 相对强，fare / congestion 相对弱；
- 模型已经可以进入 MATSim，并产生 population-scale 和 network feedback 结果。

但当前训练空间主要是：

> baseline → one-axis intervention

因此模型尚未被系统训练和验证：

> 当两个或多个条件同时变化时，能否正确综合不同方向、不同强度的行为信号？

这构成当前最重要的补训缺口。

---

# 2. 核心研究问题

## RQ-M1

Can a Student trained primarily on single-axis counterfactuals reproduce the Teacher's response under joint perturbations?

## RQ-M2

Does targeted multi-axis distillation improve fidelity under seen combinations without degrading single-axis behavior?

## RQ-M3

Can the Student generalize compositionally to an unseen combination that was never included in training?

## RQ-M4

When two perturbations induce conflicting behavioral pressures, does the Student preserve the Teacher's trade-off rather than allowing one strong axis to dominate?

## RQ-M5

Does stabilizing low-SNR Teacher targets for fare and congestion improve multi-axis response quality?

---

# 3. 主要实验假设

## H1 — 单轴训练不足以保证联合响应

即使 Student 同时读取所有 context features，它也可能只是在输入空间中插值，而非真实学习联合行为机制。

预期：

- single-axis fidelity 高；
- joint-condition fidelity 明显下降。

## H2 — 强轴可能压制弱轴

当前不同轴的 Teacher SNR 差异明显，因此在组合场景下可能出现：

> weather / disruption 主导，fare / congestion 被弱化。

需要判断这种“主导效应”到底是 Teacher 本身的行为判断，还是 Student 的训练偏置。

## H3 — 定向 multi-axis supervision 可以改善 joint fidelity

在不扩大 persona 数量的前提下，加入少量高信息密度 joint-condition targets，应能够显著改善：

- joint probability fidelity；
- interaction fidelity；
- conflicting-context response；
- unseen-combination generalization。

---

# 4. 数据扩展原则

本轮不做：

- persona 40 → 80 / 100；
- 新增第七个扰动轴；
- 六轴全排列；
- 大规模随机组合；
- 更换 Student 主架构。

本轮只增加：

> **少量、有交通解释意义、存在冲突或耦合关系的多轴组合。**

推荐目标：

- 新增约 300–700 aggregated joint states；
- 优先保持现有 40 personas；
- 继续采用 persona-holdout；
- 继续使用 baseline + counterfactual 配对逻辑。

---

# 5. 推荐多轴组合

## 5.1 Combination A — Rain × Congestion

### 交通含义

Rain 通常会：

- 降低 bike / walk 吸引力；
- 提高 car / PT 相对吸引力。

Congestion 通常会：

- 降低 car 吸引力；
- 增加 PT 或非机动车相对优势。

因此两者对 car 的影响可能相反。

### 研究价值

这是典型的：

> **conflicting behavioral pressure**

推荐 levels：

```text
rain ∈ {0, 1}
road_congestion ∈ {low, medium, high}
```

保留：

- baseline；
- rain only；
- congestion only；
- rain + congestion。

## 5.2 Combination B — Fare × Congestion

### 交通含义

Fare increase：

- 降低 PT 吸引力；
- 可能推动 car。

Congestion increase：

- 降低 car 吸引力；
- 可能推动 PT。

这是最清楚的：

> **PT–car conflict pair**

### 推荐 levels

```text
fare_multiplier ∈ {1.0, 1.5, 2.0}
road_congestion ∈ {low, medium, high}
```

该组合建议作为：

> **unseen combination holdout**

即：

- Teacher 数据全部生成；
- 训练阶段不使用其中部分或全部 joint states；
- 测试 Student 的 compositional generalization。

## 5.3 Combination C — Fare × Transit Delay

### 交通含义

两个因素都降低 PT attractiveness：

- fare ↑
- delay ↑

### 研究价值

用于检测：

> additive effect vs nonlinear amplification

Teacher 可能表现为：

- 近似线性累加；
- 某个阈值后突然放弃 PT；
- 不同 persona 出现差异化 response。

### 推荐 levels

```text
fare_multiplier ∈ {1.0, 1.5, 2.0}
transit_delay_min ∈ {0, 5, 15, 30}
```

建议控制总状态数，不需要完整笛卡尔积。

## 5.4 Combination D — Road Disruption × Congestion

### 交通含义

Road disruption 已经是当前 Teacher 最强轴之一。

与 congestion 联合后可以观察：

- route availability；
- reliability；
- travel-time deterioration；
- car probability collapse；
- mode substitution。

### 研究价值

该组合与后续 Singapore real-network experiment 高度一致。

建议用于：

- seen combination training；
- MATSim real-network scenario mapping。

---

# 6. 推荐数据生成策略

不建议生成所有 persona × trip × all pair levels 的完全笛卡尔积。

采用：

> **balanced targeted sampling**

## 6.1 每个组合建议规模

以 40 personas × 2 trips 为基础。

每个 pair 选 2–3 个 joint levels。

例如：

```text
40 personas × 2 trips × 2 joint levels = 160 states / pair
```

四组全做约：

```text
160 × 4 = 640 joint states
```

这是一个合理上限。

如果 API 成本或时间希望更低：

- 对每组只使用 20–30 personas；
- 但保持 persona-holdout 结构一致。

---

# 7. Teacher Sampling 设计

## 7.1 高 SNR 组合

对于包含：

- weather；
- road_disruption；

可以先使用：

> K = 3

因为当前单轴 Teacher 相对稳定。

## 7.2 低 SNR 组合

对于包含：

- fare；
- congestion；

建议：

> K = 5

关键测试子集：

> K = 7

推荐：

```text
single-axis legacy data: K=3
joint low-SNR data: K=5
causal / benchmark subset: K=7
```

---

# 8. 数据 Schema

每条 joint state 至少应保留：

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

---

# 9. Train / Validation / Test 切分

必须继续保留：

> persona-holdout

避免同一 persona 泄漏。

同时新增第二层 split：

> combination-holdout

## 9.1 Split A — Seen Combination Test

训练和测试都包含同类 joint combination，但 test personas 未见。

目标：

> 测试联合响应在未见 persona 上是否保持。

## 9.2 Split B — Unseen Combination Test

推荐：

训练：

- rain × congestion
- fare × transit delay
- disruption × congestion

完全 hold out：

> fare × congestion

测试：

> fare × congestion on unseen personas

这是最有价值的 compositional test。

---

# 10. Student 补训策略

建议保留当前最优 Student 作为起点，不从零开始。

定义：

```text
Student-S3 → Student-S5-Joint
```

优先：

> fine-tune

而不是重新全部训练。

## 10.1 Loss 结构

继续使用：

\[
L = L_{action} + \lambda_d L_{distribution} + \lambda_e L_{elasticity} + \lambda_h L_{heterogeneity}
\]

新增：

\[
L_{joint}
\]

用于约束 joint-condition Teacher–Student probability distribution。

可以先简单定义：

\[
L_{joint}=\|P_T(x_i,x_j)-P_S(x_i,x_j)\|_1
\]

不建议一开始设计复杂新 loss。

---

# 11. Interaction Effect 指标

定义 baseline probability：

\[
P_0
\]

单轴响应：

\[
P_i,\quad P_j
\]

联合响应：

\[
P_{ij}
\]

定义 interaction：

\[
I_{ij}=P_{ij}-P_i-P_j+P_0
\]

如果：

\[
I_{ij}\approx 0
\]

表示近似 additive。

如果：

\[
I_{ij}\neq 0
\]

表示存在 interaction。

Student interaction fidelity：

\[
E_{interaction}=\|I^T_{ij}-I^S_{ij}\|_1
\]

这是本轮最重要的新指标之一。

---

# 12. 核心评价指标

必须报告：

### Static fidelity

- mode_accuracy
- KL(P_T || P_S)
- probability L1

### Counterfactual fidelity

- ΔP gap
- sign agreement

### Joint fidelity

- joint KL
- joint probability L1

### Interaction fidelity

- interaction L1 error

### Heterogeneity

- persona-wise joint response variance
- Teacher–Student heterogeneity gap

### Generalization

- seen-combination performance
- unseen-combination performance

---

# 13. 必须保留的对照组

## M0 — Existing Single-Axis Student

当前 S3 / 最终单轴模型。

## M1 — Joint Fine-Tuned Student

加入 seen multi-axis joint data。

## M2 — Joint Fine-Tuned + Low-SNR Stabilization

在 M1 基础上：

- fare / congestion joint states 使用更高 K。

---

# 14. 最重要结果表

建议最终形成：

| Model | Single-axis KL | Seen Joint KL | Unseen Joint KL | Interaction Error | Sign Agreement |
|---|---:|---:|---:|---:|---:|
| M0 Single-axis Student | | | | | |
| M1 Joint Fine-tune | | | | | |
| M2 Joint + stabilized Teacher | | | | | |

---

# 15. 单轴能力回归测试

Multi-axis fine-tuning 完成后必须确认：

> 不要为了 joint performance 破坏已有单轴能力。

比较：

```text
before S5 vs after S5
```

在完整六轴 legacy test set 上：

- mode accuracy
- KL
- L1
- ΔP gap
- sign

如果 joint fine-tuning 导致单轴明显退化，需要：

- replay legacy single-axis samples；
- mixed mini-batch training；
- 降低 joint loss weight。

---

# 16. Success Criteria

本轮不要求所有数字都极高。

满足以下多数条件即可认为成功：

- seen joint KL 明显低于原单轴 Student；
- interaction error 明显下降；
- conflicting combinations 不再被单一强轴完全吞没；
- unseen combination 优于 naïve baseline；
- legacy six-axis performance 不明显下降；
- low-SNR stabilized model 在 fare / congestion 组合下更稳定。

---

# 17. Failure Diagnosis

## Failure A — Seen combination 都学不好

可能原因：

- Student capacity 不足；
- joint dataset 过少；
- Teacher targets 太 noisy。

下一步：

- 先提高 K；
- 再做 capacity ablation；
- 不先扩大 persona。

## Failure B — Seen combination 好，unseen combination 很差

解释：

> 模型具备记忆联合模式能力，但 compositional generalization 弱。

这是合理科研结果，不等于项目失败。

## Failure C — Strong-axis domination 仍然存在

需要区分：

1. Teacher 也如此；
2. 只有 Student 如此。

如果 Teacher 也如此：

> 属于 behavioral trade-off。

如果只有 Student：

> 属于 distillation bias。

---

# 18. 与 Singapore Experiment 的连接

S5 完成后再进入真实网络。

Singapore case study 中优先使用已经有 joint supervision 的场景：

- rain + congestion；
- disruption + congestion；
- fare + congestion。

这样后续 MATSim network feedback 的解释更加可靠。

---

# 19. 推荐代码/输出命名

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

---

# 20. 停止条件

当以下条件满足后，不再继续增加 joint combinations：

- [ ] 至少 3 个 seen pair 已训练；
- [ ] 至少 1 个 pair 完全 holdout；
- [ ] seen joint fidelity 明显改善；
- [ ] unseen combination 有正式结果；
- [ ] interaction metric 已实现；
- [ ] legacy single-axis regression test 通过；
- [ ] Teacher low-SNR stabilization 有结果。

然后进入：

> **Reasoning / Causal Mechanism Audit**

---

# 21. 一句话实验目标

> **本实验不只是让 Student“看见更多变化”，而是验证并增强它在多条件同时变化、相互冲突时对 Teacher 行为权衡的保真能力。**
