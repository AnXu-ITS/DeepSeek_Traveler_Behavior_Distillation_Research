# S7 — Mechanism-Aware Behavioral Distillation
## 机制感知补训实验执行指令

**项目主线**：DeepSeek V4 Pro Teacher → Lightweight Traveler Agent → MATSim  
**前置阶段**：S5 Multi-Axis Distillation ✅；S6 Causal Mechanism Audit ✅  
**本阶段目标**：针对 S6 暴露出的 `congestion` 与 `parking_cost` 机制保真退化进行定向补训，在尽量不损伤既有预测性能、多轴性能和单轴性能的前提下，使 Student 更接近 Teacher 的 mediator-dependent response structure。  
**阶段编号**：S7  
**推荐命名**：Mechanism-Aware / Causal-Aware Fine-Tuning  
**执行原则**：只修已被 S6 明确定位的问题，不重新扩 persona、不新增扰动轴、不重做整套 Teacher dataset。

---

# 0. S7 为什么必须做

S6 已经得到清晰结论：

## transit_delay
机制保留良好：

- Teacher `R_mediator ≈ 0.97`
- S5 Student `R_mediator ≈ 0.84`
- S5 Student `R_shortcut ≈ 0.35`

因此：

> `transit_delay` 不进入 S7 主补训范围。

## congestion
Teacher 明显依赖 mediator：

- Teacher `R_shortcut ≈ 0.52`
- Teacher `R_mediator ≈ 0.89`

但 S5 Student：

- `R_shortcut ≈ 1.02`
- `R_mediator ≈ 0.67`

而且相较 pre-S5：

- shortcut 进一步增强；
- mediator fidelity 进一步下降。

因此：

> S5 提升了联合条件预测，但没有保存 congestion → car travel time / reliability → choice 的机制结构。

## parking_cost
问题最严重：

Teacher：

- `R_shortcut ≈ 0.69`
- `R_mediator ≈ 0.56`

S5 Student：

- `R_shortcut ≈ 1.02`
- `R_mediator ≈ 0.06`

说明 Student 几乎只学到了：

```text
parking_cost label ↑
    ↓
car probability ↓
```

而没有保留 Teacher 对实际 monetary-cost mediator 的响应。

因此 S7 的首要修复目标是：

1. `parking_cost`
2. `congestion`

---

# 1. S7 核心研究问题

## RQ-S7-1
Can mechanism-aware counterfactual supervision restore mediator-response fidelity for congestion and parking cost?

## RQ-S7-2
Can this restoration be achieved without sacrificing:

- legacy single-axis fidelity；
- S5 seen joint fidelity；
- S5 unseen-combination generalization；
- mode-choice accuracy？

## RQ-S7-3
Does mechanism-aware fine-tuning reduce Student shortcut sensitivity toward high-level context labels?

## RQ-S7-4
Does the repaired Student more faithfully match the Teacher's natural / broken-path / mediator-only response structure?

---

# 2. 本阶段禁止事项

S7 默认禁止：

- [ ] persona 40 → 80 / 100；
- [ ] 增加第七个 context axis；
- [ ] 重新生成全部 1513+ legacy states；
- [ ] 重新训练 Teacher；
- [ ] 更换 Student 主体架构；
- [ ] 引入 Transformer / LLM Student；
- [ ] 强行让 broken-path effect = 0；
- [ ] 把人为定义的 mediator graph 当作绝对真实因果图；
- [ ] 为追求漂亮结果删除不利样本；
- [ ] 因为 parking/congestion 有 shortcut 就重新从零训练整个项目。

S7 是：

> **targeted repair**

不是：

> full rebuild。

---

# 3. 模型版本冻结

首先冻结三个模型：

## C0
`pre-S5 Student`

用途：

> 作为单轴阶段 associative baseline。

## C1
`S5 Multi-Axis Student`

用途：

> S7 的直接起点，也是当前主 baseline。

## C2
`S7 Mechanism-Aware Student`

用途：

> 本阶段新模型。

不得覆盖 C0/C1 checkpoint。

推荐目录：

```text
outputs/
  student_s3_c/
  student_s5_joint/
  student_s7_mechanism/
```

---

# 4. S7 数据设计

S7 不需要大量新样本。

只针对：

- `congestion`
- `parking_cost`

生成 / 整理机制审计四联组：

```text
A = baseline
B = natural intervention
C = broken-path intervention
D = mediator-only intervention
```

---

# 5. congestion 四联组

## 假设机制

```text
road_congestion
    ↓
car travel time / reliability
    ↓
car generalized attractiveness
    ↓
P(car)
```

注意：

该机制不是声称 congestion 只通过 travel time 作用。

S7 只要求：

> Student 对 Teacher 已表现出的 mediator-dependent response 进行保真。

## A — baseline

示例：

```yaml
road_congestion: low
car_travel_time: baseline
car_reliability: baseline
```

## B — natural intervention

示例：

```yaml
road_congestion: high
car_travel_time: increased
car_reliability: decreased
```

## C — broken-path

Context 改变，但 mediator 保持 baseline：

```yaml
road_congestion: high
car_travel_time: baseline
car_reliability: baseline
```

## D — mediator-only

Context label 保持 baseline，但 mediator 改变：

```yaml
road_congestion: low
car_travel_time: increased
car_reliability: decreased
```

---

# 6. parking_cost 四联组

## 假设机制

```text
parking_cost
    ↓
car monetary / generalized cost
    ↓
car attractiveness
    ↓
P(car)
```

## A — baseline

```yaml
parking_cost_multiplier: 1.0
car_monetary_cost: baseline
```

## B — natural intervention

```yaml
parking_cost_multiplier: high
car_monetary_cost: increased
```

## C — broken-path

```yaml
parking_cost_multiplier: high
car_monetary_cost: baseline
```

## D — mediator-only

```yaml
parking_cost_multiplier: 1.0
car_monetary_cost: increased
```

---

# 7. 数据来源优先级

优先顺序：

## Priority 1
复用 S6 已生成的 A/B/C/D states。

如果 S6 数据已经包含完整 Teacher target，不重新烧 API。

## Priority 2
只有以下情况才新增 Teacher 调用：

- S6 某些四联组缺少完整 Teacher probability；
- K 太低，导致机制 effect estimate 明显不稳定；
- 当前 mediator state 设计不一致；
- 某些组因可用 mode 限制无法形成有效四联组。

---

# 8. Teacher Sampling 要求

推荐：

## congestion
`K = 5`

关键 validation subset：

`K = 7`

## parking_cost
`K = 5`

如果 Teacher response 已足够稳定，不必强制 K=7。

## 重要原则

Teacher broken-path response 本身不一定为 0。

因此 Student 不应被训练成：

```text
broken effect = 0
```

而应被训练成：

```text
Student broken effect ≈ Teacher broken effect
```

这是 S7 最重要的方法原则。

---

# 9. 有效人群过滤

继续沿用 S6 的合理过滤：

## congestion / parking_cost
只保留：

> car available

的 persona / trip group。

原因：

如果 car 不可用：

```text
E_natural ≈ 0
```

则：

```text
R_shortcut
R_mediator
```

会失去解释意义。

---

# 10. Mechanism Quadruplet Dataset

建议新建：

```text
src/traveler_distillation/student/mechanism_dataset.py
```

实现：

```text
MechanismQuadrupletDataset
```

每个样本返回：

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

---

# 11. S7 Loss 设计

保留当前主损失：

\[
L_{base}
=
L_{action}
+
\lambda_d L_{distribution}
+
\lambda_e L_{elasticity}
+
\lambda_h L_{heterogeneity}
+
\lambda_j L_{joint}
\]

新增两类机制损失。

---

# 12. Natural / Mediator Mechanism Fidelity Loss

定义：

Teacher natural effect：

\[
E_T^{nat}
=
P_T(B)-P_T(A)
\]

Student natural effect：

\[
E_S^{nat}
=
P_S(B)-P_S(A)
\]

Teacher mediator effect：

\[
E_T^{med}
=
P_T(D)-P_T(A)
\]

Student mediator effect：

\[
E_S^{med}
=
P_S(D)-P_S(A)
\]

定义：

\[
L_{mechanism}
=
\|E_T^{nat}-E_S^{nat}\|_1
+
\|E_T^{med}-E_S^{med}\|_1
\]

目标：

> Student 不只是复制最终 B / D 概率，而是复制 intervention effect。

---

# 13. Broken-Path Fidelity Loss

Teacher：

\[
E_T^{broken}
=
P_T(C)-P_T(A)
\]

Student：

\[
E_S^{broken}
=
P_S(C)-P_S(A)
\]

定义：

\[
L_{broken}
=
\|E_T^{broken}-E_S^{broken}\|_1
\]

注意：

禁止定义：

\[
L_{broken}
=
\|E_S^{broken}\|
\]

因为 Teacher broken-path effect 本身不为 0。

---

# 14. 最终 S7 Loss

推荐：

\[
L_{S7}
=
L_{base}
+
\lambda_m L_{mechanism}
+
\lambda_b L_{broken}
\]

初始权重建议：

```yaml
lambda_mechanism: 0.5
lambda_broken: 0.5
```

不要一开始设太高。

---

# 15. Loss Weight Ablation

至少测试：

## W0
```text
λm = 0
λb = 0
```

即当前 S5 baseline。

## W1
```text
λm = 0.25
λb = 0.25
```

## W2
```text
λm = 0.5
λb = 0.5
```

## W3
```text
λm = 1.0
λb = 1.0
```

不需要做更复杂网格。

目标是找到：

> mechanism fidelity 明显改善，但 legacy / joint performance 不明显退化

的最小权重。

---

# 16. Replay 策略

为了避免 S7 只盯机制样本导致 catastrophic forgetting，训练 batch 必须混合：

```text
legacy single-axis
+
S5 seen joint
+
S7 mechanism quadruplets
```

建议初始比例：

```text
legacy single-axis : seen joint : mechanism
= 2 : 1 : 1
```

可选：

```text
1 : 1 : 1
```

但不建议 mechanism samples 占比 > 50%。

---

# 17. Training Strategy

推荐：

> 从 S5 最优 checkpoint fine-tune。

不是从零训练。

## Learning rate

使用低于 S5 的 LR。

例如：

```text
S5 LR × 0.25
```

或：

```text
S5 LR × 0.5
```

## Epoch

使用 early stopping。

监控：

```text
validation total loss
+
mechanism effect gap
+
legacy KL
```

不要只监控 mechanism loss。

---

# 18. Primary Evaluation

S7 完成后重新运行完整 S6 causal audit。

比较：

```text
C0 pre-S5
C1 S5
C2 S7
Teacher
```

---

# 19. 核心因果审计指标

对 congestion / parking_cost 报告：

## Natural Effect

\[
E_{natural}
=
\|P_B-P_A\|_1
\]

## Broken Effect

\[
E_{broken}
=
\|P_C-P_A\|_1
\]

## Mediator Effect

\[
E_{mediator}
=
\|P_D-P_A\|_1
\]

## Shortcut Ratio

\[
R_{shortcut}
=
\frac{E_{broken}}{E_{natural}+\epsilon}
\]

## Mediator Ratio

\[
R_{mediator}
=
\frac{E_{mediator}}{E_{natural}+\epsilon}
\]

---

# 20. 机制保真真正的评价方式

不要把：

```text
R_shortcut 越低越好
```

作为唯一目标。

因为 Teacher 自己有 residual broken-path effect。

真正评价：

\[
Gap_{shortcut}
=
|R_{shortcut}^S-R_{shortcut}^T|
\]

\[
Gap_{mediator}
=
|R_{mediator}^S-R_{mediator}^T|
\]

目标：

```text
Gap ↓
```

---

# 21. Teacher–Student Effect Gap

对每个 axis 报告：

\[
G_{nat}
=
\|E_T^{nat}-E_S^{nat}\|_1
\]

\[
G_{broken}
=
\|E_T^{broken}-E_S^{broken}\|_1
\]

\[
G_{med}
=
\|E_T^{med}-E_S^{med}\|_1
\]

这是比单独看 Student ratio 更可靠的主指标。

---

# 22. 必须做 Legacy Regression Test

S7 不能以牺牲原有能力为代价。

重新跑：

> S3 legacy six-axis test set

指标：

- mode accuracy
- KL
- probability L1
- ΔP gap
- sign agreement

## 可接受范围

建议定义：

```text
accuracy drop ≤ 1.0 percentage point
KL increase ≤ 10%
probability L1 increase ≤ 10%
```

如果超过：

> S7 视为发生明显 regression。

---

# 23. 必须做 S5 Joint Regression Test

重新评估：

## seen joint
- joint KL
- joint L1
- interaction error

## unseen holdout
- fare × congestion KL
- fare × congestion L1
- interaction error

## 可接受范围

S7 不要求 joint performance 继续提升。

但原则上：

```text
seen joint KL increase ≤ 10%
unseen joint KL increase ≤ 10%
```

---

# 24. Success Criteria — congestion

S7 成功目标不是把 ratio 变成 0。

推荐目标：

## 当前 S5
```text
R_shortcut ≈ 1.02
R_mediator ≈ 0.67
```

## Teacher
```text
R_shortcut ≈ 0.52
R_mediator ≈ 0.89
```

## S7 理想区间
```text
R_shortcut: 0.55–0.80
R_mediator: 0.75–0.95
```

只要明显向 Teacher 靠近即可。

---

# 25. Success Criteria — parking_cost

## 当前 S5
```text
R_shortcut ≈ 1.02
R_mediator ≈ 0.06
```

## Teacher
```text
R_shortcut ≈ 0.69
R_mediator ≈ 0.56
```

## S7 理想区间
```text
R_shortcut: 0.65–0.85
R_mediator: 0.35–0.65
```

parking_cost 是本阶段最重要目标。

---

# 26. Primary Success Definition

若同时满足：

1. parking_cost mediator gap 明显下降；
2. congestion mediator gap 明显下降；
3. parking shortcut gap 明显下降；
4. congestion shortcut gap 明显下降；
5. legacy single-axis 不明显退化；
6. S5 seen/unseen joint 不明显退化；

则判定：

> **S7 成功。**

---

# 27. 结果表 A — Causal Mechanism Repair

最终必须生成：

| Axis | Model | R_shortcut | R_mediator | Gap_shortcut vs Teacher | Gap_mediator vs Teacher |
|---|---|---:|---:|---:|---:|
| congestion | Teacher | | | — | — |
| congestion | C0 | | | | |
| congestion | C1 | | | | |
| congestion | C2 S7 | | | | |
| parking | Teacher | | | — | — |
| parking | C0 | | | | |
| parking | C1 | | | | |
| parking | C2 S7 | | | | |

---

# 28. 结果表 B — Effect Fidelity

| Axis | Model | Natural Gap | Broken Gap | Mediator Gap |
|---|---|---:|---:|---:|
| congestion | C0 | | | |
| congestion | C1 | | | |
| congestion | C2 | | | |
| parking | C0 | | | |
| parking | C1 | | | |
| parking | C2 | | | |

---

# 29. 结果表 C — Performance Retention

| Model | Legacy KL | Legacy L1 | Seen Joint KL | Unseen Joint KL |
|---|---:|---:|---:|---:|
| C1 S5 | | | | |
| C2 S7 | | | | |

---

# 30. Ablation 结论要求

最终必须明确回答：

### Q1
只加 `L_mechanism` 是否有效？

### Q2
只加 `L_broken` 是否有效？

### Q3
两者同时加是否最好？

最低可通过 W1/W2/W3 间的实验回答。

如果工程成本允许，可额外：

```text
A = mechanism only
B = broken only
C = both
```

但这不是强制。

---

# 31. Failure Mode A — mechanism 改善但 predictive performance 下降

处理顺序：

1. 降低 `λm / λb`；
2. 增加 legacy replay；
3. 降低 LR；
4. early stop。

不要立刻扩大模型。

---

# 32. Failure Mode B — parking mediator 仍然几乎为 0

首先检查：

- monetary_cost 是否真的进入 Student input；
- preprocessing 是否归一化过度；
- parking_cost label 与 monetary_cost 是否高度冗余；
- monetary_cost 的变化范围是否足够；
- Student 是否存在 feature scaling 问题。

如果输入链路没有问题，再做：

> capacity ablation

而不是直接重构整个模型。

---

# 33. Failure Mode C — congestion mechanism 修不回来

检查：

- travel_time / reliability 是否同时作为 mediator；
- congestion label 是否压倒 mediator feature scale；
- Teacher mediator effect 是否稳定；
- S7 quadruplet 数量是否不足。

如果 Teacher 自己不稳定：

> 结论应降低 causal claim，不强修。

---

# 34. Failure Mode D — S7 完全无效

如果：

- mechanism gap 无明显改善；
- legacy / joint performance 还下降；

则：

> S7 终止。

保留 S5 作为最终 Student。

论文结论：

> output-level and joint behavioral distillation preserve predictive behavior but do not reliably preserve mechanism structure on all axes.

这仍然是有效科研结果。

---

# 35. Optional — Structured Student v2

只有 S7 失败且论文确实需要更强机制保真时，才考虑 Student v2。

建议结构：

```text
Persona + Context + Supply
        ↓
Perceived mode attributes
        ↓
Generalized utility / latent preference
        ↓
Mode distribution
```

不要直接上 LLM Student。

---

# 36. 与 Singapore Experiment 的关系

S7 结束后，无论成功或失败，都停止继续机制补训。

然后进入：

> Singapore OSM + GTFS real-world validation

如果 S7 成功：

使用 S7 Student。

如果 S7 失败：

使用 S5 Student，但在论文中明确说明：

> network validation evaluates behavioral executability and system response, not full causal mechanism preservation.

---

# 37. 推荐代码文件

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

---

# 38. 自动化执行顺序

Agent 严格按以下顺序执行：

## Step 1
读取：

```text
EXPERIMENT_REPORT_S5_MULTI_AXIS.md
EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md
```

提取所有 baseline 数字。

## Step 2
检查 S6 A/B/C/D 数据是否可以直接复用。

优先复用。

## Step 3
实现 mechanism quadruplet dataset。

## Step 4
实现：

```text
L_mechanism
L_broken
```

并写单元测试。

## Step 5
先用 mock / small subset smoke。

要求：

```text
all tests pass
training exits normally
loss finite
quadruplet linking correct
```

## Step 6
训练 W1/W2/W3。

## Step 7
完整重跑 S6 causal audit。

## Step 8
完整跑 legacy regression。

## Step 9
完整跑 S5 seen/unseen joint regression。

## Step 10
自动选择最优 S7 checkpoint。

选择标准按优先级：

1. mechanism gap ↓；
2. parking mediator fidelity ↑；
3. congestion mediator fidelity ↑；
4. legacy 不退化；
5. joint 不退化。

## Step 11
生成最终：

```text
EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md
```

---

# 39. 最终报告必须回答

最终报告必须明确回答：

1. S7 是否改善 parking_cost mediator fidelity？
2. S7 是否改善 congestion mediator fidelity？
3. shortcut gap 是否向 Teacher 靠近？
4. natural / broken / mediator effect gap 是否缩小？
5. legacy six-axis 是否退化？
6. S5 seen joint 是否退化？
7. unseen fare×congestion 是否退化？
8. 哪个 loss weight 最优？
9. S7 是否值得作为最终 Student？
10. 下一阶段是否可以进入 Singapore real-network validation？

---

# 40. 最终判定规则

## Grade A — S7 成功

机制 fidelity 明显改善，预测性能基本保持。

下一步：

> Freeze S7 → Singapore。

## Grade B — 部分成功

只修复一个 axis，另一个仍存在 shortcut。

下一步：

> Freeze S7 or S5，按整体性能择优 → Singapore。

论文表述：

> axis-dependent mechanism preservation。

## Grade C — S7 失败

机制无明显改善，或损伤原有性能。

下一步：

> 回退 S5 → Singapore。

不要继续无限补训。

---

# 41. 一句话执行目标

> **S7 的目标不是让 Student“拥有思维链”，而是在已证明多轴行为预测有效的基础上，定向恢复 S6 中丢失的 Teacher mediator-response structure，并验证这种机制修复是否能够在不牺牲预测与联合泛化能力的情况下实现。**
