# S8 — Real-Supply Transit Accessibility Adaptation
## 基于新加坡真实公共交通供给的 Transit Accessibility 定向训练指令

**前置条件**：S7-W3 已完成正式 Freeze  
**基础模型**：S7-W3 Generic Behavioral Core v1.0  
**阶段名称**：S8 — Real-Supply Transit Accessibility Adaptation  
**目标**：让 Student 不仅知道 PT 是否“存在、贵不贵、慢不慢”，还能够根据真实公共交通的首末公里、等待、换乘、覆盖度和 door-to-door convenience 调整 mode choice。  
**核心原则**：训练的是**城市无关的 Transit Accessibility 表征**，新加坡只作为真实 supply 数据来源，不允许训练 Singapore-specific location identity。

---

# 1. S8 的研究问题

## RQ-S8-1
Can the generic Traveler Agent adapt its mode choice to real transit accessibility derived from a multimodal transport network?

## RQ-S8-2
Can this adaptation be learned using city-independent supply attributes rather than Singapore-specific identities?

## RQ-S8-3
Does S8 reduce PT feasibility / convenience-related choice errors without degrading the original generic behavioral capabilities of S7-W3?

## RQ-S8-4
Can the model generalize to unseen Singapore ODs based only on accessibility attributes?

---

# 2. S8 不是“第七扰动轴”

禁止把 S8 简化成：

```text
transit_accessibility = low/medium/high
```

S8 应定义为：

> **real-supply-conditioned mode attributes**

即：

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

---

# 3. S8 通用性硬约束

Student 输入禁止包含：

- Singapore district ID；
- Tampines / Pasir Ris 标签；
- MRT station name；
- bus stop ID；
- route number；
- stop sequence ID；
- postal code；
- geographic identity embedding；
- GTFS route_id；
- GTFS stop_id。

允许输入：

- travel time；
- access time；
- egress time；
- wait time；
- transfer count；
- transfer time；
- in-vehicle time；
- coverage ratio；
- generalized cost；
- feasibility；
- reliability；
- walking burden。

原则：

> **模型学习交通供给属性，不学习地点名字。**

---

# 4. Step 1 — Schema Audit

在任何训练之前，Agent 必须审计 S7-W3 schema。

回答：

1. 当前是否已有：
   - PT travel time
   - monetary cost
   - reliability
   - availability
2. 是否已有 mode-level feature encoder？
3. 是否可以追加：
   - access_time
   - egress_time
   - wait_time
   - transfer_count
   - transfer_time
   - coverage_ratio
4. 新 feature 是否会改变 input dimension？
5. 能否保持旧 checkpoint 权重兼容？

输出：

```text
reports/S8_SCHEMA_AUDIT.md
```

---

# 5. Schema Audit 分支

## Case A — 可直接映射到现有 mode attributes

如果现有 Student 已有通用 PT 属性槽位，且不需改 input dimension：

> 使用 S7-W3 直接 targeted fine-tune。

---

## Case B — 必须新增输入 feature

如果需要增加 input dimension：

建立：

> `Student-S8`

要求：

- 复制 S7-W3 可复用权重；
- 新增 feature encoder 初始化；
- 不覆盖原 input schema；
- 新 architecture version 明确记录。

禁止声称：

> S8 与 S7 完全相同架构。

---

# 6. Transit Accessibility Feature Set

推荐最小特征组：

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

可选：

```text
pt_direct
pt_service_frequency
pt_reliability
pt_walk_distance
```

不要一次加入大量高度相关特征。

---

# 7. Feature 定义

## 7.1 PT Feasible

```text
1 = 存在满足当前时间窗与 routing rules 的 PT itinerary
0 = 不存在
```

---

## 7.2 Access Time

origin → first boarding stop 的步行时间。

---

## 7.3 Egress Time

last alighting stop → destination 的步行时间。

---

## 7.4 Wait Time

首段等待 + 必要换乘等待。

---

## 7.5 In-Vehicle Time

所有 PT vehicle legs 的车内时间总和。

---

## 7.6 Transfer Count

PT vehicle-to-vehicle transfer 次数。

---

## 7.7 Transfer Time

换乘步行 + 等待总时间。

---

## 7.8 Door-to-Door Time

\[
T_{PT}
=
T_{access}
+
T_{wait}
+
T_{invehicle}
+
T_{transfer}
+
T_{egress}
\]

---

## 7.9 Coverage Ratio

推荐：

\[
R_{coverage}
=
\frac{D_{PT,vehicle}}
{D_{OD}}
\]

必要时 clip 到：

```text
[0,1]
```

---

## 7.10 Generalized Cost

如果已有统一 generalized cost framework：

沿用已有定义。

如果没有：

不要随意发明固定价值时间参数。

先保留原始 feature，由 Student / Teacher 学习。

---

# 8. 数据来源

S8 使用：

> Singapore OSM + Singapore GTFS real supply

但必须记录：

- OSM snapshot date；
- GTFS source/version；
- study area；
- routing rules；
- departure time window；
- transfer rule；
- walk radius；
- max transfer；
- PT validity rate。

---

# 9. OD 采样原则

不要随机生成大量完全无结构 OD。

目标是覆盖 Transit Accessibility spectrum。

按 accessibility 分层采样：

## Class A — Excellent
典型：

- access ≤ 5 min
- egress ≤ 5 min
- 0–1 transfer
- high coverage
- competitive door-to-door time

## Class B — Good

## Class C — Moderate

## Class D — Poor
典型：

- long access / egress
- 2 transfers
- lower coverage
- high total time

## Class E — Infeasible
不存在合理 PT itinerary。

---

# 10. 推荐数据规模

S8 是 targeted adaptation，不做巨大扩训。

建议：

```text
300–800 unique OD-context states
```

根据 API 成本调整。

优先信息密度，不追求数量。

Persona 保持现有 40-persona 体系。

默认：

> 不扩 persona。

---

# 11. Counterfactual Accessibility Curves

S8 最重要的训练结构不是随机点，而是同一 persona / trip 下的 accessibility curve。

例如：

## PT Good

```text
access = 3
egress = 4
wait = 4
transfer = 0
coverage = 0.92
door_to_door = 28
```

## PT Medium

```text
access = 8
egress = 7
wait = 8
transfer = 1
coverage = 0.75
door_to_door = 42
```

## PT Poor

```text
access = 15
egress = 12
wait = 12
transfer = 2
coverage = 0.48
door_to_door = 61
```

## PT Infeasible

```text
pt_feasible = 0
```

---

# 12. Teacher Labeling

对真实 supply-conditioned states 调 DeepSeek Teacher。

Teacher 输入必须包含：

- persona；
- trip；
- non-PT mode attributes；
- dynamic context；
- Transit Accessibility features。

禁止给 Teacher：

- Singapore location name；
- route number；
- station identity。

目标：

> Teacher 依据通用交通属性做判断。

---

# 13. Teacher Sampling

建议：

```text
K = 3
```

作为基础。

对以下边界样本：

- PT good ↔ medium；
- medium ↔ poor；
- poor ↔ infeasible；

可使用：

```text
K = 5
```

不建议全数据 K=7。

---

# 14. 数据 Split

必须有：

## Split A — Persona holdout
保留 unseen personas。

## Split B — OD holdout
训练 OD 与测试 OD 完全不重叠。

## Split C — Accessibility holdout
保留部分 accessibility profile / level 作为测试。

至少保证：

> 测试集不包含训练见过的完全相同 OD。

---

# 15. Unseen OD Test

核心目标：

> 模型不能靠记忆 Singapore OD。

因此：

```text
train: OD set A
test: completely unseen OD set B
```

Student 只看 accessibility vector。

---

# 16. Baseline Models

必须保留：

## B0 — S7-W3 Frozen
不见 S8 数据。

## B1 — S8 Student
Transit Accessibility adapted。

## Teacher
参考上限。

如果 Case B 改 architecture，必须明确：

> S8 与 S7-W3 参数量变化。

---

# 17. S8 Training Strategy

优先：

> 从 S7-W3 初始化。

训练 batch 混合：

```text
legacy S3
+
S5 joint
+
S7 mechanism
+
S8 accessibility
```

建议初始比例：

```text
2 : 1 : 1 : 1
```

如果 S8 学不动：

```text
2 : 1 : 1 : 2
```

但 accessibility 不建议超过总 batch 40–50%。

---

# 18. Loss

首先复用现有：

\[
L_{action}
+
L_{distribution}
+
L_{elasticity}
+
L_{heterogeneity}
+
L_{joint}
+
L_{mechanism}
\]

不要立刻新增复杂 loss。

---

# 19. Accessibility Response Loss（可选）

只有 baseline fine-tune 不足时才增加。

定义：

Teacher accessibility effect：

\[
E_T^{acc}
=
P_T(PT|good)-P_T(PT|poor)
\]

Student：

\[
E_S^{acc}
=
P_S(PT|good)-P_S(PT|poor)
\]

可定义：

\[
L_{accessibility}
=
|E_T^{acc}-E_S^{acc}|
\]

第一轮建议先不加，避免同时改变数据和 objective。

---

# 20. Primary Metrics

## 20.1 PT Probability Fidelity

- KL
- probability L1
- PT probability MAE

---

## 20.2 Accessibility Sensitivity

\[
\Delta P_{PT}
=
P(PT|good)-P(PT|poor)
\]

比较：

- Teacher
- S7-W3
- S8

---

## 20.3 Monotonicity

检查：

```text
PT good
→ PT medium
→ PT poor
```

是否满足：

\[
P(PT)_{good}
>
P(PT)_{medium}
>
P(PT)_{poor}
\]

定义 monotonic agreement rate。

---

# 21. PT Feasibility Violation Rate

定义：

\[
FVR
=
\frac{
\#(\text{PT infeasible state 中 Student 仍将 PT 作为最高概率 mode})
}{
\#(\text{PT infeasible states})
}
\]

也可同时报告：

\[
P(PT|\text{infeasible})
\]

的均值。

S8 目标：

> 显著低于 S7-W3。

---

# 22. PT Convenience Error

定义：

\[
E_{conv}
=
|P_T(PT)-P_S(PT)|
\]

分别在：

- Excellent
- Good
- Moderate
- Poor
- Infeasible

五档报告。

---

# 23. Real Routing Consistency

将 Student choice 与真实 GTFS itinerary 对齐。

每个 trip 保存：

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

---

# 24. Phase B PT Validity 回归

S8 后重新运行：

> Singapore PT Routing Validity

确认：

- PT intended trips 中可行率；
- routing success；
- fallback；
- boarding/alighting consistency。

注意：

S8 目标不是让 routing success 从 98% 变 100%。

它目标是：

> Student 更少选择明显低便利/不可行 PT。

---

# 25. Generic Capability Regression

S8 后必须重跑：

## S3 legacy
- accuracy
- KL
- L1
- ΔP
- sign

## S5 multi-axis
- seen joint KL
- unseen joint KL
- interaction

## S6/S7 mechanism
至少：
- parking G_med
- congestion Gap_shortcut

---

# 26. Regression Gate

默认要求：

```text
legacy accuracy drop ≤ 1 pp
legacy KL increase ≤ 10%
seen joint KL increase ≤ 10%
unseen joint KL increase ≤ 10%
```

mechanism 指标不能明显回退。

---

# 27. 通用性 Gate

S8 只有满足以下条件才能称为：

> supply-aware extension of generic behavioral core

必须满足：

- [ ] no Singapore ID input；
- [ ] unseen OD test；
- [ ] legacy generic behavior 保留；
- [ ] accessibility sensitivity 提升；
- [ ] feasibility violation 降低；
- [ ] feature schema 是 city-independent。

---

# 28. 可允许的论文表述

如果成功：

> **a city-agnostic supply-aware Traveler Agent conditioned on transferable transit accessibility attributes**

或者：

> **a generic behavioral core augmented with real-supply transit accessibility features**

不建议：

> universally generalizable traveler model

也不建议：

> trained on real Singapore traveler behavior

---

# 29. Singapore 数据在论文中的正确表述

允许：

> real-world Singapore transport supply

> real OSM/GTFS-derived transit accessibility

> real-supply-conditioned behavioral distillation

禁止：

> real Singapore behavioral labels

除非未来加入 observed human choice data。

---

# 30. Optional Zero-Shot Cross-City Test

非 S8 必须项。

如果后续需要强化 city-agnostic claim：

选第二城市少量 GTFS OD：

```text
50–200 OD
```

不训练。

直接：

```text
GTFS
→ accessibility vector
→ S8 inference
```

只做 zero-shot sanity check。

不要让这一步阻塞 Phase C。

---

# 31. 推荐代码

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

---

# 32. 自动执行顺序

Agent 必须严格执行：

## Step 1
确认 S7-W3 Freeze 完成。

## Step 2
执行 Schema Audit。

## Step 3
输出 Case A / Case B 判定。

## Step 4
构建 Singapore accessibility feature pipeline。

## Step 5
做 feature sanity checks。

## Step 6
构造 stratified OD/accessibility dataset。

## Step 7
冻结 train/val/test split。

## Step 8
调用 Teacher。

## Step 9
训练 S8 baseline。

## Step 10
运行：
- accessibility evaluation
- unseen OD
- feasibility violation
- generic regression

## Step 11
只有 baseline 不足时，再考虑 accessibility-specific loss。

## Step 12
生成最终报告。

---

# 33. Stop Rule

满足以下条件后停止 S8：

- [ ] accessibility response 明显优于 S7-W3；
- [ ] unseen OD 保持；
- [ ] FVR 明显下降；
- [ ] legacy / joint / mechanism 无明显回退；
- [ ] Singapore-specific ID 未进入模型；
- [ ] feature schema 可迁移。

然后：

> Freeze S8 → Phase C。

---

# 34. Failure Case

如果 S8 失败：

- 不删除 S7-W3；
- 不继续无限扩 feature；
- 不增加 taxi/Grab；
- 不换大模型 Student。

回退：

> S7-W3 作为最终 generic core

并在 Phase C 中明确 PT accessibility 是 limitation。

---

# 35. S8 成功后的最终模型关系

```text
S7-W3
Generic Behavioral Core v1.0
        │
        └────── initialization ──────→ S8
                                      │
                                      ↓
                        Supply-Aware Traveler Agent
```

S7-W3 永远保留作为 baseline。

---

# 36. 一句话执行原则

> **S8 不是把模型训练成“新加坡模型”，而是用新加坡真实 OSM/GTFS 产生的城市无关 Transit Accessibility 属性，训练一个仍可迁移到其他城市供给环境的 supply-aware Traveler Agent。**
