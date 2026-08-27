# Reasoning & Causal Mechanism Audit
## Student 推理结构 / 因果机制测试实验设计

**项目主线**：DeepSeek V4 Pro Teacher → Lightweight Traveler Agent → MATSim  
**实验阶段建议名称**：S6 — Reasoning & Causal Mechanism Audit  
**前置条件**：完成 S5 Multi-Axis Behavioral Distillation  
**目标**：判断 Student 学到的是行为机制、组合权衡还是 shortcut mapping；严格区分 counterfactual responsiveness、causal consistency 与真正的显式 causal reasoning。

---

# 1. 实验目的

当前 Student 已经能够：

- 预测 mode distribution；
- 响应单轴 context perturbation；
- 保留部分 behavioral elasticity；
- 保留部分 persona heterogeneity；
- 进入 MATSim；
- 形成 population-scale 与 network-level feedback。

但这些现象本身不能证明：

> Student 内部存在“因果推理链”。

Student 当前本质是结构化输入 → MLP → behavioral output。

因此本实验不直接宣称：

> “Student has chain-of-thought.”

而是回答更严谨的问题：

> **Does the Student rely on causally meaningful intermediate transport mechanisms, or does it exploit shortcut correlations between context flags and actions?**

---

# 2. 能力层级定义

## L0 — Predictive Mapping

\[
x \rightarrow P(mode)
\]

是否能预测行为。

## L1 — Interventional Responsiveness

改变一个 context 后：

\[
do(x_i)
\]

行为是否按合理方向变化。

## L2 — Counterfactual Fidelity

Student 是否能复制 Teacher 对：

\[
baseline \rightarrow counterfactual
\]

的方向与幅度。

## L3 — Compositional Reasoning

多个 context 同时改变时，是否能保持 Teacher 的联合权衡。

## L4 — Mechanism Consistency

当 context 与 causal mediator 被人为拆开时，Student 是否主要响应真正的中介变量，而不是 context label shortcut。

当前已有证据主要覆盖：

> L0–L2

S5 用于补充：

> L3

本 S6 重点测试：

> **L4**

---

# 3. 核心研究问题

## RQ-C1

Does the Student respond to causal mediators or merely to high-level context labels?

## RQ-C2

When the natural causal pathway is broken, does the Student still produce a large behavioral response?

## RQ-C3

Can a mediator-only intervention reproduce the behavior induced by the natural context intervention?

## RQ-C4

Does the Student preserve the Teacher's causal response structure across unseen personas?

## RQ-C5

Does multi-axis training improve causal mechanism consistency or only improve joint interpolation?

---

# 4. 核心假设

## H1 — Shortcut Student

如果 Student 学到：

```text
congestion=high → car↓
```

那么即使：

```text
congestion=high
car travel time unchanged
```

它仍会显著降低 car probability。

这是 shortcut evidence。

## H2 — Mechanism-Consistent Student

如果 Student 学到：

```text
congestion
→ car travel time / reliability deterioration
→ car utility decline
→ car probability decline
```

那么：

- natural intervention 应产生明显变化；
- broken-path intervention 应显著减弱；
- mediator-only intervention 应重新产生变化。

---

# 5. 首批因果审计变量

只选择 causal mediator 容易定义的三个轴。

## 5.1 Congestion

### 假设机制

\[
Road\ Congestion \rightarrow Car\ Travel\ Time \rightarrow Car\ Generalized\ Utility \rightarrow P(Car)
\]

可能附带：

\[
Congestion \rightarrow Reliability \downarrow
\]

## 5.2 Transit Delay

### 假设机制

\[
Transit\ Delay \rightarrow PT\ Travel\ Time / Reliability \rightarrow PT\ Utility \rightarrow P(PT)
\]

## 5.3 Parking Cost

### 假设机制

\[
Parking\ Cost \rightarrow Generalized\ Car\ Cost \rightarrow Car\ Utility \rightarrow P(Car)
\]

该轴因果路径相对最干净，适合作为 positive-control audit。

---

# 6. 四联组 Causal Audit 设计

每个 persona × trip × causal axis 构造四个 state。

## State A — Baseline

Context 正常，mediator 正常。

例如：

```text
road_congestion = low
car_travel_time = 20 min
```

输出：

\[
P_A
\]

## State B — Natural Intervention

Context 改变，同时 mediator 按自然机制改变。

例如：

```text
road_congestion = high
car_travel_time = 40 min
```

输出：

\[
P_B
\]

## State C — Broken-Path Intervention

Context 改变，但人为固定 mediator。

例如：

```text
road_congestion = high
car_travel_time = 20 min
```

输出：

\[
P_C
\]

目的：

> 检测 context label shortcut。

## State D — Mediator-Only Intervention

Context 保持 baseline，但 mediator 改变。

例如：

```text
road_congestion = low
car_travel_time = 40 min
```

输出：

\[
P_D
\]

目的：

> 检测模型是否真正依赖 mediator。

---

# 7. 理想因果响应模式

若模型机制一致，则应大致满足：

\[
P_B \approx P_D
\]

且：

\[
P_C \approx P_A
\]

对 car probability 可写为：

\[
P_B(car) < P_A(car)
\]

\[
P_D(car) < P_A(car)
\]

\[
P_C(car) \approx P_A(car)
\]

不是要求完全相等，而是要求效应结构合理。

---

# 8. Teacher 也必须接受同样审计

重要原则：

> 不要默认 DeepSeek Teacher 就一定具有正确因果机制。

对 A/B/C/D 四个 state：

- Teacher K=5 或 K=7；
- Student 单次 deterministic / repeated seed inference；
- 同时比较 Teacher 与 Student。

这样可区分：

### Case 1 — Teacher causal-consistent，Student shortcut

→ distillation failure。

### Case 2 — Teacher 和 Student 都 shortcut

→ Teacher target 本身问题。

### Case 3 — Teacher 和 Student 都 mechanism-consistent

→ 支持 causal mechanism fidelity。

### Case 4 — Teacher 不稳定

→ 当前 causal claim 不应成立。

---

# 9. 核心效应指标

## 9.1 Natural Effect

\[
E_{natural}=P_B-P_A
\]

## 9.2 Broken-Path Effect

\[
E_{broken}=P_C-P_A
\]

## 9.3 Mediator Effect

\[
E_{mediator}=P_D-P_A
\]

---

# 10. Shortcut Sensitivity

定义：

\[
S_{shortcut}=\|P_C-P_A\|_1
\]

越低越好。

如果：

\[
S_{shortcut}\approx\|P_B-P_A\|_1
\]

说明模型基本直接响应 context label。

---

# 11. Mediator Dependency

定义：

\[
S_{mediator}=\|P_D-P_A\|_1
\]

如果 natural intervention 明显，而 mediator-only 几乎无响应：

> causal mechanism evidence 很弱。

---

# 12. Causal Path Fidelity

Teacher：

\[
E_T^{natural}, E_T^{broken}, E_T^{mediator}
\]

Student：

\[
E_S^{natural}, E_S^{broken}, E_S^{mediator}
\]

实现阶段建议优先直接报告三类 effect gap，而不是一开始设计复杂综合分数。

---

# 13. Shortcut Ratio

定义：

\[
R_{shortcut}=\frac{\|P_C-P_A\|_1}{\|P_B-P_A\|_1+\epsilon}
\]

解释：

- 接近 0：broken path 几乎无效；
- 接近 1：context label 本身产生和 natural intervention 同级别效果；
- >1：可能存在异常 shortcut 或 nonlinear response。

---

# 14. Mediator Recovery Ratio

\[
R_{mediator}=\frac{\|P_D-P_A\|_1}{\|P_B-P_A\|_1+\epsilon}
\]

理想情况下：

- 接近 1：mediator-only 可近似恢复 natural effect；
- 接近 0：模型基本不依赖 mediator。

---

# 15. 多轴推理测试

S5 完成后，应增加：

> causal audit + joint context

推荐：

### rain × congestion

测试：

- rain 增加 car tendency；
- congestion 通过 car travel time 抑制 car。

对 congestion mediator 做 broken-path。

如果：

```text
rain = true
congestion = high
car travel time = baseline
```

Student 仍然极度降低 car：

> 可能是 congestion shortcut。

---

# 16. Unseen Combination Reasoning Test

保留 S5 中完全未见组合：

> fare × congestion

直接比较：

- Teacher joint response；
- Student joint response；
- interaction effect；
- causal mediator sensitivity。

目标：

> 测试 compositional counterfactual generalization。

---

# 17. Sequential Reasoning / System-Level Chain Test

Student 本身是 stateless MLP，因此不能把系统 feedback 错写成模型内部 CoT。

但可以测试：

> Student + MATSim 系统是否形成稳定的多步 causal response。

## Path A — Sequential Path

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

## Path B — Direct Final-State Path

直接构造：

```text
rain = true
congestion = observed_high
```

让 Student 一次决策。

比较：

\[
P_{sequential}
\]

与：

\[
P_{direct}
\]

如果结果近似一致：

> 说明 Student 对最终 causal state 有稳定响应。

但不能称：

> Student internally reasoned through the entire chain。

---

# 18. 必须加入的对照模型

## C0 — Pre-S5 Single-Axis Student

原始模型。

## C1 — S5 Multi-Axis Student

联合补训模型。

## C2 — Teacher

DeepSeek V4 Pro。

## C3 — Optional MNL

用于观察传统模型在 broken-path 条件下的响应。

---

# 19. 推理测试结果分级

## Grade A — Strong Mechanism Consistency

表现：

- natural effect 明显；
- broken-path effect 小；
- mediator-only effect 明显；
- Student 与 Teacher 结构一致；
- unseen combination 仍能保持。

可表述：

> **causally consistent behavioral response**

不建议直接写：

> full causal reasoning chain。

## Grade B — Partial Mechanism Consistency

表现：

- broken-path 仍有响应；
- mediator effect 更强；
- joint combination 有一定 generalization。

可表述：

> **partial causal mechanism preservation**

## Grade C — Shortcut-Dominated

表现：

- broken-path ≈ natural；
- mediator-only 很弱；
- unseen combination 崩溃。

结论：

> Student 主要学到 associative shortcut。

---

# 20. 如果发现 Shortcut，下一步怎么做

不要立即推倒整个项目。

保留当前模型作为：

> associative baseline。

新增：

> causal counterfactual fine-tuning dataset。

数据形式继续使用 A/B/C/D 四联组。

---

# 21. Causal-Aware Fine-Tuning 可选设计

如果 S6 失败，再进入 S7。

新增 loss：

\[
L=L_{action}+L_{distribution}+L_{elasticity}+L_{heterogeneity}+\lambda_{path}L_{path}+\lambda_{shortcut}L_{shortcut}
\]

## 21.1 Path Loss

让 Student natural / mediator response 更接近 Teacher：

\[
L_{path}=\|E_T^{natural}-E_S^{natural}\|+\|E_T^{mediator}-E_S^{mediator}\|
\]

## 21.2 Shortcut Penalty

如果 Teacher 在 broken path 上效应弱：

\[
L_{shortcut}=\|E_T^{broken}-E_S^{broken}\|
\]

---

# 22. 不建议做的“伪因果证明”

不要因为以下现象就声称 causal reasoning：

- feature importance 高；
- attention weight 高；
- SHAP 显示 congestion 重要；
- Student 能响应 intervention；
- Phase 10 feedback loop 收敛；
- 多轴条件下预测准确。

这些都不能单独证明因果机制。

---

# 23. 推荐输出表

## Table A — Causal Audit

| Axis | Model | Natural Effect | Broken Effect | Mediator Effect | Shortcut Ratio | Mediator Ratio |
|---|---|---:|---:|---:|---:|---:|
| congestion | Teacher | | | | | |
| congestion | Student | | | | | |
| transit delay | Teacher | | | | | |
| transit delay | Student | | | | | |
| parking cost | Teacher | | | | | |
| parking cost | Student | | | | | |

## Table B — Reasoning Generalization

| Model | Seen Joint KL | Unseen Joint KL | Interaction Error | Shortcut Ratio |
|---|---:|---:|---:|---:|
| Pre-S5 | | | | |
| S5 Multi-axis | | | | |

---

# 24. 推荐图

## Figure 1 — Causal Intervention Diagram

```text
Context
  ↓
Mediator
  ↓
Behavior
```

展示：

- natural path；
- broken path；
- mediator-only。

## Figure 2 — Four-State Response Plot

对每个 axis 画：

```text
Baseline
Natural
Broken
Mediator-only
```

对应 mode probability。

## Figure 3 — Teacher vs Student Causal Effect Gap

## Figure 4 — Capability Ladder

```text
Single-axis Student
        ↓
Multi-axis Student
        ↓
Causal-aware Student (if needed)
```

---

# 25. 推荐代码结构

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

---

# 26. 成功停止条件

满足以下条件后即可停止 causal audit：

- [ ] congestion A/B/C/D 完成；
- [ ] transit delay A/B/C/D 完成；
- [ ] parking cost A/B/C/D 完成；
- [ ] Teacher 与 Student 都完成同样 audit；
- [ ] Shortcut Ratio 实现；
- [ ] Mediator Ratio 实现；
- [ ] 至少一个 unseen multi-axis combination 测试完成；
- [ ] sequential MATSim path 与 direct final-state path 完成；
- [ ] 对 causal claim 给出明确等级。

---

# 27. 论文表述边界

根据结果选择措辞。

### 如果结果较弱

> The Student preserves counterfactual behavioral responsiveness.

### 如果部分通过

> The Student preserves partial causal consistency under mediator interventions.

### 如果明显通过

> The Student exhibits causally consistent behavioral responses and compositional counterfactual generalization.

除非以后加入显式 causal latent modules / interpretable intermediate reasoning states，否则不建议写：

> The Student possesses a full causal chain-of-thought.

---

# 28. 一句话实验目标

> **本实验的目的不是证明一个 24k MLP“会思考”，而是严格检验它在蒸馏 Teacher 行为后，究竟保留了可迁移的交通行为机制，还是只记住了 context-to-action shortcuts。**
