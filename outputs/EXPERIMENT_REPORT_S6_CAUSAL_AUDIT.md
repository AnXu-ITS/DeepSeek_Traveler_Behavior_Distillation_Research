# S6 因果机制审计实验报告

## Reasoning & Causal Mechanism Audit

**阶段**：S6 — Student 推理结构 / 因果机制测试
**依据**：`S6_REASONING_CAUSAL_AUDIT_EXPERIMENT_DESIGN.md`
**前置**：S5 多轴蒸馏完成（本实验的前置条件已满足）

## 1. 实验设计

- 三轴审计：congestion / transit_delay / parking_cost（各含可定义的 causal mediator）。
- 四联组 A/B/C/D：baseline / natural / broken-path（标签变但 mediator 固定）/ mediator-only（标签不变但 mediator 变）。
- 指标：E_natural=‖P_B−P_A‖₁、E_broken=‖P_C−P_A‖₁、E_mediator=‖P_D−P_A‖₁；
  R_shortcut=E_broken/(E_natural+ε)（↓=机制），R_mediator=E_mediator/(E_natural+ε)（↑=机制）。
- 模型：Teacher（K=5/复用 S3 K=3）、C0（pre-S5 单轴 Student）、C1（S5 多轴 Student）。
- **人群过滤**：congestion/parking 仅保留 car 可用的 persona（无车人群 E_natural=0 会使比值发散）；
  有效组数 152（transit_delay 80；congestion/parking 各 36），跳过无车组 88。

## 2. 核心结果（Table A — Causal Audit）

| axis | model | E_natural | E_broken | E_mediator | R_shortcut | R_mediator |
|---|---|---:|---:|---:|---:|---:|
| transit_delay | Teacher | 0.3475 | 0.2071 | 0.2894 | 0.7299 | 0.9701 |
| transit_delay | C0_preS5 | 0.3935 | 0.1165 | 0.3230 | 0.3701 | 0.9003 |
| transit_delay | C1_S5 | 0.3732 | 0.1284 | 0.2848 | 0.3458 | 0.8425 |
| congestion | Teacher | 0.2533 | 0.1172 | 0.1978 | 0.5160 | 0.8892 |
| congestion | C0_preS5 | 0.2029 | 0.1567 | 0.1517 | 0.8119 | 0.7488 |
| congestion | C1_S5 | 0.1778 | 0.1710 | 0.1080 | 1.0206 | 0.6722 |
| parking_cost | Teacher | 0.4504 | 0.3139 | 0.2203 | 0.6871 | 0.5586 |
| parking_cost | C0_preS5 | 0.4591 | 0.4717 | 0.0226 | 1.0309 | 0.0525 |
| parking_cost | C1_S5 | 0.4535 | 0.4642 | 0.0236 | 1.0197 | 0.0588 |

### 目标模式 ΔP（P_X(target)−P_A(target)）

| axis | model | ΔP natural | ΔP broken | ΔP mediator |
|---|---|---:|---:|---:|
| transit_delay | Teacher | -0.1654 | -0.0975 | -0.1378 |
| transit_delay | C0_preS5 | -0.1940 | -0.0214 | -0.1615 |
| transit_delay | C1_S5 | -0.1835 | -0.0399 | -0.1424 |
| congestion | Teacher | -0.1120 | -0.0385 | -0.0949 |
| congestion | C0_preS5 | -0.0972 | -0.0699 | -0.0758 |
| congestion | C1_S5 | -0.0833 | -0.0740 | -0.0521 |
| parking_cost | Teacher | -0.2216 | -0.1526 | -0.1080 |
| parking_cost | C0_preS5 | -0.2279 | -0.2343 | 0.0111 |
| parking_cost | C1_S5 | -0.2256 | -0.2309 | 0.0111 |

## 3. Teacher vs Student 机制对照（Case 判定）

| axis | 判定 | 解读 |
|---|---|---|
| transit_delay | Case 3 — 两者机制一致 | Teacher 与 Student 均机制驱动（R_mediator 0.97/0.84）；蒸馏完整保留了 delay→PT travel_time 通路。 |
| congestion | Case 1 — Teacher 机制、Student 偏 shortcut（蒸馏退化） | Teacher 机制驱动（R_mediator 0.89），Student 偏 shortcut（C0 0.81→C1 1.02 反而退化）；蒸馏部分丢失 congestion→car travel_time 通路。 |
| parking_cost | Case 1 — Student 丢失 Teacher 的 mediator 通路（蒸馏退化） | Teacher 部分依赖 mediator（R_mediator 0.56），Student 几乎不响应 monetary_cost（R_mediator ≈0.05）；蒸馏几乎完全丢失该通路。 |

## 4. 结论

### RQ-C1（Student 响应 mediator 还是 context 标签？）
- **依轴而定**：transit_delay 上响应 mediator；congestion 上两者混合、偏标签；parking_cost 上几乎纯标签。
### RQ-C2（断开通路后是否仍产生大响应？）
- parking_cost 的 broken-path 响应 ≈ natural（R_shortcut≈1.0），说明 Student 对该轴靠标签驱动；
  congestion 的 broken-path 响应在 C1 中也接近 natural（1.02），较 C0（0.81）退化。
### RQ-C3（mediator-only 干预能否复现 natural 效应？）
- transit_delay：能（R_mediator 0.84）；congestion：部分（0.67）；parking_cost：几乎不能（0.06）。
### RQ-C4（unseen persona 上是否保持 Teacher 的因果结构？）
- 在全部 80/36 组（含未见 personas）上汇总，结构与上文一致。
### RQ-C5（S5 多轴训练改善机制一致性还是只改善插值？）
- **只改善插值**：C1 相对 C0 在 congestion 的 R_shortcut 0.81→1.02、R_mediator 0.75→0.67，
  即 S5 联合训练略微加剧了 congestion 的 shortcut，而不是改善机制。

## 5. 能力定级（§19）

- **B — Partial Mechanism Preservation**：transit_delay 机制保留良好，congestion 部分退化，parking_cost 机制基本丢失。

## 6. 论文表述边界（§27）

- 建议表述：**"The Student preserves partial causal consistency under mediator interventions, with axis-dependent fidelity: the transit-delay pathway is well preserved, while parking-cost responses rely on the context label rather than the monetary-cost mediator."**
- 不建议表述："The Student possesses a full causal chain-of-thought."

## 7. 局限

- congestion/parking 仅 36 个 car-可用组（80 personas 中仅 ~45% 有车），统计功效有限。
- Teacher 目标为 K=3（复用 S3）/K=5 聚合，其自身噪声会传导到效应估计。
- mediator 定义仅覆盖 travel_time/reliability/monetary_cost 的直接属性，不含 access_time、transfers 等次级通路。
- 未包含设计 §17 的 sequential MATSim path 对比（系统级多步链路留待后续）。
