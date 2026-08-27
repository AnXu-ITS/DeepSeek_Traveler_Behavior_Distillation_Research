# S9 实验报告 — Real-Supply Transit Accessibility Adaptation v2（修正后重训）

**依据**：用户批准的重训（2026-08-26/27）；S8 因 walk/bike 旅行时间数据错误废弃
（见 `docs/S8_DEPRECATION.md`）。
**基础模型**：S7-W3 Generic Behavioral Core v1.0（FROZEN, tag `s7-w3-generic-core-v1.0`）
**架构**：Case B — `student_s8_v1`（与 S8 相同：24,562 参数，12 维 mode-level 特征）
**与 S8 的唯一差异**：数据集（修正后的 walk/bike 速度）。

## 0. S8 数据错误与 S9 修正

| 量 | S8（错误） | S9（修正） |
|---|---|---|
| walk 旅行时间 | length / 道路自由流速度（有效速度中位 29.1 km/h） | length / 1.34 m/s（实测有效 3.28 km/h） |
| bike 旅行时间 | length / freespeed（29.1 km/h） | length / 4.17 m/s（实测 10.14 km/h） |
| pt access/egress 步行 | length / freespeed | length / 1.34 m/s |
| car | length / freespeed（不变，31.09 km/h） | 不变 |
| MATSim walk/bike 执行 | network modes（按 link freespeed 行驶、占道路容量） | teleported（walk 1.39 m/s / bike 3.9 m/s） |

修正后世界：336 态中 **pt 快于 walk 的占 234/336（70%）**（S8 数据为 0/338）。

## 1. 数据与 Split（重建）

- 336 态：class {A_excellent: 79, B_good: 16, C_moderate: 77, D_poor: 84, E_infeasible: 80}；
  split {train: 234, val: 51, test: 51}；OD pool {train 79, val 14, test 20} 不相交。
- 三套 holdout：persona 28/6/6；OD 不相交；accessibility holdout = 步行负担 ≥15 min 仅测试集。
- Sanity：door-to-door 恒等式最大误差 0.001 min；<3 类曲线组 0；身份泄漏 0 命中。
- Teacher 标注：**1,502 次有效调用**（K={3: 89, 5: 247}，边界 K=5；49 次瞬时 SSL 失败已重试），
  336/336 聚合，0 incomplete。prompt `teacher_s8_accessibility_v0.1` 不变。

**教师信号（全量 336 态）**——修正后呈现真实 accessibility 梯度：

| class | n | Teacher P(PT) |
|---|---|---|
| A_excellent | 79 | 0.448 |
| B_good | 16 | 0.429 |
| C_moderate | 77 | 0.372 |
| D_poor | 84 | 0.189 |
| E_infeasible | 80 | 0.000 |

（S8 数据中教师梯度被压缩到 A 0.236 / D 0.123 —— 因为当时"走路 6 倍速"压平了模式差异。）

## 2. 训练

- 从 frozen S7-W3 初始化（Case B 权重迁移同 S8）；replay 2:1:1:1；LR 1.25e-4；
  λ_accessibility=1.0（S8 的 R1/R2 消融结论沿用，直接跑 R2 配置）。
- best_epoch=17（early stop），runtime 11.34 s，24,562 参数。
- val：accessibility KL 0.2993→0.2362；response gap 0.4484→0.3488；
  legacy KL 0.0445→0.0442（无回退）。

## 3. 主指标 — Accessibility（test-only：51 态，未见 persona × 未见 OD，95% 配对 bootstrap）

| 指标 | B0 S7-W3 | S9 | Teacher |
|---|---|---|---|
| PT prob MAE | 0.1750 | **0.1346** | — |
| mean P(PT\|infeasible) | 0.2924 | **0.2130** | 0.000 |
| FVR rate | 0.3333 | **0.0833** | 0.000 |
| pair monotonicity | 0.6316 | **0.6579** | 0.7895 |
| sensitivity ΔP_PT | 0.0845 | **0.1527** | 0.4480 |

**S9 vs S7-W3 配对差分（* = CI 不含 0）**：
- PT prob MAE Δ：**-0.0404 [-0.0529, -0.0279]\***
- mean P(PT|infeasible) Δ：**-0.0794 [-0.1004, -0.0611]\***
- FVR rate Δ：**-0.2500 [-0.5000, -0.0833]\***（修正后的世界里 FVR 有真实意义：B0 33.3% → S9 8.3%）
- pair monotonicity Δ：+0.0263 [-0.1053, +0.1579]

## 4. Unseen OD Test

- OD holdout：train 79 / test 20，重叠 []，verified=True；test 51 态全部使用未见 OD × 未见
  persona（all_unseen=True）。

## 5. Generic Capability Regression（§25–26 同口径）

| 门禁 | 值 | 判定 |
|---|---|---|
| legacy accuracy drop | -4.43 pp | ✅ ≤ 1 pp |
| legacy KL 变化 | -26.04% | ✅ ≤ +10% |
| seen joint KL 变化 | -3.33% | ✅ ≤ +10% |
| unseen joint KL 变化 | -36.45% | ✅ ≤ +10% |

机制（test 四联组）：congestion G_med Δ(S9−S7W3) -0.0064 [-0.0823, +0.0798]、
parking G_med Δ -0.0085 [-0.0192, +0.0014] —— CI 均含 0，无明确回退。

## 6. Stop Rule 判定（六项）

1. accessibility response 明显优于 S7-W3：✅（PT-MAE、P(PT|inf)、FVR 三项差分 CI 均不含 0）
2. unseen OD 保持：✅（all_unseen=True，重叠 []）
3. FVR 明显下降：✅（0.333→0.083，Δ -0.25*）
4. legacy/joint/mechanism 无明显回退：✅（门禁全过）
5. Singapore-specific ID 未进入模型：✅（引号级检查 0 命中）
6. feature schema 可迁移：✅（同一 city-independent schema）

**判定：满足 → Freeze S9。**

## 7. 诚实边界

- S8 已废弃（walk/bike 速度错误），其 Phase C 结果作废；本报告为修正后版本。
- Teacher 标签仍为 synthetic-persona LLM 推断，非真实新加坡出行行为；禁止
  "trained on real Singapore traveler behavior"。
- GTFS 为社区构建 feed（MRT frequency-based）；供给仅 Tampines + Pasir Ris 子区域。
- 需求侧 synthetic personas/trips；mode share 不宣称复现新加坡真实分担率。

*生成依据：outputs/s9_accessibility_eval/eval_metrics.json、outputs/s9_regression/eval_metrics.json、
outputs/s9_unseen_od/unseen_od_audit.json、data/singapore_accessibility/{split,generation}_manifest.json、
outputs/student_s9/val_metrics.json。*
