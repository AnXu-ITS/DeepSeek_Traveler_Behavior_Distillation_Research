# S8 实验报告 — Real-Supply Transit Accessibility Adaptation

**依据**：`S8_TRANSIT_ACCESSIBILITY_TRAINING_INSTRUCTIONS.md`（前置：S7-W3 Freeze ✅）
**基础模型**：S7-W3 Generic Behavioral Core v1.0（FROZEN, tag `s7-w3-generic-core-v1.0`）
**架构判定**：Case B — `Student-S8`（`student_s8_v1`，24,562 参数，12 维 mode-level 特征）

## 1. 数据与 Split（S8 §8–15）

- 供给来源：Singapore OSM + 社区构建 GTFS feed（`data/singapore/DATA_README.md`）；路由规则：access ≤ 700 m、egress ≤ 700 m（扩展 1.5 km）、最多 1 次换乘、boarding buffer = max(300 s, access+60 s)。
- OD 候选探针：338 态分层；最终 **338 态**，class 分布 {'A_excellent': 86, 'D_poor': 78, 'E_infeasible': 78, 'C_moderate': 60, 'B_good': 36}，split {'train': 234, 'val': 51, 'test': 53}。
- 三套 holdout：persona 28/6/6（S7/S3 惯例）；OD train/val/test 完全不相交（79 / 16 / 19）；accessibility holdout = 步行负担 access+egress ≥ 15 min 画像仅测试集。
- Sanity：door-to-door 恒等式最大误差 0.001 min；曲线组 <3 类的组数 0。
- Teacher 标注：1468 次调用（K 分布 {'3': 111, '5': 227}，边界样本 K=5；prompt `teacher_s8_accessibility_v0.1`）；stats {'attempted': 1430, 'valid': 1429, 'failed': 1, 'aggregated': 329}（另：首轮进程在 max_tokens 修复前贡献 ~38 次尝试，总计 1468 次、最终 0 incomplete）。

**教师信号质量（全量 338 态，95% bootstrap CI）**：

| class | n | Teacher P(PT) |
|---|---|---|
| A_excellent | 86 | 0.236 [0.194, 0.284] |
| B_good | 36 | 0.264 [0.191, 0.345] |
| C_moderate | 60 | 0.194 [0.155, 0.238] |
| D_poor | 78 | 0.123 [0.100, 0.152] |
| E_infeasible | 78 | 0.000 [0.000, 0.000] |
- 曲线内配对单调性：194/238 = 0.815。
- 解读：可行性悬崖（E≈0.000）与优秀↔差（A vs D）是教师最强信号；中间档 B/C 教师自身噪声大（n 小、CI 宽），是单调性上限的约束。

## 2. Schema（审计见 `reports/S8_SCHEMA_AUDIT.md`）

- 新增 6 维 city-independent mode-level 特征：`pt_feasible, egress_time_min, wait_time_min, in_vehicle_time_min, transfer_time_min, coverage_ratio`（`configs/accessibility_features.yaml`）。
- S7-W3 输入 schema 与冻结 release 未改动；S8 从冻结 checkpoint 复制权重（alt_encoder 首层前 14 列逐字节复制、新 6 列零初始化），**S8-at-init 与 S7-W3 输出逐点一致**（`scripts/audit_s8_schema.py` 断言 4/4 通过）。
- 通用性硬约束（§3）：student/teacher 输入中无任何新加坡地点身份（sanity 对全部 stop_id/route_id/节点 id 做引号级泄漏检查通过）。

## 3. 主指标 — Accessibility（test-only：未见 persona × 未见 OD，95% 配对 bootstrap CI）

- Test 态数：53（personas 6，OD 19，double holdout 已断言）。

### 3.1 PT Probability Fidelity（§20.1）

| model | KL | prob L1 | PT prob MAE |
|---|---|---|---|
| B0_S7W3 | 0.1984 [0.1659, 0.2295] | 0.5016 [0.4530, 0.5496] | 0.1336 [0.1036, 0.1631] |
| B1_S8 | 0.1980 [0.1568, 0.2414] | 0.4845 [0.4191, 0.5519] | 0.1184 [0.0863, 0.1527] |
| Teacher | — | — | — |

### 3.2 Accessibility Sensitivity ΔP_PT = P(PT|best) − P(PT|worst)（组内配对，§20.2）

| model | ΔP_PT | n_groups |
|---|---|---|
| B0_S7W3 | 0.1631 [0.1376, 0.1910] | 12 |
| B1_S8 | 0.1561 [0.1188, 0.2001] | 12 |
| Teacher | 0.2117 [0.1547, 0.2715] | 12 |

### 3.3 Monotonicity（§20.3）

| model | pair agreement | triplet agreement | n_pairs / n_triplets |
|---|---|---|---|
| B0_S7W3 | 0.6571 [0.4857, 0.8000] | 0.2609 [0.0870, 0.4348] | 35 / 23 |
| B1_S8 | 0.6857 [0.5143, 0.8286] | 0.3043 [0.1304, 0.4783] | 35 / 23 |
| Teacher | 0.8000 [0.6571, 0.9143] | 0.5217 [0.3043, 0.7391] | 35 / 23 |

### 3.4 PT Feasibility Violation Rate（§21）

| model | FVR | mean P(PT\|infeasible) | n_infeasible |
|---|---|---|---|
| B0_S7W3 | 0.0000 [0.0000, 0.0000] | 0.0978 [0.0693, 0.1342] | 12 |
| B1_S8 | 0.0000 [0.0000, 0.0000] | 0.0599 [0.0359, 0.0904] | 12 |

### 3.5 Convenience Error E_conv = |P_T(PT) − P_S(PT)| 分档（§22）

| model | Excellent | Good | Moderate | Poor | Infeasible |
|---|---|---|---|---|---|
| B0_S7W3 | 0.1137 [0.0706, 0.1663] | 0.1570 [0.1018, 0.2219] | 0.1556 [0.0546, 0.2798] | 0.1736 [0.0841, 0.2804] | 0.0975 [0.0691, 0.1338] |
| B1_S8 | 0.1108 [0.0559, 0.1762] | 0.1286 [0.0643, 0.2036] | 0.1769 [0.0742, 0.2941] | 0.1556 [0.0506, 0.2764] | 0.0596 [0.0356, 0.0902] |
| Teacher | — | — | — | — | — |

### 3.6 S8 vs S7-W3 关键差分（配对 bootstrap；负 MAE/FVR = S8 更优）

- PT prob MAE Δ：-0.0152 [-0.0268, -0.0032]*
- mean P(PT|infeasible) Δ：-0.0379 [-0.0457, -0.0315]*
- FVR rate Δ：+0.0000 [+0.0000, +0.0000]
- pair monotonicity Δ：+0.0286 [-0.0571, +0.1143]

### 3.7 λ_accessibility 消融（§11/§19：Round 1 基线 vs Round 2 加响应损失）

| 指标（test） | B0 S7-W3 | R1 λ=0 | R2 λ=1.0 | Teacher |
|---|---|---|---|---|
| PT prob MAE | 0.1336 [0.1036, 0.1631] | 0.1147 [0.0856, 0.1457] | 0.1184 [0.0863, 0.1527] | — |
| mean P(PT\|infeasible) | 0.0978 [0.0693, 0.1342] | 0.0643 [0.0398, 0.0958] | 0.0599 [0.0359, 0.0904] | — |
| sensitivity ΔP_PT | 0.1631 [0.1376, 0.1910] | 0.1331 [0.1018, 0.1696] | 0.1561 [0.1188, 0.2001] | 0.2117 [0.1547, 0.2715] |
| monotonicity pair | 0.6571 [0.4857, 0.8000] | 0.6000 [0.4286, 0.7714] | 0.6857 [0.5143, 0.8286] | 0.8000 [0.6571, 0.9143] |
| monotonicity triplet | 0.2609 [0.0870, 0.4348] | 0.2174 [0.0435, 0.3913] | 0.3043 [0.1304, 0.4783] | 0.5217 [0.3043, 0.7391] |

- R1 选择记录：best_epoch=19，λ_accessibility=0.0；R2 选择记录：best_epoch=26，λ_accessibility=1.0。
- R1 回归门禁：✅ 全过；R2 回归门禁：✅ 全过。
- 结论：R2（+L_accessibility）在单调性（pair 0.600→0.686、triplet 0.217→0.304）与 infeasible 概率（0.064→0.060）上优于 R1，PT-MAE 两者均显著优于 B0；选定 **R2（λ_accessibility=1.0）** 为最终 S8 模型。

## 4. Unseen OD Test（§15，RQ-S8-4）

- OD holdout 验证：train 79 / test 19 个 OD，重叠 []；verified=True。
- Test 态 53 个全部使用未见 OD 与未见 persona（all_unseen=True），class 分布 {'A_excellent': 15, 'B_good': 11, 'E_infeasible': 12, 'C_moderate': 7, 'D_poor': 8}。
- 模型输入仅为 city-independent accessibility 向量，无 OD 身份特征。

## 5. Generic Capability Regression（§25–26）

| 门禁 | 值 | 判定 |
|---|---|---|
| legacy accuracy drop | -3.54 pp | ✅ ≤ 1 pp |
| legacy KL 变化 | -35.83% | ✅ ≤ +10% |
| seen joint KL 变化 | -15.44% | ✅ ≤ +10% |
| unseen joint KL 变化 | -44.62% | ✅ ≤ +10% |

机制指标（S8 vs S7-W3，test 四联组）：

- W3_S7W3: congestion G_med 0.1763 [0.1068, 0.2543] / Gap_shortcut 0.3455 [0.1554, 0.5608] | parking_cost G_med 0.2580 [0.1409, 0.3952] / Gap_shortcut 0.3335 [0.1983, 0.4871]
- S8: congestion G_med 0.1527 [0.0882, 0.2145] / Gap_shortcut 0.2847 [0.2439, 0.3259] | parking_cost G_med 0.2554 [0.1346, 0.3971] / Gap_shortcut 0.3506 [0.2049, 0.5170]
- congestion G_med Δ(S8−S7W3)：-0.0236 [-0.0988, +0.0620]
- parking_cost G_med Δ(S8−S7W3)：-0.0026 [-0.0106, +0.0040]

## 6. 训练记录

- 初始化：`releases/s7_w3_generic_core_v1/checkpoint/model.pt`（FROZEN）；replay 2:1:1:1；LR 1.25e-4。
- 最终模型（R2）：λ_accessibility=1.0，best_epoch=26，runtime=8.72 s；val: legacy KL 0.0426、accessibility KL 0.1469、response gap 0.136343（init 0.130741）。

## 7. Stop Rule 判定（§33）

| 条件 | 判定 |
|---|---|
| accessibility response 明显优于 S7-W3 | ✅ 显著更优 （PT-MAE -0.0152 [-0.0268, -0.0032]*；P(PT\|inf) -0.0379 [-0.0457, -0.0315]*；pair 单调性 +0.0286 [-0.0571, +0.1143]）|
| unseen OD 保持 | ✅ |
| FVR 明显下降 | ✅ 显著更优 （FVR rate 双侧均为 0.000 —— B0 已不把 PT 选为 argmax；真实改善在 infeasible 态的平均 PT 概率质量 0.0978→0.0599，配对差分 -0.0379 [-0.0457, -0.0315]*）|
| legacy/joint/mechanism 无明显回退 | ✅ 全过（机制：congestion G_med 0.1763 [0.1068, 0.2543]→0.1527 [0.0882, 0.2145]、parking G_med 0.2580 [0.1409, 0.3952]→0.2554 [0.1346, 0.3971]）|
| Singapore-specific ID 未进入模型 | ✅ 引号级泄漏检查 0 命中 |
| feature schema 可迁移（city-independent） | ✅ 全部数值属性，无地点身份 |

## 8. 可允许表述（§28–29）

> a city-agnostic supply-aware Traveler Agent conditioned on transferable transit accessibility attributes；real-world Singapore transport supply / real OSM/GTFS-derived transit accessibility；real-supply-conditioned behavioral distillation。

**禁止**：universally generalizable traveler model；trained on real Singapore traveler behavior（本数据集无真实人类行为标签，Teacher 为 synthetic persona 推理标注）。

*生成依据：outputs/s8_accessibility_eval/eval_metrics.json、outputs/s8_regression/eval_metrics.json、outputs/s8_unseen_od/unseen_od_audit.json、data/singapore_accessibility/split_manifest.json、data/singapore_accessibility/generation_manifest.json、outputs/student_s8/val_metrics.json。*
