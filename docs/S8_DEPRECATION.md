# S8 Deprecation Record

**Date**: 2026-08-27
**Object**: S8 Supply-Aware Traveler Agent v1.0 (`releases/s8_supply_aware_v1/`, tag `s8-supply-aware-v1.0`)
**Status**: **DEPRECATED** — superseded by S9 Supply-Aware Traveler Agent v2.0
(`releases/s9_supply_aware_v2/`, tag `s9-supply-aware-v2.0`).

## 1. 废弃原因：明确的数据错误（walk/bike 旅行时间）

S8 数据管线（`src/traveler_distillation/accessibility/gtfs_accessibility.py` 的
`SupplyIndex` 与 `build_real_alternatives`）用 **道路自由流速度**（`length / freespeed`）
计算 walk/bike 的旅行时间，导致：

- 走路/骑车的有效速度中位数 **29.1 km/h**（实际步行 4.3–5.0 km/h、骑行 12–16 km/h）；
- 338 个训练态中 **0/338** 出现 "pt 快于 walk"（用现实速度重算应为 ~75%）；
- 模型（与 Teacher，标注时看到同一 biased alternatives）学到的世界是
  "walk/bike ≫ pt" → S8 的 Phase C baseline pt share 仅 2.8%、C3 delay 15min 使 pt 归零；
- MATSim 侧 walk/bike 作为 network modes 按 link freespeed 行驶并占用道路容量。

**发现过程**：用户对 Phase C 结果（pt 2.8%、delay 后 pt 归零）提出质疑 →
事件级与数据集级分析定位到速度错误（证据：114-OD 池有效速度分布、
338 态 pt-vs-walk 对比、示例 OD 2.98 km walk=13.3 min ≈ 13.4 km/h）。

## 2. 治理依据

`NEXT_STEP_PLAN_SINGAPORE_AIT.md` §0：「重新训练主模型的唯一条件：发现明确的
数据错误或方法错误」。本次为明确数据错误，用户于 2026-08-27 批准重训（S9）。

## 3. 处置方式

- **S8 冻结 release 与 tag 保持字节不变、不删除**（冻结制审计价值）；
  git 历史保留全部 S8 训练/冻结 commit。
- S8 的 Phase C 结果（`outputs/singapore_phase_c/`）**作废**，论文不得引用；
  新 Phase C 以 frozen S9 重跑（`outputs/singapore_phase_c_s9/` 或后续指定目录）。
- S8 旧数据集已归档至 `data/singapore_accessibility_s8_legacy/`（供审计对照）。
- 论文中 supply-aware extension = S9；S8 仅在方法/消融叙事中作为
  "数据管线缺陷→修正"的诚实记录（可选）。

## 4. S9 修正内容

| 项 | S8 | S9 |
|---|---|---|
| walk 速度 | link freespeed | 1.34 m/s |
| bike 速度 | link freespeed | 4.17 m/s |
| pt access/egress 步行 | link freespeed | 1.34 m/s |
| MATSim walk/bike | network modes | teleported（walk 1.39 / bike 3.9 m/s） |
| 数据集 | 338 态（biased） | 336 态（重建+重新标注 1,502 次调用） |

S9 评估与冻结记录：`releases/s9_supply_aware_v2/FINAL_S9_FREEZE.md`。
