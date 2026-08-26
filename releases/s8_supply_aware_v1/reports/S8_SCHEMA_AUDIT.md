# S8 Schema Audit

**依据**：`S8_TRANSIT_ACCESSIBILITY_TRAINING_INSTRUCTIONS.md` §4–5  
**审计对象**：S7-W3 Generic Behavioral Core v1.0（FROZEN，`releases/s7_w3_generic_core_v1/`，tag `s7-w3-generic-core-v1.0`）  
**审计日期**：2026-08-26

---

## Step 1 — S7-W3 Freeze 确认（S8 前置条件）

- [x] `releases/s7_w3_generic_core_v1/` 存在，46 文件全量 SHA256 冻结，全部 read-only。
- [x] Git tag `s7-w3-generic-core-v1.0` → commit `d672cf3`。
- [x] 冻结 checkpoint `releases/s7_w3_generic_core_v1/checkpoint/model.pt`（SHA256 `8263faec…b2b6`，24,370 参数）。
- [x] S7-W3 复现性 gate 7/7 通过（Δ=0.0000）。

→ S8 允许启动；S8 只 load S7-W3，不修改其本体。

## 审计问题（§4）逐项回答

### Q1. 当前是否已有 PT travel time / monetary cost / reliability / availability？

| 属性 | S7-W3 状态 | 说明 |
|---|---|---|
| PT travel time | ✅ 已有 | `TravelAlternative.travel_time_min`（per-mode；S7 中为合成值，S8 将用真实 door-to-door `T_PT` 填充 pt） |
| monetary cost | ✅ 已有 | `TravelAlternative.monetary_cost` |
| reliability | ✅ 已有 | `TravelAlternative.reliability_delay_min` |
| availability | ✅ 已有 | `TravelAlternative.available` → `alt_mask`（masked softmax） |

### Q2. 是否已有 mode-level feature encoder？

✅ 已有。`TravelerStudent.alt_encoder`：
`Linear(mode_emb_dim 8 + n_alt_num 6 = 14 → alt_hidden 32) → ReLU → Linear(32→32) → ReLU`。
输入 = mode embedding（8 维）+ 6 个 mode-level 数值特征（z-score）。

S7-W3 冻结 ALT_NUM（`schema/feature_order.txt`）：
`travel_time_min, monetary_cost, access_time_min, transfers, reliability_delay_min, weather_exposure`

### Q3. 是否可以追加 access/egress/wait/transfer_count/transfer_time/coverage_ratio？

| S8 特征（§6–7） | 映射方式 |
|---|---|
| `pt_feasible` | 已有槽位：`available` 仅表达“mode 存在”；S8 语义是“时间窗内有可行 itinerary” → **新增数值特征**（见 Q3 注），不由 mask 承担（否则 FVR 恒 0，§21 失去意义） |
| `pt_access_time_min` | **已存在**（`access_time_min`），直接用真实值 |
| `pt_egress_time_min` | ❌ 新增 |
| `pt_wait_time_min` | ❌ 新增 |
| `pt_in_vehicle_time_min` | ❌ 新增 |
| `pt_transfer_count` | **已存在**（`transfers`），直接用真实值 |
| `pt_transfer_time_min` | ❌ 新增 |
| `pt_total_door_to_door_time_min` | 已有槽位：pt 的 `travel_time_min`（`T_PT = T_access+T_wait+T_invehicle+T_transfer+T_egress`） |
| `pt_coverage_ratio` | ❌ 新增（`R_coverage = D_PT,vehicle / D_OD`，clip [0,1]） |
| `pt_generalized_cost` | **不加**（§7.10：不发明固定价值时间参数，保留原始 feature） |

### Q4. 新 feature 是否会改变 input dimension？

**会。** `n_alt_num` 6 → 12（新增 6 维：`pt_feasible, egress_time_min, wait_time_min, in_vehicle_time_min, transfer_time_min, coverage_ratio`）。

```text
pt_feasible            (0/1)
pt_egress_time_min
pt_wait_time_min
pt_in_vehicle_time_min
pt_transfer_time_min
pt_coverage_ratio      [0,1]
```

→ `alt_encoder` 首层输入 14 → **20**；global encoder / scorer / departure head / embeddings 不变。
参数量变化：alt_encoder.0.weight 32×14 → 32×20，**+32×6 = +192 参数**（24,370 → **24,562**）。

### Q5. 能否保持旧 checkpoint 权重兼容？

**可以（部分复制）。**

- 可逐字节复用：global_encoder、scorer、departure_head、全部 embedding、alt_encoder.1/2/3（bias 与后续层）、mode_embedding。
- 需要扩展：`alt_encoder.0.weight`（32,14）→（32,20）：前 14 列逐字节复制自 S7-W3，新增 6 列**零初始化**。
- 零初始化的性质：新 feature 经 z-score（S8 train 上拟合）后输入，乘零权重 → 贡献恒 0 ⇒ **S8-at-init 与 S7-W3 行为逐点一致**（可验证的初始化不变量，作为 sanity check）。
- FeatureExtractor：新 feature 的 mean/std 在 S8 train split 上拟合；6 个旧 feature 的 vocab/mean/std 沿用冻结 extractor_state（不修改冻结 release）。

---

## Case 判定（§5）：**Case B**

> 必须新增 input dimension（6 维新 mode-level feature）→ 建立 `Student-S8`：
> - 复制 S7-W3 可复用权重（alt_encoder 首层部分复制 + 新列零初始化）；
> - 新增 feature encoder 初始化 = 零（记录在案）；
> - **不覆盖** S7-W3 原 input schema（冻结 release 保持字节不变）；
> - 新 architecture version 明确记录：`student_s8_v1`，24,562 参数。
> - 禁止声称 S8 与 S7 完全相同架构。

### 通用性硬约束（§3）遵守方案

| 禁止输入 | 保证方式 |
|---|---|
| 新加坡 district / Tampines / 站名 / stop_id / route_id / postal / 地理身份 | 数据管线只输出数值 accessibility 向量；state 中不含任何 ID 字符串（origin/destination 以内部匿名 index 表示，不入 feature） |
| 允许输入 | travel/access/egress/wait/in-vehicle/transfer 时间、transfer 数、coverage、feasibility、reliability、monetary cost、walking burden（= access+egress） |

### 后续管线承诺（S8 剩余步骤）

1. `src/traveler_distillation/accessibility/`：`gtfs_accessibility.py`（真实供给 → itinerary → 特征）、`accessibility_features.py`（Student-S8 + 扩展 extractor + 权重复用）、`accessibility_dataset.py`（分层采样 + 三套 split）。
2. 数据：`data/singapore_accessibility/`（stratified OD pool + states + split manifest + 无任何地点身份；accessibility holdout = 高步行负担画像仅测试集，因 planner 按路由规则上限 1 次换乘）。
3. 训练：`outputs/student_s8/`（从 S7-W3 初始化，replay 2:1:1:1，输出目录必须经 `assert_not_frozen_output`）。
4. 评估：accessibility sensitivity / unseen OD / FVR / generic regression，全部 test-only。
