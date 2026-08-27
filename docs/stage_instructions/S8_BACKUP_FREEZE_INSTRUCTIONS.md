# S8 Backup & Freeze Instructions
## S8 Supply-Aware Traveler Agent 正式备份冻结执行指令

**对象**：S8 final checkpoint  
**前置**：S7-W3 Generic Behavioral Core v1.0 已冻结且不得修改  
**代码基线**：git commit `65e505f`  
**S8 状态**：Schema Audit = Case B；参数量 24,562；Stop Rule 已满足  
**冻结后角色**：Supply-Aware Traveler Agent v1.0

---

# 1. 冻结决策

批准冻结 S8。

冻结后正式定义：

> S8 = Supply-Aware Traveler Agent v1.0 = FROZEN

后续 Phase C 只能加载该版本，不得继续训练、改权重、改 schema、改 normalization。

S7-W3 继续永久保留：

> S7-W3 = Generic Behavioral Core v1.0

用于论文 baseline 与 generic-core 对照。

---

# 2. 冻结前最终核验

确认以下结果与 `EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md` 完全一致：

- 338 accessibility states
- A-E accessibility classes
- persona / OD / accessibility 三套 holdout
- identity leakage = 0
- Teacher calls = 1,468
- incomplete = 0
- Student-S8 parameters = 24,562
- best_epoch = 26
- PT probability MAE: 0.1336 -> 0.1184
- infeasible-state PT probability: 0.098 -> 0.060
- monotonicity: 0.657 -> 0.686
- legacy KL 相比 S7-W3 改善约 35.8%
- unseen-joint KL 相比 S7-W3 改善约 44.6%
- regression gates 全部通过
- tests: 155 passed / 0 error
- S8 Stop Rule 六项全部满足

任何数字不一致时暂停 Freeze。

---

# 3. 冻结目录

创建独立 release：

```text
releases/
  s8_supply_aware_v1/
    checkpoint/
    config/
    schema/
    normalization/
    accessibility/
    metrics/
    reports/
    data_manifest/
    code_manifest/
    provenance/
    checksums/
    README.md
    FINAL_S8_FREEZE.md
```

禁止写入：

```text
releases/s7_w3_generic_core_v1/
```

---

# 4. Step 1 — 锁定 final checkpoint

确认并记录：

- final checkpoint path
- best_epoch = 26
- training seed
- architecture
- parameter count = 24,562
- initialization source = S7-W3
- optimizer / scheduler / training state（如存在）

复制到：

```text
releases/s8_supply_aware_v1/checkpoint/
```

---

# 5. Step 2 — 冻结配置

保存真实生效的：

- architecture config
- training config
- accessibility feature config
- replay ratio
- learning rate
- loss weights
- batch size
- early stopping
- split rules
- normalization
- R1 / R2 配置
- 最终选中训练配置

输出：

```text
config/student_s8.yaml
config/training_s8.yaml
config/accessibility_features.yaml
```

---

# 6. Step 3 — 冻结 Schema 差异

由于 S8 = Case B，必须保存：

```text
schema/input_schema_s8.json
schema/output_schema_s8.json
schema/feature_order_s8.txt
schema/schema_diff_vs_s7.json
```

`schema_diff_vs_s7.json` 明确记录：

- S7 继承 feature
- S8 新增 feature
- 新增维度
- 新增 encoder
- 旧权重如何迁移
- normalization 是否变化

---

# 7. Step 4 — 冻结 Transit Accessibility 定义

保存实际使用 feature 的：

- 名称
- 单位
- 计算方法
- 范围
- normalization
- 数据来源

至少覆盖实际启用的：

- pt_feasible
- pt_access_time_min
- pt_egress_time_min
- pt_wait_time_min
- pt_in_vehicle_time_min
- pt_transfer_count
- pt_transfer_time_min
- pt_total_door_to_door_time_min
- pt_coverage_ratio
- pt_generalized_cost

输出：

```text
accessibility/feature_definitions.md
accessibility/feature_schema.json
```

---

# 8. Step 5 — 冻结 Singapore Supply Provenance

记录 OSM：

- snapshot date
- study area
- clipping boundary
- CRS
- conversion version

记录 GTFS：

- source
- version/date
- checksum
- routing assumptions
- transfer rule
- walk radius
- service window
- PT validity version

输出：

```text
provenance/singapore_supply.json
provenance/osm_manifest.json
provenance/gtfs_manifest.json
```

必须明确：

> S8 使用真实 Singapore transport supply，但 Teacher 行为标签不是 observed Singapore traveler behavior。

---

# 9. Step 6 — 冻结 S8 Dataset Manifest

记录：

- total states = 338
- A-E 分层
- train / val / test 数量
- persona holdout
- OD holdout
- accessibility holdout
- Teacher repeats
- 1,468 API calls
- incomplete = 0
- identity leakage = 0 hits
- dataset checksum

输出：

```text
data_manifest/s8_accessibility_dataset.json
data_manifest/s8_split_manifest.json
```

---

# 10. Step 7 — 冻结 Teacher Provenance

保存：

- Teacher model
- endpoint
- prompt version
- K policy
- output schema
- aggregation method
- total calls
- incomplete count
- cost / latency summary（如已有）

输出：

```text
provenance/teacher_s8.json
```

---

# 11. Step 8 — Freeze 前复现 Gate

不重新训练，只加载 final S8 checkpoint。

必须重新跑：

## Accessibility
- PT probability MAE
- infeasible-state PT probability
- monotonicity

## Unseen OD
- unseen OD metrics
- identity leakage audit

## Generic regression
- S3 legacy
- S5 seen joint
- S5 unseen joint
- S6/S7 mechanism key metrics

## Singapore smoke
- S8 inference -> adapter
- PT accessibility pipeline
- MATSim minimal execution
- no schema mismatch

结果与最终报告明显不一致则暂停 Freeze。

---

# 12. Step 9 — 冻结最终指标

生成：

```text
metrics/final_metrics.json
metrics/comparison_s7_vs_s8.csv
```

至少包含：

## Accessibility
- KL
- probability L1
- PT probability MAE
- monotonicity
- accessibility sensitivity
- infeasible PT probability
- FVR

## Unseen OD
- KL
- L1
- PT MAE

## Legacy
- accuracy
- KL
- L1
- delta-P gap
- sign agreement

## Multi-axis
- seen joint KL
- unseen joint KL
- interaction error

## Mechanism
- parking G_med
- congestion Gap_shortcut
- 其他最终报告中的关键指标

---

# 13. Step 10 — 冻结关键报告

至少复制：

```text
reports/S8_SCHEMA_AUDIT.md
reports/EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md
```

并建立：

```text
reports/S8_EVIDENCE_INDEX.md
```

索引 S5/S6/S7/S8 与 Singapore Phase B/B.5 的关键证据文件。

---

# 14. Step 11 — 冻结代码版本

当前 S8 完成 commit：

```text
65e505f
```

记录：

- git commit
- git branch
- git status
- Python version
- PyTorch version
- Java version
- MATSim version

若 Freeze 文件本身产生新提交：

```text
freeze: S8 supply-aware traveler agent v1.0
```

---

# 15. Step 12 — Git Tag

创建：

```bash
git tag -a s8-supply-aware-v1.0 -m "Freeze S8 supply-aware traveler agent v1.0"
```

禁止 force 覆盖已有 tag。

---

# 16. Step 13 — SHA256

至少对以下生成 SHA256：

- final checkpoint
- config
- schema
- normalization
- accessibility feature definitions
- final metrics
- dataset manifest
- Teacher provenance
- final S8 report

输出：

```text
checksums/SHA256SUMS.txt
```

---

# 17. Step 14 — 创建 FINAL_S8_FREEZE.md

必须包含：

```text
Model: S8
Release: Supply-Aware Traveler Agent v1.0
Base: S7-W3 Generic Behavioral Core v1.0
Parameters: 24,562
Schema evolution: Case B
Status: FROZEN
```

允许表述：

- supply-aware traveler agent
- city-independent transit-accessibility representation
- real-supply-conditioned behavioral adaptation
- generic behavioral core augmented with transferable supply attributes

禁止表述：

- trained on real Singapore traveler behavior
- universally validated across cities
- full causal reasoning
- exact reproduction of Singapore demand

---

# 18. Step 15 — 写保护与 Phase C Guard

Freeze 后：

- S8 release 只读（如环境支持）
- Phase C 只能 load checkpoint
- 禁止 optimizer resume
- 禁止权重更新
- 禁止改 normalization
- 禁止改 feature schema
- Phase C output 不得写进 release 目录

增加 hard assertion。

Phase C 固定：

```text
Student = frozen S8
N* = 10,000
flowCapacityFactor = 0.3
storageCapacityFactor = 0.3
Singapore supply = frozen Phase B.5 version
PT planner = frozen validated version
```

---

# 19. S7-W3 永久保留

S8 Freeze 不取代 S7。

永久保留：

```text
S7-W3 = Generic Behavioral Core v1.0
S8 = Supply-Aware Traveler Agent v1.0
```

论文中：

- S7-W3 = generic baseline
- S8 = supply-aware extension

---

# 20. Freeze Gate

以下全部满足才允许 Phase C：

- [ ] final checkpoint 唯一
- [ ] best epoch 确认
- [ ] Case B schema diff 完整
- [ ] accessibility definitions 完整
- [ ] Singapore supply provenance 完整
- [ ] S8 dataset manifest 完整
- [ ] Teacher provenance 完整
- [ ] metrics snapshot 完整
- [ ] unseen OD 结果归档
- [ ] generic regression 归档
- [ ] S7 release 0 修改
- [ ] git commit/tag 完成
- [ ] SHA256 完成
- [ ] reproduction gate 通过
- [ ] write guard 启用
- [ ] FINAL_S8_FREEZE.md 已生成

---

# 21. Freeze 后 Phase C 顺序

严格执行：

```text
C0 baseline
-> C1 heavy rain
-> C2 PT fare increase
-> C3 transit delay
-> C4 road disruption
-> C5 joint scenario
```

所有场景使用：

> 同一个 frozen S8 + 同一个 frozen Singapore supply + 同一个 N* + 同一 capacity 配置。

Phase C 中不得根据结果重新训练 Student。

---

# 22. 最终 Agent 汇报

仅需汇报：

1. freeze directory
2. final checkpoint SHA256
3. Freeze commit
4. Git tag
5. S7 release 是否 0 修改
6. reproduction gate 是否通过
7. working tree 是否 clean
8. Phase C guard 是否启用
9. FINAL_S8_FREEZE.md 路径
10. `READY_FOR_PHASE_C=True/False`

---

# 23. 一句话执行原则

> S8 已经完成研究任务。现在把它固化为不可再训练的 Supply-Aware Traveler Agent v1.0；之后 Singapore Phase C 只检验真实网络情景下的行为与系统后果，不再改变模型本身。
