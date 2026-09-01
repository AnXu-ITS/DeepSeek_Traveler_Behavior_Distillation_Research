# DATA_ISOLATION_CHECK — Reference Pipeline 开发前数据隔离与实验冻结检查

> 检查日期：2026-09-01 · 目的：在 Reference Pipeline 封装开始前，确认数据 / 模型 /
> benchmark / cache / output 不存在覆盖或交叉污染风险。
> 结论速览：**冻结面清晰，但存在 9 项已识别的污染/覆盖风险（R1–R9），
> Reference Pipeline 必须在新命名空间开发，并实现带绑定元数据的 cache 层。**

---

## 1. 冻结清单（Frozen Registry）——只读，禁止任何写入

### 1.1 已训练 Student checkpoint 与 release 包

| 路径 | 内容 | 状态 |
|---|---|---|
| `releases/s7_w3_generic_core_v1/` | S7-W3 Generic Behavioral Core v1.0（24,370 参数，论文 generic baseline） | ❄️ FROZEN |
| `releases/s8_supply_aware_v1/` | S8 Supply-Aware v1.0（walk/bike 速度数据错误） | ❄️ FROZEN（DEPRECATED，字节不变） |
| `releases/s9_supply_aware_v2/` | **S9 Supply-Aware v2.0（24,562 参数，论文主模型）** | ❄️ FROZEN |

- 硬保护：`src/traveler_distillation/student/release_guard.py` 的
  `assert_not_frozen_output()` 对解析进这三个目录的写路径直接抛 `RuntimeError`。
  Reference Pipeline 的所有输出函数必须调用它。
- 复现 gate：`python scripts/freeze_s9_release.py verify-gate`（12/12 指标 Δ=0.0000）。

### 1.2 训练时 normalization / scaler / category mapping

| 冻结件 | 路径 |
|---|---|
| z-score 统计（scaler） | `releases/s9_supply_aware_v2/normalization/normalization.json`（S7/S8 同结构） |
| 类别映射 | `releases/s9_supply_aware_v2/normalization/category_mapping.json`、`mode_mapping.json` |
| 特征顺序与 schema | `releases/s9_supply_aware_v2/schema/feature_order_s9.txt`、`input_schema_s9.json`、`output_schema_s9.json`、`schema_diff_vs_s7.json` |
| 可达性特征定义 | `releases/s9_supply_aware_v2/accessibility/feature_definitions.md`、`feature_schema.json`、`config/accessibility_features.yaml` |

⚠️ **R2 特别警示**：`configs/accessibility_features.yaml`（live）与 S9 release 内副本都仍写
`walk_speed_ms: 1.2  # network-derived walk tt (length/free)`——这是 **S8 时代的错误注释**。
S9 的有效路由参数在代码与修正记录里，不在这个 YAML 字段里：
`src/traveler_distillation/accessibility/gtfs_accessibility.py`（walk 1.34 m/s、bike 4.17 m/s、
car length/freespeed）与 `releases/s9_supply_aware_v2/provenance/data_fix.json`。
**Reference Pipeline 若"照 YAML 重建路由"会复现 S8 数据错误。**
有效参数来源 = 冻结 release 的 provenance + 对应代码 commit（见 `code_manifest/git_commit.txt`）。

### 1.3 正式 benchmark 输入（frozen）

| 数据 | 路径 | 说明 |
|---|---|---|
| Singapore 真实供给 | `data/singapore/`（`gtfs/raw/`、`osm/`、`transit/`） | OSM 快照 2026-08-25 bbox；社区 GTFS sha256 `1fcc6f5f…`；构建产物 `network_with_transit.xml`（47 MB）、`transitSchedule.xml`（519 MB，全天 20,966 班次）、`transitVehicles.xml`、`prep_stops.jsonl`、`trips_by_stop.json`、`stop_snapping_report.json`、`activity_nodes.json` |
| S9 可达性数据集（**修正后**） | `data/singapore_accessibility/` | records.jsonl（338 states）、split_manifest.json（三重 holdout）、states_with_teacher.jsonl、boundary_pairs.json 等 |
| S8 可达性数据集（**错误速度，legacy**） | `data/singapore_accessibility_s8_legacy/` | 与 S9 目录**文件几乎同名**（R1）；永不被 Reference 引用 |
| E5 Helsinki 供给 | `data/helsinki/`、`hsl/`（OSM PBF + GTFS zip） | E5 zero-shot 输入，frozen |
| 历史实验数据集 | `data/student_s7_mechanism/`、`data/student_s5_joint/`、`data/student_v0_*/`、`data/causal_audit/`、`data/population_experiment/`、`data/phase10_loop/` | 训练/早期实验输入；Reference Pipeline 不重训、不消费 |
| 冻结实验设置 | N\*=10,000（seed 2026）、capacity 0.3/0.3、`lastIteration=0`、PT 规划规则（access 700 m、egress 700 m+1.5 km 延伸、max 1 换乘、boarding buffer、connection 180–2700 s、departure window 1 h） | 记录于 `data/singapore/DATA_README.md` 与 Phase C 报告 |

### 1.4 已生成正式结果（frozen）

| 结果 | 路径 | 说明 |
|---|---|---|
| 论文证据快照（git 跟踪） | `evidence/`（`phase_c_s9/`、`s9_eval/`、`e1_mnl/`–`e5_helsinki/`） | 全部论文数字的来源；索引见 `evidence/README.md` |
| 正式报告（git 跟踪） | `reports/`（PHASE_C_SINGAPORE_REPORT.md、EXPERIMENT_REPORT_S9_* 等） | 冻结文本 |
| Phase C 原始结果 | `outputs/singapore_phase_c_s9/`（C0–C5 + `phase_c_records.json`） | E3/E4 G 门禁的比对锚 |
| S9 评估/冒烟 | `outputs/s9_accessibility_eval/`、`outputs/s9_regression/`、`outputs/s9_unseen_od/`、`outputs/s9_singapore_smoke/` | 冻结快照 |
| E1–E5 运行产物 | `outputs/e1_mnl/`–`outputs/e5_helsinki/` | 冻结（部分 GB 级未入库，SHA 见 `evidence/README.md` §3） |

### 1.5 原 legacy pipeline 输出（frozen）

| 产物 | 路径 |
|---|---|
| 归档工具/数据/产物 | `archive/legacy_20260820/`、`archive/legacy_20260821/` |
| Phase 9/10（合成网络闭环）数据 | `data/phase10_loop/`（baseline/rain/fare_surge + phase10_results.json） |
| S8 废弃线全部输出 | `outputs/s8_accessibility_eval*/`、`outputs/s8_regression*/`、`outputs/s8_unseen_od/`、`outputs/s8_singapore_smoke/`、`outputs/student_s8*/` |
| v0.x / S5–S7 训练线与评估 | `outputs/student_v0_*/`、`outputs/student_s5_*/`、`outputs/student_s7_*/`、`outputs/s5_*/`、`outputs/s7_*/`、`outputs/teacher_audit_v0_1/`、`outputs/causal_audit/` |
| Latency 审计产物（含 cache 与 ONNX） | `outputs/pipeline_benchmark/`（`od_accessibility_cache.pkl`、`s9.onnx`） |
| 开发期探针残留 | `outputs/_g3_default_path_check/`、`outputs/_probe_profile/` |
| 用户资料（与本实验无关，勿动） | `新加披调查问卷/`、根目录 `屏幕截图 2026-09-01 155331.png`（git 未跟踪） |

---

## 2. Reference Pipeline 开发目录（全部新建，独立命名空间）

```text
data/reference_smoke/              # dev-only example/smoke 输入（persona/trip/OD/context fixture）
outputs/reference_pipeline/        # Reference Pipeline 全部输出（新建，唯一写入根）
  cache/                           #   smoke-test cache（带绑定元数据，见 §3.3）
  smoke/                           #   smoke 场景运行目录（每情景一个子目录）
  regression/                      #   冻结 benchmark 只读回归结果（后置 gate，见 §6）
scripts/reference/                 # Reference Pipeline 封装脚本（git 跟踪）
docs/reference/                    # Reference 设计与 runbook（git 跟踪）
```

规则：
1. Reference Pipeline 的**任何**脚本不允许向 `data/singapore*/`、`releases/`、`evidence/`、
   `reports/`、`archive/` 及 §1.4/§1.5 列出的既有 `outputs/` 目录写入。
2. `data/`、`outputs/` 已被 `.gitignore` 忽略——dev fixture、cache、输出默认不入库（正确，
   保持一致）；只有 `scripts/reference/` 与 `docs/reference/` 会被 git 跟踪。
3. smoke fixture 用 **dev 专用 seed**（建议 `seed=0` + `"reference"` 标记）独立生成，
   不从 `data/singapore_accessibility/` 的任何 split 抽样（R4/R8）。
4. 冻结 supply 文件可以**只读复用**（它们是输入不是 cache，见 §3.1），每次运行先做
   SHA256 校验（复用 `data/singapore/gtfs/raw/checksum.sha256` 与
   `releases/s9_supply_aware_v2/provenance/` 中的供给 SHA）。
5. 所有 Reference 输出路径先过 `assert_not_frozen_output()`。

---

## 3. Cache 治理

### 3.1 可以复用（只读，性质是"共享输入"而非 cache）

| 项 | 路径 | 复用条件 |
|---|---|---|
| 冻结 Singapore 供给 | `data/singapore/transit/network_with_transit.xml`、`transitSchedule.xml`、`transitVehicles.xml`、`prep_stops.jsonl`、`trips_by_stop.json`、`stop_snapping_report.json`、`activity_nodes.json` | 只读 + SHA256 校验；禁止以默认路径重建（R6） |
| 冻结模型与编码统计 | `releases/s9_supply_aware_v2/checkpoint/model.pt`、`normalization/*.json`、`schema/*` | 只读加载 |
| 生成器参数 | `configs/generation_v0_1.yaml` | 只读；fixture 输出仍写 dev 目录 |
| 执行引擎 | `tools/matsim-2026.0-release/`、`tools/java/`（RunMatsimPreloaded） | 只读调用 |

### 3.2 必须隔离 / 禁止复用的 cache

| Cache | 位置 | 判定 |
|---|---|---|
| `od_accessibility_cache.pkl` | `outputs/pipeline_benchmark/` | **禁止复用**。latency 审计产物：纯 `[(OD, dep), acc]` 列表，**无任何元数据绑定**（无网络 SHA、无模式、无路由规则版本、无代码 commit）。且该目录属冻结审计产物，重跑 benchmark 脚本会覆盖它（覆盖侧见 O3）。 |
| S8-legacy 特征/数据集 | `data/singapore_accessibility_s8_legacy/`、`outputs/s8_*/` | **禁止复用**（错误速度）。 |
| 进程内 cache（`_sp_cache`、`_walk_cache`、`pair_cache`、`run_phase_c` 的 `_tt_cache`/`acc_cache`） | 内存 | 进程退出即消失，不落盘、无污染；但 Reference Pipeline **不得**把这些 ad-hoc 结构直接 pickle 持久化——必须走 §3.3 的绑定格式。 |
| S8 训练统计/映射 | `releases/s8_supply_aware_v1/normalization/` | 只存档，禁止用于编码。 |

### 3.3 Reference cache 的强制绑定键（route/accessibility cache 设计约束）

要求：**route/accessibility cache 必须与 network version、mode、routing scenario/state 绑定，
禁止跨不兼容 scenario 复用。** Reference Pipeline 的持久化 cache 采用两层结构 + 元数据头：

**Supply-layer cache（可达性向量 / mode 旅行时间 / leg 路由）**，键 =
`(network_version, schedule_version, mode, routing_state_version, query)`：

| 键分量 | 内容 |
|---|---|
| `network_version` | `network_with_transit.xml` 的 SHA256（或 release provenance 中网络 id） |
| `schedule_version` | `transitSchedule.xml` + `trips_by_stop.json` + `prep_stops.jsonl` + `stop_snapping_report.json` 的 SHA256 组合 |
| `mode` | `car` / `bike` / `walk` / `pt`（car=free-flow；walk 1.34 m/s、bike 4.17 m/s 为 S9 修正常量） |
| `routing_state_version` | access radius 700 m、egress 700 m+1.5 km、max transfer 1、boarding buffer、connection window 180–2700 s、departure window 3600 s、速度常量（1.34/4.17）+ `gtfs_accessibility.py`/`adapter.py` 的 git commit |
| `query` | `(origin, destination, departure_sec)` |

**元数据头（每个 cache 文件必含，加载时硬校验）**：生成时间、生成命令、
代码 git commit、上述全部版本字段、provenance 引用（`data_fix.json`）、
"smoke/dev" 标记。任何字段不匹配 → 拒绝命中并告警，绝不静默复用。

**Scenario-layer 产物（population.xml / adapter_manifest / MATSim output）**：
永远按情景隔离目录（`outputs/reference_pipeline/smoke/<scenario_id>/`），
不做跨情景共享。context 扰动（雨/延误/票价）不进入 supply-layer cache——可达性只依赖
(OD, 出发时刻, 供给)，但该"可跨情景共享"的前提由上述绑定键保证（S8→S9 速度错误正是
"同 OD 同网络、不同 routing state"产生不同向量的先例）。

---

## 4. Data Leakage 风险清单

| # | 风险 | 判定 | 管控 |
|---|---|---|---|
| R1 | `data/singapore_accessibility/`（S9 修正）与 `data/singapore_accessibility_s8_legacy/`（错误速度）**文件同名并列**，管线指错目录会静默使用错误特征 | 真实存在 | Reference 加载数据集必须显式路径 + 校验 generation_manifest 中的 S9 修正标记；legacy 目录只读、永不引用 |
| R2 | live 与 release 内的 `accessibility_features.yaml` 仍含 S8 时代错误注释（`walk_speed_ms: 1.2`）；照 YAML 重建路由会复现 S8 错误 | 真实存在（已核实代码与 YAML 不一致） | 有效参数以 `gtfs_accessibility.py`（1.34/4.17）+ `provenance/data_fix.json` 为准；Reference 不解析该 YAML 字段作为路由参数，并记录代码 commit |
| R3 | 无绑定 cache 跨网络/情景复用 → 陈旧向量静默注入（S8→S9 同型错误） | 结构性风险 | 实施 §3.3 绑定键 + 元数据头硬校验 |
| R4 | 用 benchmark 数据重新拟合 scaler/category mapping | 被禁止 | Reference 只读加载 `releases/s9_supply_aware_v2/normalization/*.json`；smoke fixture 独立生成，不触碰 S9 三重 holdout split（尤其 test personas / test ODs / ≥15 min 步行负担池） |
| R5 | 调用 DeepSeek Teacher API（`.env` 密钥）或消费 teacher 标注当 smoke 输入 | 被禁止 | Reference 脚本不 import `teacher/`、不读 `.env`；`states_with_teacher.jsonl` 是训练输入，非 smoke 输入 |
| R6 | 以默认路径重建供给，覆盖冻结 `network_with_transit.xml` / `transitSchedule.xml` 等 → 破坏一切绑定其 SHA 的证据 | 真实存在（`osm_network.build_network` / `gtfs_prep` / `build_transit` 默认输出即冻结文件） | Reference 开发期禁止默认路径供给重建；供给实验一律写 dev 目录 |
| R7 | 运行既有脚本的默认输出路径覆盖 legacy 正式目录（如 `run_s8_smoke.py` 默认 `--output outputs/s8_singapore_smoke`；`run_phase_c.py` 默认 `outputs/singapore_phase_c`） | 真实存在 | Reference 永不调用既有脚本的默认输出；一律显式 dev 路径 + `assert_not_frozen_output` |
| R8 | smoke 沿用 `sha256(persona_id) % len(activity_nodes)` 派生 + seed 2026，生成与冻结 Phase C population 相同的人-OD 对，冒烟结果易与正式结果混淆 | 真实存在（确定性派生） | dev 专用 seed + fixture 命名 `reference_smoke_*`，manifest 标记 "dev fixture, not Phase C" |
| R9 | 决策/推理正确性 vs 冻结基准未隔离：Reference 若改动特征或编码顺序，结果与 `evidence/` 冻结数字不可比 | 结构性风险 | 封装完成后先做 §6 的逐位一致性 gate（fixture 上 Reference decide vs 冻结 S9 decide 100% mode 一致），再做只读回归 |

---

## 5. Output Overwrite 风险清单

| # | 风险 | 管控 |
|---|---|---|
| O1 | 写入 `releases/*` | `assert_not_frozen_output()` 硬断言（已有，Reference 必须调用） |
| O2 | 覆盖 `outputs/singapore_phase_c_s9/`（E3/E4 门禁比对锚） | 禁止任何脚本以 `singapore_phase_c` 前缀写输出；Reference 输出根 = `outputs/reference_pipeline/` |
| O3 | 覆盖 `outputs/pipeline_benchmark/od_accessibility_cache.pkl`、`s9.onnx`（审计产物） | 不重跑 `scripts/benchmark_student_pipeline.py` 的 cached/onnx 模式；若确需 benchmark，先复制脚本与输出根到 dev |
| O4 | MATSim 场景目录内 `output/ITERS` 反复写入 | 每次 smoke 用新情景子目录；禁止指向任何既有目录 |
| O5 | 覆盖 git 跟踪的 `reports/`、`evidence/` | Reference 从不写这两处；回归结果只进 `outputs/reference_pipeline/regression/`，晋升需评审（§6） |
| O6 | 覆盖 `data/singapore*/transit/` 构建产物 | 见 R6：禁止默认路径供给重建 |

---

## 6. Gate 顺序（Reference Pipeline 之后的推进约束）

1. **Step 1 — 封装**：仅使用 §2 dev 目录 + §3.3 绑定 cache，跑通 example/smoke；
2. **Step 2 — 一致性 gate**：dev fixture 上 Reference 决策 vs 冻结 S9 `decide()`
   （`releases/s9_supply_aware_v2/checkpoint/model.pt` + 冻结 normalization）
   mode 100% 一致、数值在浮点噪声内；cache-hit 路径 ≡ cold 路径；
3. **Step 3 — 只读回归**：允许对 frozen benchmark（`data/singapore_accessibility/`、
   `data/singapore/**`）做只读 regression：输入只读、输出写入
   `outputs/reference_pipeline/regression/`，与 `evidence/` 冻结数字比对；
4. **Step 4 — 晋升评审**：任何结果要成为正式产物（写入 `reports/`、`evidence/`），
   必须先经人工评审，Reference Pipeline 自身无权自动晋升。

---

## 7. 附：关键冻结证据引用

- 冻结 guard：`src/traveler_distillation/student/release_guard.py`
- S9 修正记录：`releases/s9_supply_aware_v2/provenance/data_fix.json`、
  `docs/S8_DEPRECATION.md`
- 供给出处与 SHA：`data/singapore/gtfs/raw/source_metadata.json`、`checksum.sha256`、
  `releases/s9_supply_aware_v2/provenance/{osm_manifest,gtfs_manifest,singapore_supply}.json`
- 证据索引：`evidence/README.md`（含未入库大文件 SHA 与再生成命令）
- 冻结实验设置：`data/singapore/DATA_README.md`、`reports/PHASE_C_SINGAPORE_REPORT.md`
