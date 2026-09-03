# DeepSeek Traveler Behavior Distillation Research

> 将 DeepSeek V4 Pro 对「不同居民（Persona）× 动态城市环境（Context）→ 出行行为响应」
> 的推理能力，蒸馏为可在 **MATSim** 中大规模运行的轻量 Traveler Agent，并在
> **新加坡真实路网 + 公交供给**上做情景实验。

核心链路：

```text
DeepSeek V4 Pro teaches → Lightweight Traveler Agents execute → MATSim simulates (real Singapore supply)
```

## 研究动机

LLM 具备对异质出行者行为响应的常识性推理能力，但单次调用成本高、无法支撑
百万级人口仿真。本项目把教师模型（DeepSeek V4 Pro）在
`persona × context → action distribution` 上的行为策略蒸馏到一个小模型
（student，~2.5 万参数），使其：

1. 在 **可变选择集** 上输出与教师一致的行为偏好分布；
2. 保留教师对扰动轴（天气、票价、延误、拥堵、停车费）的 **弹性 / 方向 / 幅度** 响应；
3. 对 **未见人群 / 未见 OD / 未见可达性**（三重 holdout）具备泛化能力；
4. 能写入 MATSim 场景并在真实路网级仿真中运行。

## 冻结模型（Releases）

| Release | 角色 | 状态 |
|---|---|---|
| `s7-w3-generic-core-v1.0` | S7-W3 **Generic Behavioral Core v1.0**（24,370 参数，单轴/多轴/机制蒸馏线收口） | ✅ FROZEN — 论文 generic baseline |
| `s8-supply-aware-v1.0` | S8 Supply-Aware Traveler Agent v1.0（24,562 参数，Case B +6 可达性特征） | ⚠️ **DEPRECATED** — walk/bike 速度数据错误（见 `docs/S8_DEPRECATION.md`），release 保持字节不变 |
| `s9-supply-aware-v2.0` | **S9 Supply-Aware Traveler Agent v2.0**（24,562 参数，修正后重训） | ✅ FROZEN — 论文 supply-aware extension |

每个 release 均为完整冻结包：checkpoint（SHA256）、config、schema（Case B diff）、
normalization、可达性特征定义、供给/教师 provenance、数据集 manifest、最终指标、
复现 gate、只读保护（`src/traveler_distillation/student/release_guard.py` 硬断言）。

## Phase C 主实验结果（Singapore 真实网络，frozen S9）

**设置**：N\*=10,000 agents（seed 2026，六情景同一 population）· Tampines + Pasir Ris
真实 OSM 路网 + 全天 20,966 班次公交供给 · flow/storage capacity 因子 0.3（B.5C 标定）·
扰动仅经 Student context 注入，供给不变 · `lastIteration=0`（不重规划）。

| 情景 | car | pt | bike | walk | PT 登车 | car VKT | gate |
|---|---|---|---|---|---|---|---|
| **C0 baseline** | 28.3% | **25.3%** | 34.0% | 12.4% | 5,613 | 33,939 km | ✅ |
| **C1 heavy rain** (0.75) | **37.5%** | **45.3%** | 13.3% | 3.8% | **9,761** | 42,778 km | ✅ |
| **C2 PT fare ×1.5** | 29.9% | 25.7% | 32.5% | 11.8% | 5,674 | 35,798 km | ✅ |
| **C3 transit delay 15min** | 29.0% | **10.2%** | 35.8% | **25.0%** | 2,474 | 34,685 km | ✅ |
| **C4 road disruption** | **1.8%** | **37.5%** | **41.8%** | 19.0% | 8,537 | **2,943 km** | ✅ |
| **C5 rain + delay** | **37.6%** | 18.6% | 24.5% | 19.2% | 4,301 | 42,823 km | ✅ |

关键读数（paired，同一 10k population）：

- **heavy rain** 把步行/骑行压入 car/pt（pt 25.3%→45.3%，PT 登车 +74%）；
- **PT 延误 15 min** 使 pt 25.3%→10.2%（健康响应，非病态清零）；
- **道路中断** 使 car 28.3%→1.8%（VKT −91%），pt/bike 吸收转移需求；
- **联合情景（雨+延误）** 的 pt（18.6%）介于两个单轴之间——效应可解释、非简单叠加；
- 票价弹性弱（C2 +0.4pp）如实报告。

完整报告与诚实边界：`reports/PHASE_C_SINGAPORE_REPORT.md`；模型侧证据：
`reports/EXPERIMENT_REPORT_S9_TRANSIT_ACCESSIBILITY_V2.md`（PT-MAE −0.040*、
**FVR 0.333→0.083**、回归门禁全过、Stop Rule 六项满足）。

## 目录结构

```text
src/traveler_distillation/    # 核心包：schemas / generators / teacher / dataset / student /
                              # accessibility（真实供给可达性）/ singapore（OSM/GTFS→MATSim）/ matsim
scripts/                      # 可执行管线（生成 / 标注 / 训练 / 评估 / MATSim / 冻结 / Phase C）
reference_pipeline/           # Reference MATSim Integration Pipeline 包（配置/校验/特征/Student/
                              # 路由缓存/MATSim plan 构建；见 docs/REFERENCE_PIPELINE.md）
run_pipeline.py               # Reference 一键入口（等价 scripts/run_reference_pipeline.py）
examples/                     # Reference 示例输入（sample_population.csv）
configs/                      # YAML 配置（generation / teacher / student v0.x / S5-S9 / reference_example）
releases/                     # 冻结模型发布包（S7-W3 / S8-deprecated / S9，含 SHA256 与复现 gate）
reports/                      # 实验报告 / 审计 / Phase C 报告（git 追踪）
evidence/                     # 论文证据快照：E1–E5 报告与结果 JSON、Phase C 原始结果、S9 评估快照
                              #（索引与大文件清单见 evidence/README.md）
docs/plans/                   # 研究蓝图 / Singapore 验证计划 / 任务阶段清单
docs/stage_instructions/      # 各阶段执行指令与实验设计（S5-S8、Phase C、冻结指令）
docs/                         # 供给集成说明 / S8_DEPRECATION.md / Reference 文档 / 数据隔离检查
tests/                        # pytest（172 passed，含 Reference 单测）
data/                         # 生成的数据集（JSONL，git 忽略，可复现生成）
outputs/                      # 训练与运行产物（git 忽略；报告均复制至 reports/ 追踪）
archive/legacy_*/             # 已归档的旧版文件
```

> 2026-08-27 根目录整理：阶段指令/计划文档从仓库根目录移入 `docs/`（下表为旧路径映射，
> 冻结 release 文档内引用的仍是冻结时的旧路径）。代码/数据/输出的默认路径未变。

| 旧路径（根目录） | 新路径 |
|---|---|
| `S5_MULTI_AXIS_DISTILLATION_EXPERIMENT_DESIGN.md` | `docs/stage_instructions/` |
| `S6_REASONING_CAUSAL_AUDIT_EXPERIMENT_DESIGN.md` | `docs/stage_instructions/` |
| `S7_MECHANISM_AWARE_FINETUNING_INSTRUCTIONS.md` | `docs/stage_instructions/` |
| `S7_W3_BACKUP_FREEZE_INSTRUCTIONS.md` | `docs/stage_instructions/` |
| `S8_TRANSIT_ACCESSIBILITY_TRAINING_INSTRUCTIONS.md` | `docs/stage_instructions/` |
| `S8_BACKUP_FREEZE_INSTRUCTIONS.md` | `docs/stage_instructions/` |
| `PHASE_C_SINGAPORE_SCENARIO_INSTRUCTIONS.md` | `docs/stage_instructions/` |
| `DeepSeek_Traveler_Behavior_Distillation_Research_Blueprint.md` / `NEXT_STEP_PLAN_SINGAPORE_AIT.md` / `Task_Phase.txt` | `docs/plans/` |

## 环境要求

- Python ≥ 3.12、PyTorch ≥ 2.5（CPU 可跑全部训练/评估）
- Java 25 + MATSim 2026.0（Singapore 场景仿真；发行包从 matsim-org/matsim-libs
  官方 GitHub Releases 下载，解压为 `tools/matsim-2026.0-release/`）
- DeepSeek API key（官方直连，教师标注用）
- 外部数据源（OSM/GTFS）：来源、下载日期与 SHA256 见
  `data/singapore/gtfs/raw/source_metadata.json`、`releases/s9_supply_aware_v2/provenance/`
  与 `evidence/e5_helsinki/supply/`（Singapore 研究区 OSM 提取已随仓库提供）

## 快速开始

```powershell
# 1. 依赖
.venv\Scripts\python.exe -m pip install -e .[dev]

# 2. 配置密钥：复制 .env.example 为 .env 并填写 DEEPSEEK_API_KEY
#    （仅教师标注需要；Reference Pipeline 与 MATSim 部署不调用 Teacher，无需密钥）

# 3. 跑测试
.venv\Scripts\python.exe -m pytest          # 172 passed

# 4. 加载冻结 S9 并做一次推断
.venv\Scripts\python.exe scripts\singapore\run_s8_smoke.py `
  --checkpoint releases\s9_supply_aware_v2\checkpoint\model.pt

# 5. Reference MATSim Integration Pipeline：CSV → Student 决策 → population.xml（可选 --run-matsim）
.venv\Scripts\python.exe run_pipeline.py --config configs\reference_example.yaml
```

## 关键设计决策

- **Universal State/Action Schema**：simulator-independent（`schemas/`），可变选择集。
- **可变选择集 Student**：masked softmax 只在 available alternatives 上归一化。
- **三重 holdout**：persona（28/6/6）+ OD 不相交 + 可达性（高步行负担仅测试集），
  test-only 评估，配对 bootstrap CI（B=2000）。
- **Case B 架构演化**：S8/S9 在冻结 S7-W3 之上新增 6 维 city-independent
  可达性特征（alt_encoder 14→20），共享权重逐字节复制 + 新列零初始化
  （S9-at-init ≡ S7-W3 输出）。
- **City-independence**：模型输入只有数值可达性向量，无任何地点身份
  （引号级泄漏检查 0 命中）。
- **冻结纪律**：release 只读 + `assert_not_frozen_output` 硬断言；
  重训仅因明确数据错误（S8→S9 一例，审计链见 `docs/S8_DEPRECATION.md`）。

## 路线图（详见 `docs/plans/Task_Phase.txt` / `PROGRESS.md`）

| 阶段 | 状态 | 内容 |
|---|---|---|
| Phase 0–7（S1–S7） | ✅ | Teacher 审计（K=3/5）→ 蒸馏 v0.x → S5 多轴 → S6 因果审计 → S7 机制补训（seed 稳定）→ Freeze S7-W3 |
| S8 | ⚠️ 废弃 | 真实供给可达性适配（Case B）——因 walk/bike 速度数据错误废弃 |
| S9 | ✅ | 修正后重训 + Freeze（`s9-supply-aware-v2.0`） |
| Singapore Phase A / B / B.5 | ✅ | 真实供给跑通门禁 / 规模验证 / PT 有效性 + 容量标定（N\*=10k, 0.3/0.3） |
| **Phase C** | ✅ | **六情景主实验完成（本 README 表 + `reports/PHASE_C_SINGAPORE_REPORT.md`）** |
| **补充实验 E1–E5（TRC_AIT_5）** | ✅ | MNL-B 基线 · DeepSeek vs S9 速率/成本 · 人口扩展 1k–50k · multi-seed 稳健性 · Helsinki zero-shot 迁移（六情景 C0–C5，门禁全过）；Stop Rule 达成 → manuscript v1 |
| Phase D | 🚧 | 真实网络反馈闭环（Student → MATSim → 拥堵观测 → 再决策收敛） |

## Reference MATSim Integration Pipeline（可复用部署入口）

把上面的 Student→MATSim 部署流程封装成一条命令（面向陌生用户的 reference 实现，
非 universal adapter）：

```powershell
# 输入自检（不加载模型）
.venv\Scripts\python.exe scripts\run_reference_pipeline.py `
  --config configs\reference_example.yaml --validate-only

# CSV → Student 决策（batch）→ routing（磁盘缓存）→ population.xml + manifest + summary
.venv\Scripts\python.exe scripts\run_reference_pipeline.py `
  --config configs\reference_example.yaml

# 再直接跑 MATSim
.venv\Scripts\python.exe scripts\run_reference_pipeline.py `
  --config configs\reference_example.yaml --run-matsim
```

- 输入规范 / 配置 / 缓存行为 / 排障：`docs/REFERENCE_PIPELINE.md`
- 与原 pipeline 的正确性对齐（决策 100% 一致、population.xml 字节一致、10k 只读回归、
  Helsinki 第二城跨城市验证）：
  `docs/REFERENCE_PIPELINE_VALIDATION.md`
- 实测时间缩短（两城同一缓存机制、口径统一）：Singapore 10k warm **6.5×** vs 原 pipeline
  （3,063→470 s；cold→warm 6.9×）；Helsinki 10k warm **92.1×** vs 原 pipeline
  （9,507→103 s；cold→warm 101.7×）；决策 10,000/10,000 保留
- 数据隔离约定：`docs/DATA_ISOLATION_CHECK.md`

## 复现说明

`data/`、`outputs/` 已被 git 忽略；数据集与运行产物均可由 `scripts/` 从 `configs/`
复现生成（大文件：MATSim 发行包、GTFS 数据不入仓库）。**论文全部数字的证据快照**
在 `evidence/`（E1–E5 报告与结果 JSON、Phase C 原始结果、S9 评估快照，共约 21 MB）；
超过 20 MB 的原始工件不随仓库分发，其大小、SHA256 与再生成命令列于
`evidence/README.md` §3。冻结模型的复现 gate：
`python scripts/freeze_s9_release.py verify-gate`（12/12 指标 Δ=0.0000）。

## 许可

Private research repository. All rights reserved.
