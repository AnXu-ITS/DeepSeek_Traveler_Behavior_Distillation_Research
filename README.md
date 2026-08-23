# DeepSeek Traveler Behavior Distillation Research

> 将 DeepSeek V4 Pro 对「不同居民（Persona）× 动态城市环境（Context）→ 出行行为响应」
> 的推理能力，蒸馏为可在 **MATSim** 中大规模运行的轻量 Traveler Agent。

核心链路：

```text
DeepSeek V4 Pro teaches → Lightweight Traveler Agents execute → MATSim simulates
```

## 研究动机

LLM 具备对异质出行者行为响应的常识性推理能力，但单次调用成本高、无法支撑
百万级人口仿真。本项目把教师模型（DeepSeek V4 Pro）在
`persona × context → action distribution` 上的行为策略蒸馏到一个小模型
（student），使其：

1. 在 **可变选择集** 上输出与教师一致的行为偏好分布；
2. 保留教师对扰动轴（天气、票价等）的 **弹性 / 方向 / 幅度** 响应；
3. 对 **未见人群（persona-holdout）** 具备泛化能力；
4. 能写入 MATSim 场景并在真实路网级仿真中运行，含网络反馈闭环。

## 目录结构

```text
src/traveler_distillation/    # 核心包：schemas / generators / teacher / dataset / student / audit / matsim
scripts/                      # 可执行管线脚本（生成 / 训练 / 评估 / MATSim 运行）
configs/                      # YAML 配置（generation / teacher / student v0.2-A/B/C、v0.3-A/B/C）
tests/                        # pytest（116 passed）
data/                         # 生成的数据集（JSONL，git 忽略，可复现生成）
outputs/                      # 训练与审计产物（报告 / checkpoint / 指标，git 忽略）
tools/MATSim/                 # MATSim 2026.0 发布包（Phase 8+ 使用，大文件忽略）
archive/legacy_*/             # 已归档的旧版文件（见 archive/README.md）
```

## 环境要求

- Python ≥ 3.11
- Java 25 + MATSim 2026.0（Phase 8 起的场景仿真）
- DeepSeek API key（官方直连）

## 快速开始

```powershell
# 1. 依赖
.venv\Scripts\python.exe -m pip install -e .[dev]

# 2. 配置密钥：复制 .env.example 为 .env 并填写 DEEPSEEK_API_KEY

# 3. 跑测试
.venv\Scripts\python.exe -m pytest

# 4. 教师 API smoke test（真实 DeepSeek）
.venv\Scripts\python.exe scripts\smoke_test_teacher.py
```

## 数据集生成（K=3 聚合教师目标）

Teacher 可复现审计结论（见 `outputs/teacher_audit_v0_1/`）：same-state 概率向量存在
真实采样方差（mean pairwise L1=0.19），故每个 state 做 **3 次 cache-busted 调用取均值
分布**作为蒸馏目标（K3↔K5 L1 仅 0.047）。

```powershell
# dry-run 看规模与调用预算
.venv\Scripts\python.exe scripts\generate_aggregated_teacher_dataset.py `
  --dry-run --num-personas 8 --num-trips 3 --axes weather_intensity,fare_multiplier

# 真实生成（4 并发，可 --resume 断点续跑）
.venv\Scripts\python.exe scripts\generate_aggregated_teacher_dataset.py `
  --num-personas 8 --num-trips 3 --axes weather_intensity,fare_multiplier `
  --workers 4 --output data/student_v0_2_a
```

规模：8 personas × 3 trips × {weather_intensity, fare_multiplier} = 192 states × 3 = 576 次调用。

## Student 训练

```powershell
# v0.2-A baseline（L_action + L_distribution）
.venv\Scripts\python.exe scripts\train_student_v0_2_a.py `
  --config configs/student_v0_2_a.yaml `
  --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl `
  --output outputs/student_v0_2_a

# v0.2-B（+ L_elasticity，行为弹性保留）
.venv\Scripts\python.exe scripts\train_student_v0_2_b.py `
  --config configs/student_v0_2_b.yaml `
  --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl `
  --output outputs/student_v0_2_b

# 对比两个 run
.venv\Scripts\python.exe scripts/compare_students.py `
  --baseline outputs/student_v0_2_a --candidate outputs/student_v0_2_b `
  --output outputs/comparison_v0_2_a_vs_b.md
```

### v0.3：persona-holdout + 分解弹性损失

```powershell
# 生成 v0.3 数据集（20 personas x 2 trips x {weather, fare} = 320 states）
.venv\Scripts\python.exe scripts\generate_aggregated_teacher_dataset.py `
  --num-personas 20 --num-trips 2 --axes weather_intensity,fare_multiplier `
  --workers 4 --output data/student_v0_3

# 一键跑完整 v0.3 实验（校验数据 + 训练 A/B/C + 对比 + 汇总表）
.venv\Scripts\python.exe scripts\run_v0_3_experiment.py `
  --dataset data/student_v0_3/aggregated_teacher_dataset.jsonl `
  --repeats data/student_v0_3/repeat_records.jsonl `
  --outputs outputs
```

v0.3 关键变化：

- **persona-holdout 切分**：`group_field=persona_group_id`，test personas 在训练中
  完全不可见（未见人群泛化检验）；per-persona test breakdown + counterfactual
  sign agreement 作为新评估指标。
- **分解弹性损失**：`lambda_elasticity`(L1) / `lambda_direction`(方向对齐，只罚
  与 teacher 反向的移动) / `lambda_magnitude`(幅度分解) 由 config 控制。

## MATSim 集成与人口级仿真

```powershell
# 构建 MATSim 场景并把 student 决策写入
.venv\Scripts\python.exe scripts\build_matsim_scenario.py

# 运行 MATSim（Java 25 + MATSim 2026.0）
.\scripts\run_matsim.ps1

# Phase 9：1000 personas × 4 情景的人口级实验
.venv\Scripts\python.exe scripts\run_population_experiment.py

# Phase 10：网络反馈闭环
.venv\Scripts\python.exe scripts\run_phase10_loop.py
```

## 关键设计决策

- **Universal State/Action Schema**：simulator-independent（`schemas/`），可变选择集。
- **可变选择集 Student**：masked softmax 只在 available alternatives 上归一化。
- **切分按 `split_group_id`（persona::trip）**：baseline 与其全部反事实曲线同组，
  保证弹性评估不跨 split 泄漏。
- **蒸馏目标**：mean behavioral preference distribution（非多数投票）。
- **评估不以 accuracy 为唯一指标**：重点看 probability L1 / KL 与 teacher 噪声下限
  0.047 的对照、counterfactual ΔP 保留度（|ΔP_T − ΔP_S|）。

## 路线图（详见 `Task_Phase.txt` / `PROGRESS.md`）

| 阶段 | 状态 | 内容 |
|---|---|---|
| Phase 0–7 | ✅ | Teacher 审计（K=3 聚合）→ 蒸馏（v0.2-A/B/C、v0.3-A/B/C persona-holdout + 分解弹性 + 异质性）→ 个体级验证 |
| Phase 8 | ✅ | `MATSimAdapter`（`src/traveler_distillation/matsim/`）—— student 决策写入 MATSim 场景并真实运行成功 |
| Phase 9 | ✅ | 1000 personas × 4 动态情景的人口级仿真，需求转移方向全部正确 |
| Phase 10 | ✅ | 网络反馈闭环，3 情景收敛到均衡拥堵 |
| 下一步 | 🚧 | 补训扰动轴（拥堵/延误/停车费）、OSM/GTFS 真实路网（新加坡）、S5 多轴蒸馏 / S6 推理因果审计实验 |

## 复现说明

`data/`、`outputs/` 已被 git 忽略；所有数据集与实验产物均可由上文脚本从
`configs/` 配置复现生成。大文件（MATSim 发行包、GTFS 数据）不入仓库。

## 许可

Private research repository. All rights reserved.
