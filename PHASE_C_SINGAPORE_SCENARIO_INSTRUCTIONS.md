# Phase C — Singapore Real-Network Scenario Experiments
## 执行指令（草案，供确认）

**前置**：S8 = Supply-Aware Traveler Agent v1.0 = FROZEN（`releases/s8_supply_aware_v1/`，
tag `s8-supply-aware-v1.0`）；Freeze Gate 16/16 通过；`READY_FOR_PHASE_C=True`。
**依据**：`S8_BACKUP_FREEZE_INSTRUCTIONS.md` §18/§21、`NEXT_STEP_PLAN_SINGAPORE_AIT.md` §4。
**阶段角色**：论文主实验 —— 在真实新加坡供给网络上检验扰动情景下 frozen S8 的行为响应与系统后果。

---

# 1. 冻结设置（一律不得改动）

```text
Student           = frozen S8 (releases/s8_supply_aware_v1/checkpoint/model.pt)
N*                = 10,000 agents（所有情景同一 population）
flowCapacityFactor   = 0.3
storageCapacityFactor = 0.3
Singapore supply  = frozen Phase B.5 版本（network_with_transit.xml + 全天 20,966 trips）
PT planner        = frozen validated 版本（直连 + 1 换乘 + 1.5 km 步行延伸）
Student 行为      = 只 load、只推断；禁止训练/改权重/改 schema/改 normalization
Phase C 输出      = outputs/singapore_phase_c/（release 目录禁止写入，guard 硬断言）
```

**population 确定性**：persona/trip 生成 seed=2026（与 Phase A smoke 同源）在所有情景间固定，
保证 C0–C5 是同一批 10,000 个 agent 的 paired 对照（需求侧不冻结生成代码，但冻结本次 seed）。

---

# 2. 情景定义（注入方式：扰动只经 Student context 进入个体决策，网络供给不变）

| 情景 | 参数（其余字段 = baseline） | 与 synthetic 阶段公式的对应 |
|---|---|---|
| **C0 baseline** | clear / intensity 0.0 / road_congestion 0.3 / delay 0 / fare ×1.0 / 无 disruption | `generation_v0_1.yaml` baseline_context |
| **C1 heavy rain** | weather=rain, intensity=**0.75** | 暴露模式 tt ×(1+0.75×0.15×exposure)：walk +11.3%、bike +10.1%、pt +4.5%、car +0.6% |
| **C2 PT fare increase** | fare_multiplier=**1.5**（+50%） | pt 票价 ×1.5（真实网络 pt 费用 = (2.0 + d×0.15)×1.5） |
| **C3 transit delay** | transit_delay_min=**15** | pt tt += 15 min、reliability += 15 min（只改 pt alternative 属性；6 维 accessibility 特征保持供给真值；MATSim 时刻表不改） |
| **C4 road disruption** | road_disruption=**true** | car tt += 20 min、reliability += 20 min（网络不改；demand-side 感知型 disruption） |
| **C5 joint** | heavy rain（0.75）**+** transit delay（15 min） | 两个单轴叠加；理由：雨+延误联合压制 PT/walk/bike，是对 supply-aware PT 决策的最强单组合测试（备选：rain+fare、fare+delay，报告中注明） |

**供给侧不变**：所有情景用同一 `network_with_transit.xml`、同一时刻表、同一 capacity 因子；
C1–C5 的扰动不修改 MATSim 网络/时刻表，只改变 Student 决策输入（与 plan §4“网络供给保持不变”一致）。
MATSim 运行 `lastIteration=0`（不重规划，Student 决策原样执行）。

---

# 3. 运行器与复现

**运行器**：`scripts/singapore/run_phase_c.py`（新建）。

```bash
python scripts/singapore/run_phase_c.py --scenarios C0_baseline,C1_heavy_rain \
    --num-agents 10000 --checkpoint releases/s8_supply_aware_v1/checkpoint/model.pt \
    --output outputs/singapore_phase_c
```

- 复用 `S8MATSimAdapter`（arch `student_s8_v1` 硬断言、12 维特征、load-only）；
- alternative 构建 = S8 真实供给管线（`plan_accessibility` + `build_real_alternatives`）
  + 情景效应（C3/C4 的 pt/car 属性增量，公式与 `AlternativeGenerator._apply_context` 一致）；
- MATSim：`java -Xmx6g -cp <matsim 2026.0> RunMatsimPreloaded config.xml`，`lastIteration=0`；
- 每情景产出 `outputs/singapore_phase_c/<scenario>/`（population/config/java_run.log/result JSON）。

**复现清单**（写入 result JSON）：checkpoint SHA256、supply 文件路径、seed、N、capacity 因子、
MATSim/Java 版本、exit code、runtime。

---

# 4. 指标与 Gate

## 4.1 每情景 Gate（先过 gate 再谈结果）

- MATSim exit = 0；
- **真人 stuck ≤ B.5C 基线水平**：stuckAndAbort 事件同时计入在 30:00 模拟结束时仍未完成的
  transit 车辆与真人。经 C0 实证：B.5C 标定基线（S7-W3，10k，0.3/0.3）同样有 4,347 辆 transit
  车辆（20.7%）在 30:00 被截断 + 266 真人 stuck（2.66%）——transit 截断是冻结设置的既有特性，
  只报告不 gate；gate 只对**真人 stuck ≤ 266（B.5C 基线）**；
- pt alightings ≤ boardings（截断导致少量在车乘客未下车，记录差值并归因）；
- 四模式（car/pt/walk/bike）leg 均可执行（C0 必须全有；C1–C5 允许某模式为 0 但必须记录）；
- failed trips 与 C0 同量级（无结构性断裂）。

## 4.2 情景指标（event-based，复用 `scenario_metrics.collect_metrics`）

- 决策侧：student mode share、executed mode share、departure shift 分布（早/晚/不变）；
- 系统侧：PT boardings、mean trip time、car mean travel time、road delay mean s/passage、
  slow-passage share、car VKT、link delay P50/P90、waiting_for_pt；
- 与 C0 的 paired 差分（同一 population，直接逐情景对比，无 bootstrap——population 固定）。

## 4.3 报告

`outputs/singapore_phase_c/PHASE_C_REPORT.md`：C0 绝对值表 + C1–C5 vs C0 差分表 +
每情景一句话解读 + 诚实边界。关键读数预期方向（写报告时逐条核对，不符也要如实写）：
C1 rain → walk/bike 下降、car/pt 上升；C2 fare ×1.5 → pt share 下降；C3 delay → pt 下降、
waiting 上升；C4 disruption → car 下降；C5 联合 → 效应叠加且不可简单线性分解。

---

# 5. 诚实边界（报告必须包含）

- 扰动为 **demand-side context 注入**；网络供给与时刻表在所有情景中不变（不是网络级故障模拟）。
- MRT 时刻为 frequency-based/synthetic（社区 GTFS feed，非官方 LTA DataMall）；bus 旅行时间含估算。
- 论文表述沿用冻结口径："a controlled real-network experiment under calibrated
  effective capacity"；不称 "calibrated reproduction of real Singapore congestion"、
  不称 "real Singapore traveler behavior"。
- 需求侧为 synthetic personas/trips（非冻结生成代码，但 seed=2026 固定）。
- Phase C 全程不重训 Student；运行中发现的行为边界记录为 known limitation，不作为重训理由。
- **Transit 截断（已知）**：10k + capacity 0.3 设置下约 20% 的 transit 车辆在 30:00 模拟结束时
  仍未完成（B.5C 基线 4,347/20,966 辆，S7-W3 同设置同现象），属冻结设置的既有特性；PT 绝对量
  指标受此截断影响，情景间相对差分（同一设置）仍然有效，报告如实标注。

---

# 6. 执行顺序

```text
C0 baseline → C1 heavy rain → C2 PT fare increase → C3 transit delay
            → C4 road disruption → C5 joint scenario
```

顺序严格；每个情景 gate 失败先暂停排查，不跳过、不伪造。全部完成后更新 PROGRESS.md
并汇报：各情景 gate 状态、与 C0 的关键差分、报告路径。

---

# 7. 一句话执行原则

> 用同一个 frozen S8、同一批 10,000 agents、同一个 frozen supply，把六个情景在真实网络上各跑一遍；
> 只读模型，不读回训练——Phase C 检验行为与系统后果，模型本身不再变。
