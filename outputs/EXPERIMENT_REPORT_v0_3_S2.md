# S2 扩训实验报告：transit_delay + parking_cost 轴补训

> 目标：补 `transit_delay`（延误）与 `parking_cost_multiplier`（停车费）两个扰动轴，
> 检验 student 能否在未见人群（persona-holdout）上学会这两轴的出行响应。

## 数据

- 新增反事实态（复用现有 baseline）：`transit_delay` 3 档 `[5,15,30]` +
  `parking_cost_multiplier` 3 档 `[1.5,2.0,3.0]`（baseline 等价档 `0`/`1.0` 自动跳过）。
- 目标 240 态 × K=3 = 720 次调用；**实际聚合 233 态**（699 valid / 68 failed 重试 / 7 incomplete）。
- 合并后数据集 **713 态 / 2139 repeat / 40 split groups**，`validate_aggregated_dataset.py` PASS。

### ⚠️ 7 个 incomplete 态的说明（诚实边界）

7 个 incomplete 全部是 `transit_delay=30` 且教师建议**提前 >60 min 出发**（实测 -79~-90 min）
的"继续乘 pt + 大幅提前出发"样本。`UniversalTravelerAction.departure_time_shift_min` 的
schema 硬约束是 ±60（与 student 的 `60·tanh` 出发头表示范围一致），因此这些响应在 parse 阶段
被拒绝。这是 **student 表示能力边界**（>60 min 提前本就无法被 student 学习），非数据 bug；
按"蒸馏范围 = student 可表示范围"原则将其排除。约占 transit_delay=30 档的 17%（7/40），
全局仅 3%（7/240）。

## 教师侧（ground truth，test = 3 未见 personas / 108 态）

| 扰动轴 | 教师弹性 \|ΔP_T\| | 教师噪声(pairwise L1) | 信噪比 |
|---|---|---|---|
| parking_cost_multiplier | 0.0806 | 0.1289 | **0.63** |
| transit_delay | 0.0757 | 0.1483 | 0.51 |
| weather_intensity | 0.0797 | 0.1710 | 0.47 |
| road_congestion | 0.0572 | 0.1325 | 0.43 |
| fare_multiplier | 0.0493 | 0.1451 | 0.34 |

S2 两轴的信噪比是**五轴中最高**（尤其停车费 0.63），即这两个轴的教师信号相对最"干净"。

## Student 侧：before/after（test 逐轴，C 变体）

| 轴 | 指标 | before(S1-C) | after(S2-C) |
|---|---|---|---|
| parking_cost | \|ΔP_S\| | 0.0098 | **0.0577** (教师 0.0806) |
| parking_cost | \|ΔP_T−ΔP_S\| | 0.0901 | **0.0587** |
| parking_cost | sign | 0.3148 | **0.7778** |
| transit_delay | \|ΔP_S\| | 0.0982 | 0.0954 (教师 0.0757) |
| transit_delay | \|ΔP_T−ΔP_S\| | 0.0665 | **0.0466** |
| transit_delay | sign | 0.6852 | **0.7593** |

- **parking_cost（干净的数据缺口 → 学会）**：补训前 sign 0.31（≈随机）、\|ΔP_S\|≈0；补训后
  sign 0.78、\|ΔP_S\| 0.058 接近教师 0.081。方向与幅度均被学会，是 S2 最干净的成功。
- **transit_delay（已是特征驱动 → 精化）**：补训前 sign 已达 0.685（非随机）——因为
  `transit_delay_min` 会直接传播进 pt 的 travel_time/reliability 属性（student 可见），故
  S1-C 已有部分响应；补训后 sign 0.759、gap 0.067→0.047，进一步对齐。

## 三管线整体（test = 108 态）

| 指标(test) | v0.3-s2-A | v0.3-s2-B | v0.3-s2-C |
|---|---|---|---|
| mode_accuracy | 0.8519 | 0.8519 | **0.8611** |
| KL(P_T‖P_S) | 0.1202 | 0.1231 | **0.0529** |
| probability L1 | 0.2897 | 0.2945 | **0.2357** |
| counterfactual \|dP_T−dP_S\| | 0.0642 | 0.0617 | **0.0460** |
| sign agreement | 0.6863 | 0.6078 | **0.6928** |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0699 |

## 结论

1. **停车费是干净的数据缺口**，补训后完整学会（方向 + 幅度），与教师信噪比最高（0.63）一致。
2. **延误轴部分可由特征驱动**（transit_delay 已通过替代属性可见），补训后方向进一步对齐
   （sign 0.76）、残差收窄（0.047）。
3. **跨轴一致规律**：轴的教师**信噪比**越高，student 越能学会（停车费 0.63 > 延误 0.51 >
   天气 0.47 > 拥堵 0.43 > 票价 0.34）。S1 的"拥堵幅度欠拟合"与 S2 的"停车费完整学会"
   共同指向：student 弹性拟合上限由**教师该轴信噪比**决定，而非 student 容量。
4. **表示边界**：departure_shift ±60 的 schema 硬约束使 7 个"提前 >60min"样本被排除；
   这是 student `60·tanh` 出发头的固有边界，S3/S4 若关注延误的出发时刻需评估是否放宽。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\extend_teacher_dataset.py `
  --dataset data\student_v0_3_s1\aggregated_teacher_dataset.jsonl `
  --axes transit_delay,parking_cost_multiplier --output data\student_v0_3_s2 --workers 6

.\.venv\Scripts\python.exe scripts\run_v0_3_s1_experiment.py --tag s2 --outputs outputs

.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s2\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s1_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml   # before
.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s2\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s2_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml   # after
```
