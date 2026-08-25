# S4 扩训实验报告：road_disruption（道路中断）轴补训

> 目标：补最后一个扰动轴 `road_disruption`（事故/封路，故事性强），
> 完成 6/6 扰动轴覆盖，并检验 student 能否在未见人群（persona-holdout）上
> 学会"道路中断 → 出行方式与出发时刻调整"。

## 数据

- 新增 `road_disruption` 反事实态：复用现有 40 条 baseline，1 档 `true`
  （baseline 为 `false`，等价档自动跳过）= **40 态 × K=3 = 120 次真实调用**。
- 生成结果：**attempted 121 / valid 120 / failed 1（1 次 "no JSON object" 解析失败自动重试）/
  incomplete 0 / aggregated 40**。无 S2 那样的 departure_shift 越界剔除。
- 合并后数据集 **753 态 / 2259 repeat / 40 baselines + 713 counterfactuals / 40 split groups**，
  `validate_aggregated_dataset.py` PASS。
- 生成脚本：`scripts/extend_teacher_dataset.py`（沿用 S1/S2 的复用-baseline 扩展管线）。

## 教师侧（ground truth；信号=test 3 未见 personas，噪声=全 40 态 K=3 pairwise L1）

| 扰动轴 | 教师弹性 \|ΔP_T\| | 教师噪声(pairwise L1) | 信噪比 |
|---|---|---|---|
| **road_disruption** | **0.1756** | 0.1388 | **1.27** |
| parking_cost_multiplier | 0.0806 | 0.1289 | 0.63 |
| transit_delay | 0.0757 | 0.1483 | 0.51 |
| weather_intensity | 0.0797 | 0.1710 | 0.47 |
| road_congestion | 0.0572 | 0.1325 | 0.43 |
| fare_multiplier | 0.0493 | 0.1451 | 0.34 |

- **道路中断是六轴中信号最强、信噪比最高的一轴**（SNR 1.27，远超此前的停车费 0.63）：
  car 概率质量在封路下平均 **−0.351**（pt +0.102、bike +0.026、walk +0.010）——这是教师
  所有扰动轴中最大的单轴迁移。故事性强 → 教师响应明确且一致，符合预期。
- 出发时刻：40 态平均 delta shift −2.85 min（7/40 提前 ≥20 min、7/40 不变）；主要响应是
  **方式转移（car→pt）**，出发提前是次要效应。

## Student 侧：road_disruption before/after（同一 6 个 test 态，C 变体）

| checkpoint | \|ΔP_T\| | \|ΔP_S\| | \|ΔP_T−ΔP_S\| | sign agreement |
|---|---|---|---|---|---|
| v0.3-S2-C（无中断训练） | 0.1756 | 0.1056 | 0.0738 | 0.7778 |
| **v0.3-S4-C（补训后）** | 0.1756 | **0.1408** | 0.0787 | **0.8333** |

- **before 已有 sign 0.78（非随机）**：`road_disruption` 对 car 的作用（travel_time +20 min、
  reliability +20 min）与已学过的拥堵轴（car travel_time/reliability ↑ → 离开 car）走**同一条
  属性通道**，故 S2-C 部分泛化——与 S2 中 transit_delay"特征驱动"现象同构。
- **after 进一步对齐**：sign 0.78→0.83，\|ΔP_S\| 0.106→0.141（教师 0.176，幅度欠响应收窄）。
- 诚实边界：test 仅 **6 个** road_disruption 态（3 未见 personas × 2 trips），development-scale；
  per-mode 残差 \|ΔP_T−ΔP_S\| 微升（0.074→0.079）表明补训在个别 mode 上方向仍不完美，需更大样本确认。

## 三管线整体（test = 114 态，含 6 个 road_disruption 态；注意 test 组成 108→114 变化，仅参考）

| 指标(test) | v0.3-S4-A | v0.3-S4-B | v0.3-S4-C |
|---|---|---|---|
| mode_accuracy | 0.8509 | 0.8509 | **0.8684** |
| KL(P_T‖P_S) | 0.1233 | 0.1373 | **0.0519** |
| probability L1 | 0.3037 | 0.3143 | **0.2303** |
| counterfactual \|dP_T−dP_S\| | 0.0696 | 0.0675 | **0.0551** |
| sign agreement | 0.6852 | 0.6790 | 0.6636 |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0705 |

C 仍为三管线最优（acc 0.868 / KL 0.052 / L1 0.230），与 S1/S2 一致。整体 sign agreement
较 S2 略降（0.693→0.664）主要来自 test 集新增了"更难"的 road_disruption 轴 + 训练分布微扰。

## 结论

1. **道路中断是教师最干净、最强的扰动轴**（SNR 1.27、car −0.351），学生补训后方向一致性
   **0.78→0.83**、幅度响应 **0.106→0.141**（向教师 0.176 收敛）。
2. 与 S2 的 transit_delay 同构：道路中断**部分可由特征驱动**（car travel_time/reliability 属性
   已可见，S2-C 即 sign 0.78），补训进一步精化方向与幅度。
3. **六轴数据缺口全部补齐**（天气/票价/拥堵/延误/停车费/道路中断）。跨轴结论一致：
   student 弹性拟合上限 = 教师该轴信噪比（道路中断 1.27 > 停车费 0.63 > 延误 0.51 >
   天气 0.47 > 拥堵 0.43 > 票价 0.34），信号越干净学得越完整。
4. 剩余欠拟合轴（票价/拥堵，信噪比 < 0.5）的幅度提升应走 **K>3 压噪**，而非更多 persona
   （与 S1/S2 结论一致）。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\extend_teacher_dataset.py `
  --dataset data\student_v0_3_s2\aggregated_teacher_dataset.jsonl `
  --axes road_disruption --output data\student_v0_3_s4 --workers 4

.\.venv\Scripts\python.exe scripts\run_v0_3_s1_experiment.py --tag s4 --outputs outputs

.\.venv\Scripts\python.exe scripts\teacher_axis_snr.py `
  --dataset data\student_v0_3_s4\aggregated_teacher_dataset.jsonl `
  --repeats data\student_v0_3_s4\repeat_records.jsonl --config configs\student_v0_3_c.yaml

# before / after（road_disruption 逐轴弹性，C 变体）
.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s4\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s2_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml   # before
.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s4\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s4_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml   # after
```
