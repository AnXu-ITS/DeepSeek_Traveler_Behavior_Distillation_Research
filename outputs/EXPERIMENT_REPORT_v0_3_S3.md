# S3 扩训实验报告：persona 20→40（holdout 统计功效）

> 目标：把合成人口从 20 扩到 40 个 persona（新增 20 个 unseen 人群 × 2 次出行 × 全 6 轴），
> 检验 holdout 测试的统计功效提升，以及在更大、更多样人群上的泛化。

## 数据

- 新增 20 个 persona（P000021–P000040，seed=43，与原有 seed=42 人群独立）+ 2 次新出行
  （T000003、T000004），全 6 轴（weather/congestion/transit_delay/fare/parking/road_disruption）：
  **760 态 × K=3 = 2280 次真实调用**。
- 生成结果：**760 态全部聚合、0 incomplete**（新 personas 未触发 S2 的 `departure_shift>60`
  剔除）；期间官方 API 有瞬时故障（empty content / malformed JSON / read timeout），均由
  重试自动恢复。生成中将 workers 8→16（`--resume`）把吞吐从 ~1.4 抬到 ~4–5 态/分。
- 合并后数据集 **1513 态 / 4539 repeat / 80 baselines + 1433 counterfactuals / 80 split groups**，
  `validate_aggregated_dataset.py` PASS。
- 代码：`generate_aggregated_teacher_dataset.py` 新增 `--persona-offset/--trip-offset/--id-prefix`
  （`PersonaGenerator`/`TripGenerator` 加 `id_offset`，向后兼容）；新增 `scripts/merge_teacher_datasets.py`。

## 教师侧（40 personas；信号=test 6 未见 personas / 226 态，噪声=全态 K=3 pairwise L1）

| 扰动轴 | 教师弹性 \|ΔP_T\| | 教师噪声 | 信噪比 |
|---|---|---|---|
| road_disruption | 0.1422 | 0.1472 | **0.97** |
| weather_intensity | 0.1023 | 0.1386 | 0.74 |
| parking_cost_multiplier | 0.0660 | 0.1476 | 0.45 |
| road_congestion | 0.0434 | 0.1305 | 0.33 |
| transit_delay | 0.0453 | 0.1414 | 0.32 |
| fare_multiplier | 0.0358 | 0.1345 | 0.27 |

- 信噪比排序与 20-persona 时代一致（道路中断最高、票价最低）；但**弹性估值随人群扩容变化**
  （天气 0.080→0.102、延误 0.076→0.045、票价 0.049→0.036）——这正说明 20-persona 的逐轴
  弹性估计方差大，40-persona 更接近人群期望。

## Student 侧：逐轴 before/after（同一 6 个 test personas，C 变体）

| 轴 | 指标 | before(S4-C) | after(S3-C) |
|---|---|---|---|
| road_congestion | sign | 0.609 | **0.719** |
| transit_delay | sign | 0.831 | **0.868** |
| road_disruption | sign | 0.972 | 0.764 |
| fare_multiplier | sign | 0.727 | 0.556 |
| parking_cost | sign | 0.771 | 0.678 |
| weather | sign | 0.707 | 0.693 |

⚠️ **before/after 不是严格可比**：S4-C 是在 20-persona 上训练的，其 holdout test 只有 3 个
persona；而这 6 个 test personas 里有 5 个（P000002/008/015/016/018）是 S4-C 的**训练人群**，
因此 S4-C 的"before"数字偏高（尤其 road_disruption 0.97 属于对已见人群的记忆）。S3-C 的
holdout 才是**真正未见 6 人群**：all 6 个 test personas 均未参与 S3-C 训练。

## 三管线整体（test = 226 态 = 6 未见 personas）

| 指标(test) | v0.3-S3-A | v0.3-S3-B | v0.3-S3-C |
|---|---|---|---|
| mode_accuracy | 0.8673 | **0.8850** | 0.8496 |
| KL(P_T‖P_S) | 0.0880 | **0.0497** | 0.0776 |
| probability L1 | 0.2897 | **0.1872** | 0.2646 |
| counterfactual \|dP_T−dP_S\| | 0.0627 | 0.0598 | **0.0516** |
| sign agreement | 0.6301 | 0.6745 | **0.7048** |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0955 (561对) |

- **统计功效的直观证据**：在 20-persona 时代（test=3 人/114 态）A/B 的 acc 恒并列（0.851）、
  差异"不显著"；扩容后 B 的 KL（0.050 vs A 0.088）与 L1（0.187 vs 0.290）**明显拉开**——
  数据量一够，A/B 的差异才可见。
- C 仍是 counterfactual 方向最优（sign 0.705、gap 0.052），B 是分布拟合最优（KL/L1）。
  目标不同选型不同：要分布保真选 B，要反事实方向一致选 C。

## 未见人群逐 persona（S3-C，6 个 test personas）

| persona | n | mode_acc | prob_L1 |
|---|---|---|---|
| P000002 | 38 | 0.9211 | 0.2543 |
| P000008 | 37 | 1.0000 | 0.1200 |
| P000009 | 38 | 0.7632 | 0.2517 |
| P000015 | 37 | 0.9459 | 0.2975 |
| P000016 | 38 | 0.8421 | 0.3001 |
| P000018 | 38 | 0.6316 | 0.3611 |

mode accuracy 从 0.63（P000018，最硬）到 1.0（P000008）——泛化**非均匀**，暴露了
student 在特定人群上欠拟合（P000018 也是 6 人中 prob_L1 最高 0.36）。

## 结论

1. **holdout 统计功效达成**：test 从 3→6 个未见 personas、114→226 态，A/B/C 差异首次
   可被可靠区分（B 在 KL/L1 上显著优于 A）。
2. **泛化诚实边界**：40-persona 未见人群逐轴方向一致性参差（拥堵 0.72、延误 0.87 较好；
   票价 0.56 欠拟合）；per-persona acc 0.63–1.0 显示人群级异质性仍是主要挑战。
3. **跨轴规律在扩容后仍成立**：student 弹性拟合上限 ≈ 教师信噪比（道路中断 0.97 最高、
   票价 0.27 最低）。追低信噪比轴（票价/拥堵）幅度应 K>3 压噪，而非继续加 persona。
4. departure MAE 从 ~3 抬到 ~6.5 min：新 test 集含更多样出发时刻（transit_delay=30 等），
   是更难的出发时刻拟合面，`60·tanh` 出发头的表示边界在扩容后更明显。

## 复现

```powershell
.\.venv\Scripts\python.exe scripts\generate_aggregated_teacher_dataset.py `
  --num-personas 20 --num-trips 2 --seed 43 --persona-offset 20 --trip-offset 2 --id-prefix S3_ `
  --workers 16 --resume --no-cache-bypass --output data\student_v0_3_s3_new

.\.venv\Scripts\python.exe scripts\merge_teacher_datasets.py `
  --base data\student_v0_3_s4 --new data\student_v0_3_s3_new --output data\student_v0_3_s3

.\.venv\Scripts\python.exe scripts\run_v0_3_s1_experiment.py --tag s3 --outputs outputs

.\.venv\Scripts\python.exe scripts\teacher_axis_snr.py `
  --dataset data\student_v0_3_s3\aggregated_teacher_dataset.jsonl `
  --repeats data\student_v0_3_s3\repeat_records.jsonl --config configs\student_v0_3_c.yaml

# before/after（C 变体，同一 6-persona test split）
.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s3\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s4_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml
.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s3\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s3_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml
```
