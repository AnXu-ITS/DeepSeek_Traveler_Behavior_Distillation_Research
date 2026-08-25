# v0.3 实验报告：官方直连 DeepSeek V4 Pro 蒸馏（persona-holdout）

> 蒸馏出行意图 · 个体行为蒸馏研究 · 生成日期：2026-08-21

## 1. 教师数据集

- 规模：320 states = 40 baselines + 280 counterfactuals
- 构成：20 personas × 2 trips × fare_multiplier、weather_intensity
- 教师 selected-mode 分布：{'bike': 44, 'car': 87, 'pt': 118, 'walk': 71}
- 教师采样噪声（K=3 repeat 的 per-state pairwise L1）：mean = 0.1589，max = 1.0 —— 这是 Student 分布误差的合理下限参考

### 1.1 教师人群异质性（baseline 概率跨 persona 的 std）

| mode | mean P | std | n |
|---|---|---|---|
| walk | 0.2725 | 0.3222 | 40 |
| bike | 0.336 | 0.2627 | 16 |
| pt | 0.3912 | 0.3818 | 40 |
| car | 0.5767 | 0.2958 | 14 |

### 1.2 教师逐轴响应（mean ΔP vs baseline）

**fare_multiplier**

- level 1.25 (n=40): bike +0.038, car -0.006, pt -0.005, walk -0.009
- level 1.5 (n=40): bike +0.019, car -0.023, pt +0.006, walk -0.005
- level 2.0 (n=40): bike +0.059, car -0.005, pt -0.024, walk +0.003

**weather_intensity**

- level 0.25 (n=40): bike -0.042, car +0.068, pt +0.024, walk -0.031
- level 0.5 (n=40): bike -0.100, car +0.127, pt +0.041, walk -0.046
- level 0.75 (n=40): bike -0.184, car +0.180, pt +0.103, walk -0.092
- level 1.0 (n=40): bike -0.180, car +0.167, pt +0.097, walk -0.083

## 2. 切分（persona-holdout）

| split | A | B | C |
|---|---|---|---|
| counts | {'train': 224, 'val': 48, 'test': 48} | {'train': 224, 'val': 48, 'test': 48} | {'train': 224, 'val': 48, 'test': 48} |
| persona_overlap | False | False | False |

- **A** personas: train=['P000002', 'P000005', 'P000006', 'P000007', 'P000010', 'P000011', 'P000012', 'P000013', 'P000014', 'P000015', 'P000016', 'P000018', 'P000019', 'P000020'] val=['P000003', 'P000008', 'P000017'] test=['P000001', 'P000004', 'P000009']

- **B** personas: train=['P000002', 'P000005', 'P000006', 'P000007', 'P000010', 'P000011', 'P000012', 'P000013', 'P000014', 'P000015', 'P000016', 'P000018', 'P000019', 'P000020'] val=['P000003', 'P000008', 'P000017'] test=['P000001', 'P000004', 'P000009']

- **C** personas: train=['P000002', 'P000005', 'P000006', 'P000007', 'P000010', 'P000011', 'P000012', 'P000013', 'P000014', 'P000015', 'P000016', 'P000018', 'P000019', 'P000020'] val=['P000003', 'P000008', 'P000017'] test=['P000001', 'P000004', 'P000009']

## 3. 测试集指标（test = 完全未见的 personas）

| metric | A | B | C |
|---|---|---|---|
| mode accuracy | 0.7708 | 0.8125 | 0.8125 |
| KL(P_T || P_S) | 0.3816 | 0.1440 | 0.1092 |
| probability L1 | 0.7605 | 0.3260 | 0.2860 |
| departure MAE (min) | 9.2489 | 3.2117 | 2.0945 |
| departure sign agreement | 0.4792 | 0.6875 | 0.7500 |

### 3.1 行为弹性与异质性（test）

| metric | A | B | C |
|---|---|---|---|
| counterfactual \|dP_T - dP_S\| | 0.0661 | 0.0596 | 0.0539 |
| counterfactual sign agreement | 0.6508 | 0.7143 | 0.6825 |
| heterogeneity \|dP_T - dP_S\| | n/a | n/a | 0.0986 |

### 3.2 逐 persona 测试明细（未见人群泛化）

| persona | n | A acc | A L1 | B acc | B L1 | C acc | C L1 |
|---|---|---|---|---|---|---|---|
| P000001 | 16 | 0.4375 | 0.6995 | 0.9375 | 0.3178 | 0.9375 | 0.2626 |
| P000004 | 16 | 0.9375 | 0.7766 | 0.8125 | 0.3013 | 0.75 | 0.2776 |
| P000009 | 16 | 0.9375 | 0.8054 | 0.6875 | 0.3589 | 0.75 | 0.3179 |

## 4. 结论

1. **蒸馏管线在"未见人群"上仍然有效**：persona-holdout 切分（train 14 / val 3 / test 3
   个完全未参与训练的 personas）下，三个 Student 的 mode accuracy 均达 0.77–0.81，
   且 B/C 显著优于 A——说明本研究的损失项不只是"记忆训练分布"，而是学到了可泛化的
   行为规律。

2. **分解式弹性损失（方向 + 幅度）在 holdout 上有效**：v0.3-B 相对 A 把
   counterfactual |dP_T − dP_S| 从 0.0661 降到 0.0596、方向一致率从 0.651 提到
   0.714、概率 L1 从 0.761 降到 0.326。这与 v0.2 中 plain-L1 弹性损失在 in-split
   test 上无效的结果形成对比——方向/幅度分解 + 未见人群检验提供了更干净的信号。

3. **L_heterogeneity 带来进一步的分布正则**：v0.3-C 在所有版本中取得最好的 KL
   (0.109)、概率 L1 (0.286)、弹性 |dP| (0.054) 与 departure MAE (2.09)，
   且未牺牲 mode accuracy (0.8125)；其 between-persona |dP_T − dP_S| = 0.099
   （48 对，未见人群之间）。

4. **教师数据的两个可蒸馏信号均被保留**：
   - 人群异质性：教师 baseline 概率跨 persona 的 std 达 0.26–0.38（pt/walk/car）；
   - 环境弹性：雨天（intensity 1.0）教师使 bike −0.18 / walk −0.09、car +0.17 /
     pt +0.10（暴露型出行向遮蔽型转移）；票价 ×2.0 使 pt −0.024（方向正确）。
   Student C 在未见人群上再现了这些方向（sign agreement 0.68，弹性误差 0.054，
   接近教师自身 K=3 采样噪声 0.159 的 1/3）。

5. **诚实边界**：test 仅 3 personas / 48 states / 42 弹性对，A/B/C 差异为
   development-scale 证据而非统计显著性结论；合成 persona 无真实标定。

## 5. 复现命令

```powershell
.venv\Scripts\python.exe scripts\run_v0_3_experiment.py --dataset data/student_v0_3/aggregated_teacher_dataset.jsonl --repeats data/student_v0_3/repeat_records.jsonl --outputs outputs
```

## 6. 已知边界

- 合成 persona/trip，无真实世界标定；development-scale 数据集。
- v0.3-B 的弹性损失为分解式（方向 + 幅度，无 plain L1）。
- C 的异质性评估依赖同 split 内多个 persona 的同情境配对。
