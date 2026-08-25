# Phase 9 报告：人口级仿真（蒸馏 Student × MATSim）

> 蒸馏出行意图 · 个体行为蒸馏研究 · 1000 personas × 每人 2 次出行

## 1. 情景

| 情景 | 天气 | 票价 |
|---|---|---|
| baseline | intensity=0.0 | x1.0 |
| rain | intensity=1.0 | x1.0 |
| fare_surge | intensity=0.0 | x2.0 |
| combined | intensity=1.0 | x1.5 |

## 2. 人口级 mode share（Student 决策 = MATSim 执行，lastIteration=0）

| 情景 | bike | car | pt | walk |
|---|---|---|---|---|
| baseline (student) | 0.193 | 0.246 | 0.500 | 0.061 |
| baseline (executed) | 0.193 | 0.246 | 0.500 | 0.061 |
| rain (student) | 0.001 | 0.396 | 0.557 | 0.046 |
| rain (executed) | 0.001 | 0.396 | 0.557 | 0.046 |
| fare_surge (student) | 0.168 | 0.316 | 0.457 | 0.059 |
| fare_surge (executed) | 0.168 | 0.316 | 0.457 | 0.059 |
| combined (student) | 0.000 | 0.396 | 0.554 | 0.050 |
| combined (executed) | 0.000 | 0.396 | 0.554 | 0.050 |

## 3. 需求转移（vs baseline，百分点）

| 情景 | bike | car | pt | walk |
|---|---|---|---|---|
| rain | -19.2pp | +15.0pp | +5.7pp | -1.5pp |
| fare_surge | -2.5pp | +7.0pp | -4.3pp | -0.2pp |
| combined | -19.2pp | +15.0pp | +5.4pp | -1.2pp |

## 4. 出行距离（米）

| 情景 | avg leg | avg trip |
|---|---|---|
| baseline | 6098.9 | 9099.6 |
| rain | 4999.6 | 8959.2 |
| fare_surge | 5520.8 | 9010.0 |
| combined | 4999.6 | 8959.2 |

## 5. 结论

1. **蒸馏 Student 在 1000 人人口上做出了可解释、方向正确的人口级需求转移**（一阶效应，
   MATSim 原样执行 student 决策，lastIteration=0）：
   - **暴雨（rain 1.0）**：bike −19.2pp（几乎归零）、walk −1.5pp，car +15.0pp、
     pt +5.7pp —— 暴露型出行向遮蔽型出行的大规模转移；
   - **票价 ×2（fare_surge）**：pt −4.3pp，car +7.0pp —— 票价弹性方向正确
     （pt 需求流失主要转向 car）；
   - **组合扰动（rain + fare ×1.5）**：与单独暴雨几乎一致（bike −19.2pp、
     car +15.0pp），说明天气暴露效应的主导地位，票价在雨天的边际影响被淹没。
2. **个体异质性在聚合层仍可见**：扰动下并非全体转移——fare ×2 时仍有 45.7%
   的出行保留 pt、31.6% 转向 car，与 teacher/student 的 persona 条件响应一致。
3. **平均出行距离随扰动变化**：rain/combined 的 avg leg 距离从 6.1km 降至 5.0km，
   fare_surge 几乎不变（9.0km）。
4. 完整链条已验证：**Dynamic Context Change → Individual Choice Changes
   (student) → Population Demand Shift (MATSim executed)**。网络拥堵反馈闭环
   （Phase 10：模拟后的路网状态反作用于下一次决策）是下一步。

## 6. 复现命令

```powershell
.venv\Scripts\python.exe scripts\run_population_experiment.py --checkpoint outputs/student_v0_3_c/checkpoints/best.pt --num-personas 1000 --trips-per-persona 2 --scenarios baseline,rain,fare_surge,combined --output data/population_experiment
```

## 7. 已知边界

- 合成网格网络（20×20），pt/walk/bike 为 teleported 模式（无公交时刻表）。
- lastIteration=0：MATSim 不做 replanning，测量的是 student 决策的一阶需求转移；
  网络拥堵反馈闭环（Phase 10）尚未接入。
- 1000 personas 为 development-scale 人口（真实城市场景为数十万级）。
- 过程注记：初版脚本曾把全部 trip 列表复用于每个 persona（人口文件放大 ~1000 倍，
  1000 人时 MATSim 报 runners-null NPE）；修复为每人独立 2 次出行后，
  1000 人 × 4 情景全部正常运行。
