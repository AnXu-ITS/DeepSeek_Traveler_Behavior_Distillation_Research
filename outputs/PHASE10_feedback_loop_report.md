# Phase 10 报告：网络反馈闭环（Student ↔ MATSim）

> 蒸馏出行意图 · 个体行为蒸馏研究 · 800 personas × 1 次出行 × 10×10 网格（per-link capacity 5 veh/h，为在开发规模人口下呈现拥堵而校准）

## 1. 闭环协议

每轮迭代：student 在 context（含 road_congestion 估计 c_ctx）下决策 → MATSim 执行计划 → 从 linkstats 提取观测拥堵 c_obs（capacity 加权平均延误比 / 0.5，截断到 [0,1]）→ c_ctx 以平滑系数 0.5 更新 → 下一轮。收敛条件 |c_obs − c_ctx| < 0.02。

## 2. 收敛轨迹

### baseline（converged=True）

| iter | c_ctx | bike | car | pt | walk | c_obs | mean_delay_ratio |
|---|---|---|---|---|---|---|---|
| 1 | 0.300 | 0.179 | 0.250 | 0.521 | 0.050 | 0.380 | 0.190 |
| 2 | 0.340 | 0.179 | 0.251 | 0.521 | 0.049 | 0.381 | 0.191 |
| 3 | 0.360 | 0.179 | 0.253 | 0.520 | 0.049 | 0.377 | 0.189 |

### rain（converged=True）

| iter | c_ctx | bike | car | pt | walk | c_obs | mean_delay_ratio |
|---|---|---|---|---|---|---|---|
| 1 | 0.300 | 0.000 | 0.385 | 0.570 | 0.045 | 0.438 | 0.219 |
| 2 | 0.369 | 0.000 | 0.385 | 0.570 | 0.045 | 0.434 | 0.217 |
| 3 | 0.402 | 0.000 | 0.385 | 0.570 | 0.045 | 0.441 | 0.221 |
| 4 | 0.421 | 0.000 | 0.385 | 0.570 | 0.045 | 0.442 | 0.221 |
| 5 | 0.432 | 0.000 | 0.385 | 0.570 | 0.045 | 0.444 | 0.222 |

### fare_surge（converged=True）

| iter | c_ctx | bike | car | pt | walk | c_obs | mean_delay_ratio |
|---|---|---|---|---|---|---|---|
| 1 | 0.300 | 0.152 | 0.314 | 0.485 | 0.049 | 0.401 | 0.201 |
| 2 | 0.351 | 0.150 | 0.316 | 0.485 | 0.049 | 0.402 | 0.201 |
| 3 | 0.376 | 0.150 | 0.316 | 0.485 | 0.049 | 0.402 | 0.201 |
| 4 | 0.389 | 0.150 | 0.316 | 0.485 | 0.049 | 0.402 | 0.201 |

## 3. 平衡点对比

| 情景 | 初始 c | 收敛 c_ctx | 平衡 c_obs | car share 变化 |
|---|---|---|---|---|
| baseline | 0.300 | 0.360 | 0.377 | +0.25pp |
| rain | 0.300 | 0.432 | 0.444 | +0.00pp |
| fare_surge | 0.300 | 0.389 | 0.402 | +0.25pp |

## 4. 结论

1. **闭环收敛，且所有情景存在稳定平衡点**：3 个情景分别在 3 / 5 / 4 轮内满足
   |c_obs − c_ctx| < 0.02。耦合系统（student 决策 ↔ 路网动态）没有发散或震荡——
   "Traveler Adaptation → Network State → New Conditions → Further Adaptation"
   链条在开发规模下闭环成立。

2. **扰动改变了平衡拥堵水平，方向正确**：
   - baseline 平衡 c* ≈ 0.38（初始 0.30 被路网现实校正上移）；
   - **rain 平衡 c* ≈ 0.44**：暴雨使 car share 达 38.5%（vs baseline 25%）→
     路网负载上升 → 均衡拥堵显著更高——天气扰动通过行为转移传导为网络级拥堵；
   - **fare_surge 平衡 c* ≈ 0.40**：票价翻倍使 car 31.6%（pt→car 转移）→
     均衡拥堵介于两者之间，量级与方向均合理。
3. **Student 的拥堵弹性较弱（诚实发现）**：单情景内 c_ctx 变化 0.30→0.43 只引起
   car share 变化 ≤0.25pp。这与 v0.2 教师审计一致（road_congestion 是教师响应
   噪声最大、方向反转最多的轴）。改进方向：在更大数据集中强化拥堵轴的
   反事实覆盖，或在蒸馏损失中对拥堵弹性单独加权。
4. 至此蓝图五层验证全部打通：个体生成 → 蒸馏 → 个体级验证 → 人口级仿真 →
   **交通系统闭环**。

## 5. 复现命令

```powershell
.venv\Scripts\python.exe scripts\run_phase10_loop.py --checkpoint outputs/student_v0_3_c/checkpoints/best.pt --num-personas 800 --trips-per-persona 1 --scenarios baseline,rain,fare_surge --max-iterations 6 --eps 0.02 --grid-n 10 --link-capacity 5 --output data/phase10_loop
```

## 6. 已知边界

- 合成网格 + 校准容量（cap=5 veh/h）：为在 development-scale 人口下
  产生可测拥堵而设定，非真实路网标定。
- 反馈只作用于 context.road_congestion（天气/票价固定）；student 的拥堵弹性
  较弱（v0.2 教师审计中拥堵响应亦为噪声最大轴）。
- 单次决策（无 within-day re-planning）；每轮迭代为独立一天。
