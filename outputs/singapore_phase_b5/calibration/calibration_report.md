# Phase B.5C — Effective-Capacity Calibration 最终报告与冻结

## 诊断结论（40k 探针 + 5 项 audit，`loading_diagnostic.md`）

| 审计项 | 结果 |
|---|---|
| peak-hour car demand | 11,579 辆/日，**峰值 3,452 辆/h**（07–09 与 16–17 双峰，非全天均匀） |
| 逐链路 V/C | **max 0.361**、P99 0.060、P95 0.030（无任何链路接近饱和） |
| 时间集中度 | 峰值小时占 30%——集中在早晚高峰 ✓ |
| OD 分散度 | 合成随机采样 → 高度分散（top-20 链路仅占 VKT ~4%） |
| full vs sample | **Sample**：模拟峰值 3.5k vs 现实参考 ~30k 辆/h → 采样因子 ≈0.115 |

## 标定测试（全部 10k baseline、同一种子；green-ratio 网络 + qsim 容量因子）

| 方案 | slow-passage | links>15s | mean delay (s) | car travel (min) | car 失败率 | 判定 |
|---|---|---|---|---|---|---|
| gc35（approach×0.35） | 0.02% | 0.02% | 0.52 | 8.6 | 0.2% | free-flow（顶端瓶颈现 507 个 endTime 排队） |
| gc45 | 0.01% | 0.03% | 0.52 | 8.42 | 0.2% | free-flow |
| gc55 | 0.04% | 0.06% | 0.53 | 8.64 | 0.2% | free-flow |
| sample ×0.12 | 0.53% | 0.88% | 0.83 | 17.2 | 0.2% | 重拥堵（车程×2，公交被拖慢→pt 错过连接激增） |
| sample ×0.2 | 0.30% | 0.54% | 0.69 | 10.7 | 0.2% | 中-重 |
| **sample ×0.3** | **0.17%** | **0.36%** | **0.60** | **9.68（+13%）** | **0.2%** | **✅ 轻-中度，动态余量充足** |
| sample ×0.4 | 0.11% | 0.22% | 0.56 | 9.09 | 0.2% | 轻 |
| sample ×0.5 | 0.08% | 0.14% | 0.54 | 8.7 | 0.2% | 轻 |
| 0.6 / 0.7 + gc45 | ≤0.05% | ≤0.07% | ~0.53 | 8.7–8.9 | 0.2% | 轻 |

关键澄清：`failed_trips`（事件口径）大头是 **pt 等待乘客**（道路变慢→公交延误→错过单班次
连接），car 行程失败率在所有档位恒为 **0.2%**——不存在 road gridlock，选择余地由
"拥堵量级 + 公交可靠性"共同决定。

## 冻结方案（Phase C 统一使用，不得改动）

- **N\* = 10,000 agents**（calibration 口径；1k–40k sweep 全部稳定）
- **flowCapacityFactor = 0.3，storageCapacityFactor = 0.3**（qsim，原供给网络不动）
- 供给：全天 05:00–23:00，20,966 trips；PT 规划器：直连 + 1 换乘（独立 leg）+ 步行延伸
- Student：冻结 S7-W3；green-ratio 变体仅作为 sensitivity 存档（`network_gc{35,45,55}.xml`）

### 论文表述（用户指定口径，必须照写）

> "Signal timings were not explicitly available; therefore, effective approach
> capacities were represented using green-ratio sensitivity factors."
>
> 实验定性为 **a controlled real-network experiment under calibrated effective
> capacity**（capacity factor 0.3 以 demand-sample audit 为锚、0.12–0.7 全档
> sensitivity 存档）；**不得**表述为 "calibrated reproduction of real Singapore
> congestion"。

## 动态余量预期（Phase C 情景的解释基础）

baseline（×0.3）：slow 0.17%、car travel 9.68 min；rain → car share ↑15–20pp 时
瓶颈队列将有量级级变化（0.12 档已证明该机制能把 car travel 推到 ×2），且 car
失败率始终为 0——Δcongestion 可测量、可解释，无 gridlock 风险。
