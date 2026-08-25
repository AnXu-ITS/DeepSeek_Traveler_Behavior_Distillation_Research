# Phase B.5A — PT Routing Validity Gate ✅

- supply：**全天服务（05:00–23:00），20,966 trips / 859 站**（215/215 序列路由成功）；
- 规划器：直连 + 1 次换乘（**独立 pt leg + 换乘步行，无 chainedRoute**）
  + 步行延伸（上下车站点可在起讫点 1.5 km 内）；
- 样本：1000 agents（seed=3000），Student 决策由冻结 S7-W3 给出。

## 结果表

| 方向 | intended PT | routed PT | direct PT | transfer PT | fallback | success rate |
|---|---|---|---|---|---|---|
| outbound | 455 | 449 | 346 | 103 | 6 | 0.9868 |
| return | 449 | 437 | 341 | 96 | 12 | 0.9733 |
| **合计** | 904 | 886 | 687 | 199 | 18 | **0.9801** |

## Fallback 原因分类（合计）

- `no_direct_or_transfer`: 18（直连、1 次换乘、步行延伸均无可行组合；非服务缺失）
- `no_stops_in_radius`: 0
- `no_service_window`: 0（全天供给消除了出发时间不可达）

## MATSim 执行验证

- exit=0；**pt 上车 1,082 = 下车 1,082**（换乘乘客全部完成两段行程）；
- 上下车不平衡车辆 0（换乘滞留类崩溃消失）；
- stuckAndAbort 3：全部发生在 **endTime 30:00:00** 的站台等待清理
  （乘客错过其单班次车辆后等待至模拟结束，3/1,082 = 0.3%，属单班次调度语义
  而非 itinerary 构建失败，Phase C 用 waitingForPt 事件统计为错过连接）。

## 判定

- 目标 ≥90%：**达成 ✅（98.0%）**。
- 迭代过程：75.9%（双时窗+直连）→ 84.1%（放宽搜索）→ 98.0%（全天供给+换乘+步行延伸）。
