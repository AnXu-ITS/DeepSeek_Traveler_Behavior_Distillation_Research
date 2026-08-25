# Singapore Phase A Gate Report — 真实供给网络跑通 ✅

**阶段**：Singapore Phase A（`NEXT_STEP_PLAN_SINGAPORE_AIT.md` v2.0 §2）
**门禁定义**：真实 OSM + scheduled PT + S7 Student population 在 MATSim 中完整执行
**结果**：**GATE PASSED（2026-08-25）**

## 1. 门禁证据

| 检查项 | 结果 |
|---|---|
| 100 个 S7-W3 agents baseline 运行 exit code | **0** |
| car / pt / walk / bike 四模式 executed legs | **全部出现**（departure 事件按 legMode 计数） |
| scheduled PT 实际执行 | **5,580 个 transit 车次全部发车**（TransitDriverStarts=5,580） |
| PT 乘客上下车 | **8 上车 / 8 下车**（PersonEnters/LeavesPtVehicle 配对） |
| stuckAndAbort | **0** |
| 行程完整性 | departure 5,796 = arrival 5,796（无失败行程） |

执行后 mode share（outbound executed）：walk 52 / car 25 / bike 15 / pt 8（Student 决策
为 pt 48 / car 31 / bike 15 / walk 6；pt→walk 回退 40 次、car→walk 回退 6 次，见 §3）。
平均 leg 距离 6.12 km、平均 trip 距离 6.61 km。

## 2. 供给资产（data/singapore/）

| 产物 | 规模 |
|---|---|
| `osm/tampines_pasir_ris.osm`（Overpass，2026-08-25） | 102,784 节点 / 30,140 ways |
| `osm/network.xml`（UTM 48N） | 100,867 节点 / 191,617 链路 / 3,520 km；car 最大连通分量 95% |
| `transit/transitSchedule.xml` | 851 站 / 170 线路 / **5,580 车次**（含 256 个 MRT 车次） |
| `transit/transitVehicles.xml` | 5,580 辆（busType/railType） |
| `transit/network_with_transit.xml` | +851 站台接入链（ai_in/ai_out）+28,681 transit 专用反向链（busr_*）+25,441 行人/自行车反向链（pdr_*） |

GTFS 来源：`data/singapore/gtfs/raw/singapore-gtfs.zip`（社区构建 feed singapore-gtfs-2025，
**非官方 LTA DataMall**，快照/校验见 `data/singapore/gtfs/README_snapshot.md`）；
工作日 WD 服务日、早高峰 06:00–10:30 窗口、供应裁剪 = 保留区域内最长连续站段
（18,327/21,286 trips 裁剪）。路由：191/191 唯一站序列路由成功、0 失败。

## 3. 本阶段已记录的工程近似（诚实边界，Phase B/C 沿用）

1. **MRT rail-on-road**：OSM 转换不含 railway，MRT 车次沿道路图路由（transportMode=rail）。
2. **单行道 transit 反向链（busr_*）**：GTFS 线路在 OSM 单行道建模 + snapping 误差下
   存在有向不可达；为 bus/rail 生成仅 transit 可用的反向链（car 交通不受影响）。
3. **行人/自行车反向链（pdr_*）**：步行/骑行可在单行道两侧通行（物理正确）。
4. **PT 直连无换乘**：Phase A 的 agent 只搜索直达车次；1 次换乘的 chainedRoute 与
   MATSim 2026 的 umlauf 车辆链机制在单班次车辆上不兼容（换乘乘客滞留至终点导致
   assert 崩溃）——已关闭换乘，无直达时步行回退并计数。换乘规划留待 Phase C。
5. **时窗**：transit 供给仅覆盖早高峰 06:00–10:30；**回程腿全部落在窗外** →
   回程 pt 全部步行回退（计 48 次 pt_fallback_walk 中的回程部分）。
6. **需求侧 synthetic**：personas/trips 与 Student 输入的 alternative 属性仍为合成值
   （模型线冻结不冻结需求生成）；活动点采样自站点 300m 内的真实路网节点。
7. 网络转换属性（freespeed/capacity）按 highway 类别映射，未经实测标定。

## 4. 过程中修复的关键工程问题（沉淀为代码）

- OSM 转换：节点/链路去重合并、双向单行道建模、孤立节点过滤；
- MATSim 2026 集成：`population_v6` 腿路由惯例（**路线必须从当前活动 link 开始**）、
  transit 车辆首站需以 ai_in 起始、`departure` 元素必需 id、车辆文件 XSD 格式、
  QSim 主模式 + teleported 参数冲突、Raptor 配置（swissRailRaptor modeMapping、
  non_network_walk 参数）、活动 link 必须为入向 link、plans dump 对 pt 路线需
  start_link/end_link 属性；
- 启动器 `RunMatsimPreloaded.java`（--release 21 编译，规避 Guice ASM 对 Java 25
  类文件的解析崩溃）。

## 5. 下一步（Phase B — 规模化验证）

按计划书 §3：100 → 500 → 1000 agents，记录 routing failures / PT boardings /
mode share / mean trip time / road delay / runtime / failed trips / network
congestion，确认规模扩大无异常。进入 Phase B 前可选改进：全天时窗供给、
换乘规划（Raptor 路线或自研直连扩展）。
