# Singapore 真实网络验证计划
## v2.0 — 模型训练线冻结 · 交通验证线主线化

**项目主线**：DeepSeek V4 Pro Teacher → 轻量 Traveler Agent（已审计、已冻结）→ MATSim 真实网络
**计划版本**：v2.0（2026-08-24，替代 v1.0，旧版归档于 `archive/legacy_20260821/root/NEXT_STEP_PLAN_SINGAPORE_AIT_v1.0.md`）
**状态**：阶段转换已完成 —— 模型训练线冻结，交通验证线成为主线。

---

# 0. 阶段转换声明

> 前面在证明 **Student 值不值得信**；现在开始证明 **这个已经审计过的 Student 放进真实交通供给系统后有没有研究价值**。

**模型线冻结（Freeze，即刻生效）**：

| 冻结项 | 值 |
|---|---|
| 最终 Student | **S7-W3**（`outputs/student_s7_w3/checkpoints/best.pt`，Grade B + 4-seed 稳定） |
| 回退点 | S5-Joint-M2（`outputs/student_s5_joint_m2/checkpoints/best.pt`） |
| 不再做的事 | 不扩 persona、不加扰动轴、不重训 Teacher、不做机制补训 S8、不换 Student 架构 |

重新训练主模型的唯一条件：发现明确的数据错误或方法错误（v1.0 Phase 11 Gate 保持有效）。

**论文表述基调**（S7 收口结论）：Student 在 output/joint 行为蒸馏下保持预测性能，机制保真为
“axis-dependent mechanism preservation under targeted mechanism-aware supervision”；
网络验证评估的是**行为可执行性与系统响应**，而非完整因果机制保真。

---

# 1. 四阶段总览

| 阶段 | 名称 | 核心问题 | 成功标准 | 论文角色 |
|---|---|---|---|---|
| **A** | 真实供给网络跑通 | 真实 OSM + scheduled PT + S7 Student 能不能在 MATSim 里完整执行？ | 100 agents baseline exit=0，四模式全可跑 | 可行性门禁（不是论文实验） |
| **B** | 规模化验证 | 100→500→1000 agents 规模扩大有没有异常？ | 各项运行指标随规模平稳、无异常断裂 | 规模证据 |
| **C** | 正式论文情景 | 真实网络上扰动情景的行为响应 | baseline + 5 单轴 + 联合情景全部完成 | 主实验 |
| **D** | 真实网络反馈闭环 | Phase 10 的 Student→MATSim→congestion→再决策能否迁移到真实网络？ | 闭环收敛 | 系统级证据 |

**执行纪律**：Phase A 未过门禁，不进入 B/C/D；任何阶段出现无法修复的供给构建阻塞时，
按 §8 的降级路径处理，不允许为了“结果好看”伪造供给。

---

# 2. Phase A — 真实供给网络跑通（当前唯一目标）

> 第一阶段只设一个明确目标：**先把真实供给网络跑通**。
> 成功标准不是“结果好看”，而是：**真实 OSM + scheduled PT + S7 Student population 能在 MATSim 中完整执行**。

## 2.1 区域

**Tampines + Pasir Ris**（v1.0 §4.1 不变）：

- 裁剪约 **10–15 km 级别**区域；
- 目标 bbox（初始值，执行时可微调）：`lat 1.330–1.400, lon 103.900–104.015`
  （GTFS 实测该范围含 **746 个公交站**，MRT East-West Line 穿区而过）；
- 区域包含 car / bus / MRT / walk / bike 全部模式，居住功能明显，路网规模可控。
- 备用区域：Jurong East + Clementi（仅当 Tampines/Pasir Ris 线路匹配严重失败时启用）。

## 2.2 数据与产物清单

```text
data/singapore/
  osm/
    source_metadata.json          # 下载源/日期/bbox/校验
    singapore.osm.pbf             # 原始下载（或 overpass .osm）
    tampines_pasir_ris.osm        # 裁剪后
    network.xml                   # OSM → MATSim 转换产物
    network_stats.json            # 节点/链路/连通分量/速度/容量统计
  gtfs/
    raw/singapore-gtfs.zip        # 已有（工作区根目录，需移入并留快照）
    README_snapshot.md            # 数据性质说明（见 §2.3）
    source_metadata.json
    checksum.sha256
  transit/
    transitSchedule.xml           # GTFS → MATSim
    transitVehicles.xml
    stop_snapping_report.json     # 每站最近 link 距离、失败站清单
    route_routing_report.json     # 每线路 routing 失败/降级清单
outputs/
  singapore_phase_a/
    smoke_100/                    # 100 agents baseline 运行产物（含 exit code 记录）
```

## 2.3 GTFS 数据现状与表述边界（已侦察）

- `singapore-gtfs.zip` **已存在于工作区根目录**，内容实测：
  - 6 个 agency（LTA + SBS Transit / SMRT / Tower Transit / Go-Ahead / 公交运营实体）；
  - **603 条线路**：593 条 bus（route_type=3）+ **9 条 MRT**（route_type=1）；
  - 5,376 stops、230,915 trips、8,169,065 stop_times、4 个 calendar 服务日模式；
  - 覆盖全新加坡（lat 1.25–1.49），目标区域 746 站。
- **表述边界（论文必须遵守）**：bus 时刻为 LTA DataMall 性质数据、旅行时间含估算；
  MRT 为 frequency/synthetic 性质 schedule；**不得**把整个 feed 表述为“官方实测 GTFS timetable”。
- 数据来源、下载日期、checksum 必须写入快照文件，保证可追溯。

## 2.4 Phase A 执行步骤（顺序严格）

1. **数据归档**：GTFS zip 移入 `data/singapore/gtfs/raw/`，生成快照 + metadata + sha256；
2. **OSM 获取**：Overpass bbox 下载（首选）或 Geofabrik Singapore 全量 + 裁剪；
   生成 `source_metadata.json`（日期、bbox、源）；
3. **OSM → network.xml**：使用 MATSim 自带 `org.matsim.core.utils.io.OsmNetworkReader`
   （已在 `tools/matsim-2026.0-release/matsim-2026.0/matsim-2026.0.jar` 内确认存在，
   Java 25 可用）；转换配置：highway 过滤（motorway…residential/service）、capacity/freespeed
   映射、mode 允许集（car,bike,walk; pt 走 transit network）；
4. **network 质量检查**：连通分量、单向链路、link length/freespeed/capacity 分布、孤立节点、CRS
   （GTFS 为 WGS84，MATSim 采用投影坐标——用 SVY21/EPSG:3414 或 UTM 48N，全链统一）；
5. **GTFS → transitSchedule.xml + transitVehicles.xml**：
   - 区域过滤：只保留穿过/停靠裁剪区域的线路（保留跨区域线路的区域内区段或裁剪其服务范围）；
   - **stop-to-link snapping**：每个 stop 投影后吸附到最近可用 link（记录吸附距离，超阈值标记）；
   - **route routing**：每个 trip 的 stop 序列在 road network 上做 shortest-path routing 生成 route path；
   - MRT：route_type=1 的线路需要 rail 路段（OSM railway=rail/subway）——若 OSM 转换不含 rail，
     走“rail-on-road 替代 + 频率化 schedule”的显式降级并在报告中记录；
   - transitVehicles：按 route 配置车辆容量与发车（departures 来自 stop_times / frequencies）；
6. **100-agent smoke**：S7-W3 Student（复用 `MATSimAdapter`，替换 synthetic grid 为真实供给）→
   `RunMatsimPreloaded` 运行，**exit=0**；
7. **模式完备性验证**：car / pt / walk / bike 四个模式的 leg 都能在真实网络上生成并被执行
   （pt 从 teleported 升级为 scheduled transit——这是本阶段的核心升级）。

## 2.5 已知风险清单（v1.0 §12.3，逐项排查）

1. GTFS stop 不在任何 road link 上 → snapping 报告 + 超阈值处理策略；
2. bus route 与 OSM 单行道不兼容 → routing 需按车辆模式限制（car-like）；
3. route stop 序列 routing 不可达 → 跳过/降级并记录；
4. CRS 不一致 → 统一投影链；
5. 线路跨出研究区域 → 裁剪策略（保留区域内运行段或剔除线路）；
6. transit vehicle / departure 配置不完整 → 校验脚本；
7. MRT 与 road-based bus routing 混合 → rail 供给缺失时按 §2.4 显式降级并记录。

## 2.6 Phase A Gate（过门禁 = 进入 Phase B）— ✅ 2026-08-25 全部达成

- [x] `network.xml` 质量检查全通过（连通分量、CRS、链路属性；car 最大连通分量 95%）
- [x] transitSchedule 覆盖区域内主要 bus + MRT，route routing 失败率记录在案且策略已执行
      （5,580 车次 / 170 线路 / 851 站；191/191 序列路由成功、0 失败）
- [x] stop snapping 报告生成，超阈值站有明确处理（mean 60m / p90 186m / max 688m，
      全部经人工接入链 ai_in/ai_out 连接）
- [x] **100 个 S7-W3 agents baseline 运行 exit=0**
- [x] car / pt / walk / bike 四模式均有实际 executed legs
- [x] 所有产物与报告落入 `data/singapore/` 与 `outputs/singapore_phase_a/`，可复现

门禁报告：`outputs/singapore_phase_a/PHASE_A_GATE.md`（含全部工程近似与诚实边界）。
**执行证据**：5,580 transit 车次全部发车、pt 8 上车/8 下车、stuckAndAbort=0、
departure 5,796 = arrival 5,796、avg trip 6.6 km。

---

# 3. Phase B — 规模化验证（100 → 500 → 1000）✅ 2026-08-25 完成

**目标**：确认规模扩大没有异常（不是追求好看数字）。

| 规模 | 100 | 500 | 1000 |
|---|---|---|---|
| 每次运行记录 | routing failures | PT boardings | mode share |
| | mean trip time | road delay | runtime |
| | failed trips | network congestion | |

- 同一供给网络、同一 Student 决策管线，只放大 population；
- 异常判据：boardings/failed-trips/mode share 出现与规模不成比例的突变、runtime 超线性爆炸、
  congestion 在 baseline 下失真（如全网瘫痪）；
- 产出 `outputs/singapore_phase_b/scale_report.md` + 规模对比表。

**结果**：失败行程 0 / stuckAndAbort 0（全部规模）；PT boardings 12→55→127（线性）；
mean trip time 8.96→9.67→9.62 min；road delay 0.51 s/passage（跨规模不变）；
congestion ≈0；runtime 27–29 s（平稳，5,580 transit 车次主导）；mode share 漂移 <5pp。
**判定：规模扩大无异常 → 进入 Phase C。**

---

# 3.5 Phase B.5 — Experimental Validity Gate（2026-08-25）

## B.5A PT Routing Validity ✅
- 目标：Student intended PT 的 itinerary 构建成功率 ≥90%，回退分类。
- 供给升级：**全天 05:00–23:00（20,966 trips / 859 站）**；规划器：直连 + **1 次换乘
  （独立 pt leg + 换乘步行，无 chainedRoute）** + 步行延伸（1.5km）。
- **结果 98.0%（intended 904 → routed 886：direct 687 / transfer 199 / fallback 18，
  全部 no_direct_or_transfer）**；MATSim：上车 1,082=下车 1,082、不平衡车辆 0。
- 报告：`outputs/singapore_phase_b5/pt_validity_v3/pt_validity.md`。

## B.5B + B.5C Demand Loading & Effective-Capacity Calibration ✅ 方案冻结
- 诊断（40k）：峰值 car 3,452 辆/h、链路 V/C max 0.361/P99 0.06、sample 因子 ≈0.115 →
  需求单边扩展不可行（需 80万–120万 agents）。
- 标定测试（10k × 同一种子）：green-ratio 三档（0.35/0.45/0.55）→ free-flow；
  qsim 容量因子 0.12→重、0.3→**轻-中度（slow 0.17%、car travel +13%）**、0.5–0.7→轻；
  **car 失败率全档 0.2%（无 gridlock）**。
- **冻结：N\*=10,000 + flowCapacityFactor=storageCapacityFactor=0.3**；论文表述
  "controlled real-network experiment under calibrated effective capacity"。
- 报告：`outputs/singapore_phase_b5/calibration/calibration_report.md`、
  `loading_diagnostic.md`。**Phase C 全部情景统一使用冻结设置，不得改动。**

# 4. Phase C — 正式论文情景

在真实网络上跑主实验（每个情景独立运行，复用 Phase B 的 1000-agent 模板）：

1. baseline；
2. heavy rain；
3. PT fare increase；
4. transit delay；
5. road disruption；
6. 联合情景：rain + congestion、fare + congestion 等。

情景注入方式：扰动经 Student 的 context 输入进入个体决策（与 synthetic 阶段同构），
网络供给保持不变；产出模式转移、网络级延误、PT 使用变化等系统级指标。

---

# 5. Phase D — 真实网络反馈闭环

把 Phase 10 已跑通的闭环迁移到 Singapore 网络：

```text
Student 决策 → MATSim 执行 → 观测 link 拥堵 c_obs → 平滑更新 c_ctx → Student 再决策 → 收敛
```

- 拥堵状态作用于真实 OSM link（而非 synthetic grid）；
- 保持 Phase 10 的收敛判据与平滑参数口径，逐情景报告均衡 c* 与收敛轮数。

---

# 6. 数据可追溯规则（贯穿 A–D）

- 所有外部数据（OSM/GTFS）记录：来源 URL、下载日期、版本/feed 起始日期、SHA-256、bbox；
- 所有转换配置（OSM 过滤规则、投影、snapping 阈值、routing 算法）写入配置文件并随产物存档；
- 所有运行记录：seed、Student checkpoint 路径、MATSim 版本（2026.0）、Java 版本（25）、exit code。

---

# 7. 当前资产盘点（已侦察确认）

| 资产 | 状态 |
|---|---|
| MATSim 2026.0（含 `matsim-2026.0.jar` + 全部依赖） | ✅ `tools/matsim-2026.0-release/` |
| `OsmNetworkReader`（OSM→network.xml） | ✅ 在核心 jar 内 |
| Java 25 | ✅ Adoptium jdk-25 |
| `singapore-gtfs.zip`（603 线路/5376 站/23 万 trips） | ✅ 工作区根目录 |
| `RunMatsimPreloaded`（预加载 scenario 运行器） | ✅ `tools/java/`（Phase 8 遗留） |
| `MATSimAdapter`（checkpoint → plans/attributes） | ✅ `src/traveler_distillation/matsim/adapter.py`（需从 synthetic grid 扩展到真实供给） |
| OSM Singapore 提取文件 | ❌ 需下载 |
| GTFS→transitSchedule 转换器 | ❌ 需实现（首选 matsim-pt2matsim 依赖接入，失败则自研 Java 转换器：stop snapping + Dijkstra route path） |

---

# 8. 停止条件与降级路径

- **Phase A 阻塞**：若 bus route routing 大面积不可达（>20% trips 无法形成 route path），
  降级顺序：① 放宽 snapping 阈值 / 修网络断层 → ② 剔除不可达线路（保留 MRT + 主 bus 走廊，
  记录剔除比例）→ ③ 切换备用区域 Jurong East + Clementi → ④ 全部失败则冻结 Phase A，
  论文改为 synthetic-grid 网络验证 + 真实供给构建的方法学章节。
- **任何阶段**：不伪造供给、不删不利样本、不把降级路径的结果表述为完整真实网络。
- **模型线**：Phase A–D 全程不再训练新 Student；如 Singapore 运行暴露 Student 决策的
  表示边界（如 S2 时代 departure ±60 裁剪），记录为已知边界而非重训理由。

---

# 9. 与 v1.0 的映射（Phase 11–21 → A–D）

| v1.0 | v2.0 | 说明 |
|---|---|---|
| Phase 11 冻结 | §0 完成 | S7-W3 已 Freeze |
| Phase 12 数据接入 | **Phase A** | 拆细、加 gate |
| Phase 13 规模 | **Phase B** | 明确 100/500/1000 |
| Phase 14 主情景 | **Phase C** | 顺序调整到供给跑通之后 |
| Phase 15 反馈闭环 | **Phase D** | 不变 |
| Phase 16–21（MNL baseline / 因果 / plausibility / 效率 / 统计） | 暂缓 | Phase C 完成后再按需启用（P1/P2 优先级） |

---

# 10. 一句话执行原则

> **先让真实供给网络在 MATSim 里完整跑起来（Phase A exit=0），再谈规模、情景与闭环。**
