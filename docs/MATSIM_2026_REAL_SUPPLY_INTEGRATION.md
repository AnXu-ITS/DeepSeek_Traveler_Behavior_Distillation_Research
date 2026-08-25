# MATSim 2026 真实供给网络集成经验文档

> 项目：Singapore 真实网络验证（Phase A，2026-08-25 门禁通过）
> 环境：MATSim 2026.0（`tools/matsim-2026.0-release/`）、Java 25（Adoptium）、纯 Python 供给管线
> 适用：任何把自产 network.xml / transitSchedule.xml / population.xml 喂给 MATSim 2026 的工程

## 0. 总览：本轮攻克的 12 个集成问题

| # | 问题 | 现象 | 解法 |
|---|---|---|---|
| 1 | 网络/车辆 XML 元素 id 不唯一 | "already a node/link with id" | 转换器按 (from,to) 去重合并；ai 节点/链路只写一次 |
| 2 | 车辆文件旧 DTD 格式 | "Required attribute id missing from <departure>" 等 | 2026 用 XSD 格式（`<vehicleType>`+`<standingRoom>`+schemaLocation） |
| 3 | `<departure>` 缺 id | 同上 | 每 departure 写唯一 id |
| 4 | 模式双定义冲突 | "mode bike is defined both as teleportation and network routing" | `clearDefaultTeleportedModeParams=true` + 显式 `non_network_walk` 参数 |
| 5 | 网络模式一致性检查 | "Network for mode walk has unreachable links" | `networkRouteConsistencyCheck=disable`（所有 leg 均为显式路由时安全） |
| 6 | Raptor 参数为 null | Guice "Object is null"（SwissRailRaptorFactory） | 提供 non_network_walk teleported 参数 + swissRailRaptor modeMapping（bus/rail→pt） |
| 7 | 输出目录非空 | "output directory already exists and is not empty" | `overwriteFiles=deleteDirectoryIfExists` |
| 8 | trip 无 routingMode | "Found a trip with multiple legs and no routingMode" | `handlingOfPlansWithoutRoutingMode=useMainModeIdentifier` |
| 9 | leg 带 routingMode 属性反而报错 | "Element <leg> has no attribute routingMode"（未声明属性） | **去掉** leg 的 routingMode 属性（population_v6.dtd 未声明，2026 靠 MainModeIdentifier） |
| 10 | plans dump 对 pt 路线 NPE | `Route.getEndLinkId() is null`（shutdown 必触发） | pt 路线元素写 `start_link`/`end_link` 属性 |
| 11 | leg 路线惯例错误（最隐蔽） | 全员 stuckAndAbort、turn-acceptance 警告 | **路线必须从 agent 当前所在的活动 link 开始**；活动 link 必须为入向 link |
| 12 | transit 车辆首站语义 | "Transit vehicle is not yet at last stop" | 车辆路线必须以首站接入链 ai_in_0 起始（先进入首站再出发） |
| 13 | 换乘与 2026 umlauf 冲突 | "vehicle at last stop but still contains passengers" | Phase A 关闭换乘（直连）；换乘须配合车辆链或独立 leg 结构 |

## 1. 供给 XML 规范细节

### 1.1 车辆文件（2026 = XSD，不再是 dtd）
```xml
<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.matsim.org/files/dtd http://www.matsim.org/files/dtd/vehicleDefinitions_v1.0.xsd">
  <vehicleType id="busType">
    <capacity><seats persons="40"/><standingRoom persons="20"/></capacity>
    <length meter="12.0"/>
    <accessTime secondsPerPerson="1.0"/>
    <doorOperation mode="serial"/>
    <passengerCarEquivalents pce="2.0"/>
  </vehicleType>
  <vehicle id="veh_xxx" type="busType"/>
</vehicleDefinitions>
```
- 元素名是 `<vehicleType>`（不是 vehicleDefinition）；`<standingRoom persons=…/>`（不是
  seats 上的 standingRoomInPersons 属性）；必须带 xsi:schemaLocation。

### 1.2 transitSchedule（dtd 仍有效，但注意）
- `<departure id=… departureTime=… vehicleRefId=…>` 三属性中 **id 必填**（dtd #REQUIRED）。
- routeProfile 中间站需要 `departureOffset`（dtd 要求除最后一站外必有）；末站只写
  `arrivalOffset` + `awaitDeparture="false"`。
- `<stop>` 的 refId 引用 stopFacility；stopFacility 的 `linkRefId` 是该站所在链路的 id。
- 网络/时刻表/车辆中的所有 id 必须全局唯一（同 id 不同属性直接抛异常）。

### 1.3 路线的语义（最重要，见 §4）
- 车辆路线链接序列 = 从首站接入链开始、按站依次 ai_in/ai_out 交替、以末站 ai_in 结束；
  每个 profile 站的 ai_in 必须按序出现在链接序列中（可用脚本全量断言）。

## 2. config.xml 关键参数（2026）

```xml
<module name="controller">
  <param name="overwriteFiles" value="deleteDirectoryIfExists"/>
  <param name="writePlansInterval" value="0"/>
  <param name="lastIteration" value="0"/>
</module>
<module name="plans">
  <param name="handlingOfPlansWithoutRoutingMode" value="useMainModeIdentifier"/>
</module>
<module name="qsim">
  <param name="endTime" value="30:00:00"/>
  <param name="mainMode" value="car,bike,walk"/>   <!-- 单数参数名，逗号分隔 -->
</module>
<module name="routing">
  <param name="networkModes" value="car,bike,walk"/>
  <param name="clearDefaultTeleportedModeParams" value="true"/>
  <param name="networkRouteConsistencyCheck" value="disable"/>
  <param name="routingRandomness" value="0.0"/>
  <parameterset type="teleportedModeParameters">
    <param name="mode" value="non_network_walk"/>
    <param name="beelineDistanceFactor" value="1.3"/>
    <param name="teleportedModeSpeed" value="1.39"/>
  </parameterset>
</module>
<module name="transit">
  <param name="useTransit" value="true"/>
  <param name="transitScheduleFile" value="..."/>
  <param name="vehiclesFile" value="..."/>
  <param name="transitModes" value="pt"/>
</module>
<module name="swissRailRaptor">
  <parameterset type="modeMapping">
    <param name="routeMode" value="bus"/><param name="passengerMode" value="pt"/>
  </parameterset>
  <parameterset type="modeMapping">
    <param name="routeMode" value="rail"/><param name="passengerMode" value="pt"/>
  </parameterset>
</module>
```

要点：
- `clearDefaultTeleportedModeParams` 会**即时清除**（含 helper 模式），必须随后显式补回
  `non_network_walk`（Raptor 的 TransitRouterConfig 硬性要求它存在，否则 Guice 报
  "Object is null"）。
- QSim 多主模式参数名是 `mainMode`（单数），值是逗号分隔列表。
- 2026 的默认 pt 路由模块是 SwissRailRaptor：无 modeMapping 时乘客模式=route mode
  （bus/rail），与 transitModes=pt 不匹配，必须显式映射到 pt。

## 3. population.xml 规范细节

- 腿元素**不要**写 `routingMode` 属性：population_v6.dtd 未声明该属性，wstx 验证会以
  "Element <leg> has no attribute routingMode" 拒绝（这条错误信息极具误导性，实际是
  "DTD 中没有这个属性"）。多腿 trip 的 routingMode 由
  `handlingOfPlansWithoutRoutingMode=useMainModeIdentifier` 处理。
- pt 腿的 `<route type="default_pt">` 必须带 `start_link`/`end_link` 属性（上/下车站的
  接入链 id），否则：① shutdown 时 plans dump 抛 NPE（`getEndLinkId() is null`，
  DumpDataAtEndImpl 无条件调用）；② 乘客到达检查 "destination link null" 触发 ABORT。
- pt 路线描述 JSON 键（Jackson 序列化确认）：`accessFacilityId`、`egressFacilityId`、
  `transitLineId`、`transitRouteId`、`boardingTime`（"HH:MM:SS" 字符串）。
- 活动元素：`link` 属性必须是**入向链路**（终点=活动节点）；每段腿的路线必须以
  该活动链路开头（见 §4）。

## 4. MATSim 腿路线惯例（本项目最大的坑）

规则：**一段腿的 route 链路序列必须以 agent 出发时所在的活动链路作为第一项**；
活动位于其 `link` 属性的链路终点。

- 错误做法（自产路线常见）：route = [node 出发的第一条出向链路, …]。QSim 会把 route[0]
  当作"当前所在链路"而跳过，第一次转向检查发生在 当前链路→route[1]，二者不相邻 →
  "Cannot move vehicle … from link X to link Y" → 全员 stuckAndAbort。
- 正确做法：
  - outbound：route = [home 活动链路] + path(home→dest)，其中 path 末链路的终点 = dest
    节点，dest 活动链路 = path 的末链路（由路线**派生**，不要另行挑选）。
  - return：route = [dest 活动链路] + path(dest→home)，最终 home 活动链路 = path 末链路。
  - pt 接驳步行：access = [活动链路] + path + [ai_in_o]；egress = [ai_in_d, ai_out_d] +
    path（乘客下车时"所在链路"是 ai_in_d）。
  - 校验脚本化：所有 leg 路线逐对断言 `link_i.to == link_{i+1}.from`（本项目的
    population 校验 0 断裂；MATSim 侧再断裂基本可排除自产路线问题）。

## 5. transit 执行语义

- 车辆 spawn 在路线第一条链路起点：**路线必须以首站的 ai_in 开头**（车辆先驶入首站，
  再经 ai_out 出发），否则司机全程错位一站，报
  "Transit vehicle is not yet at last stop"。
- 末站到达后车辆必须为空：乘客必须全部在 profile 站下车。乘客登车判定
  （BoardingAcceptance.checkLineAndStop）：线路 id 匹配 **且** 期望下车站 ∈ 剩余站序。
- **换乘（chainedRoute）与 2026 的 umlauf 车辆链机制冲突**：单班次车辆没有 chained
  departure，换乘乘客在换乘站被移入 agentRelocating 后无人接管，或错过换乘车辆时
  滞留车上直到终点触发 assert。Phase A 的结论：单班次车辆 + 直连规划；换乘要么配合
  车辆链（vehicles 跨 route 复用），要么用 Raptor 生成独立 pt leg + 换乘步行腿的结构。
- 错过自己车次的乘客会在站台永久等待（无害，用事件统计"未完成行程"），不要为此
  在调度里造虚拟班次。

## 6. 运行器（Java 25 坑）

- Guice（7.0）自带 ASM 无法解析 Java 25 类文件（major 69）——**任何 Guice
  CreationException 的错误详情格式化都会先崩**，真实错误被吞。对策：启动器用
  `javac --release 21` 编译（本项目 `tools/java/RunMatsimPreloaded.java`）。
- 输出目录：`overwriteFiles=deleteDirectoryIfExists`，否则残留输出直接拒绝运行。
- 事件文件在 2026 是 `output/ITERS/it.0/0.events.xml.zst`（zstandard），不是 .gz；
  增量写出，运行中断时可解析到崩溃点——这是调试滞留乘客/失败行程的最快入口。
- 乘客上下车事件是 **PersonEntersPtVehicle / PersonLeavesPtVehicle**（不是
  PersonEntersVehicle——后者是司机/私家车事件）。

## 7. 自产供给管线的通用防御清单

- [ ] 所有 XML id 全局唯一（节点/链路/站/线路/车次/车辆/departure）
- [ ] transitRoute 链路序列：链连续 + 每站 ai_in 按序出现（脚本断言）
- [ ] 每段 leg 路线：链连续 + 以活动链路开头（脚本断言）
- [ ] 活动链路 = 入向链路
- [ ] pt 路线带 start_link/end_link
- [ ] 网络模式集合与 config 的 networkModes / mainMode 一致
- [ ] 车辆文件用 XSD 格式
- [ ] 启动器 --release 21
