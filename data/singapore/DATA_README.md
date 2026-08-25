# Singapore 数据与产物说明（Data & Artifacts Notes）

## 推送到本仓库的数据（可复现的关键输入/证据）

- `osm/tampines_pasir_ris.osm`：Overpass bbox 提取（2026-08-25，lat 1.330–1.400 ×
  lon 103.900–104.015，含全部 highway ways；ODbL）。
- `osm/query.txt`、`osm/traffic_signal_nodes.txt`（659 个 `highway=traffic_signals` 节点）、
  `osm/network_stats.json`。
- `gtfs/README_snapshot.md`、`gtfs/raw/source_metadata.json`、`gtfs/raw/checksum.sha256`：
  GTFS feed 的出处与校验（feed 为社区构建 singapore-gtfs-2025，**非官方 LTA DataMall**）。
- `transit/prep_meta.json`、`stop_snapping_report.json`、`route_routing_report.json`。
- S7 机制四联组数据在 `data/student_s7_mechanism/`（quadruplets.jsonl + split_manifest）。

## 留在本地的大文件（未推送，含复现路径）

| 文件 | 大小 | 复现方式 |
|---|---|---|
| `gtfs/raw/singapore-gtfs.zip` | 335 MB | 用户提供快照；sha256 见 `checksum.sha256`（GitHub 100MB 限制） |
| `osm/network.xml` 及全部 `network_*.xml` | 35–60 MB/个 | `src/traveler_distillation/singapore/osm_network.py` + `build_transit.py` |
| `transit/transitSchedule.xml` / `transitVehicles.xml` | ~500 MB / ~1 MB | `gtfs_prep.py`（全天 05:00–23:00）+ `build_transit.py`（215/215 序列路由） |
| `transit/trips_by_stop.json` / `activity_nodes.json` | 38.6 / 5 MB | 同上（构建产物） |
| `outputs/**` 运行产物（events.zst、checkpoints、population.xml 等） | GB 级 | 各 `scripts/singapore/*` 运行器可重建；报告 md 已推送 |

## 复现命令（Singapore 供给 + Phase A 门禁）

```bash
# 1) 供给
python -c "from traveler_distillation.singapore import osm_network; \
osm_network.build_network('data/singapore/osm/tampines_pasir_ris.osm', \
'data/singapore/osm/network.xml','data/singapore/osm/network_stats.json')"
python -c "from traveler_distillation.singapore import gtfs_prep; \
gtfs_prep.prep_gtfs('data/singapore/gtfs/raw/singapore-gtfs.zip','data/singapore/transit')"
python -c "from traveler_distillation.singapore import build_transit; \
build_transit.build_transit('data/singapore/osm/network.xml', \
'data/singapore/transit/prep_stops.jsonl','data/singapore/transit/prep_trips.jsonl', \
'data/singapore/transit', time_windows=[(5*3600, 23*3600)])"
# 2) Phase A 门禁（100 agents, exit=0, 四模式）
python scripts/singapore/run_phase_a_smoke.py
# 3) B.5C 标定网络（green-ratio sensitivity 存档）
python scripts/singapore/build_calibrated_networks.py
```

## 冻结的 Singapore 实验设置（Phase C 统一使用）

- N* = 10,000 agents；qsim `flowCapacityFactor=storageCapacityFactor=0.3`；
- 全天供给 20,966 trips；PT 规划器 = 直连 + 1 换乘（独立 leg）+ 步行延伸（1.5 km）；
- Student = 冻结 S7-W3（`outputs/student_s7_w3/checkpoints/best.pt`，本地）；
- 论文表述：*"a controlled real-network experiment under calibrated effective
  capacity"*（不称 "calibrated reproduction of real Singapore congestion"）。
