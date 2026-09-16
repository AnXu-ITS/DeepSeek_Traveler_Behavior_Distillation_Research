# Shanghai（徐汇区）数据与产物说明

> 研究区：**徐汇区核心**，bbox `lat 31.15–31.23 × lon 121.40–121.48`（≈ 6.7×7.7 km，约 50 km²，一个区规模）。
> 投影：UTM 51N（EPSG:32651，中央经线 123E），`src/traveler_distillation/singapore/projection.py::utm51n`（进程内 swap，不动新加坡默认值）。

## 供给已建成（端到端已跑通，可复现）

| 文件 | 说明 | 来源 |
|---|---|---|
| `osm/xuhui.osm` | Overpass bbox 提取，ODbL | 真实 |
| `osm/network.xml` | MATSim 路网（30,425 节点 / 56,810 links / 1,914 km） | 真实 |
| `osm/stations.json` | 52 个地铁站元素（Overpass，含入口重复） | 真实 |
| `transit/amap_stations.json` | **34 个地铁站 + 线路归属**（高德 POI，站级去重） | 真实 |
| `gtfs/raw/shanghai-gtfs-synthetic.zip` | 合成地铁 GTFS（站点/线路真实、时刻表合成） | 混合 |
| `transit/*.xml / *.json` | prep + build_transit 产物（schedule/vehicles/network_with_transit/snapping/routing） | 派生 |
| `outputs/reference_pipeline/shanghai_xuhui/` | S9 决策 + population.xml + MATSim 迭代0输出 | 派生 |

## 复现命令（全流程）

```powershell
# 1) 拉真实地铁站（高德 Web服务 key，见 .env 的 AMAP_KEY）
python scripts/shanghai/fetch_amap_metro.py

# 2) 合成地铁 GTFS（站点/线路真实；发车间隔=典型值假设）
python scripts/shanghai/build_synthetic_gtfs.py

# 3) prep + 构建公交供给（snap 停靠站、路由、写 schedule/vehicles/network_with_transit）
python scripts/shanghai/build_shanghai_transit.py data\shanghai\gtfs\raw\shanghai-gtfs-synthetic.zip

# 4) frozen S9 决策 -> population.xml
python scripts/run_reference_pipeline.py --config configs\reference_shanghai.yaml

# 5) 跑 MATSim（lastIteration=0；需 JDK25 + Temp junction，见 scripts/run_matsim.ps1）
```

> 路网重构建：`python scripts/shanghai/build_shanghai_network.py`

## 诚实边界（务必写进报告）

1. **站点/线路 = 真实**（高德 POI + OSM 交叉）；**时刻表 = 合成**（发车间隔：高峰 3 min / 平峰 6 min / 21:00 后 10 min；首班 05:30、末班 22:30；站间行驶时间 = 直线距离 / 35 km/h，限幅 90–300 s）。
2. **线路站序 = 几何近似**（投影到线路主轴排序）：对徐汇区内的直线段地铁线路准确，环线（4 号线）局部段也成立，但**不保证全网络精确站序**。论文级供给请用真实 Transitland feed（`f-shanghai~metro`，需免费 key：`scripts/shanghai/download_shanghai_gtfs.py`）替换。
3. **供给只有地铁、无公交**（上海无公开公交 GTFS；高德/百度也不提供公交时刻表）。
4. 地铁按 **rail-on-road** 处理（OSM 转换不含 railway way，与新加坡 MRT / 赫尔辛基 rail 的 Phase A 退化一致）。
5. 容量因子 0.3/0.3 是新加坡标定，搬用到上海**不声称复现真实拥堵**，只作受控有效容量实验。
6. frozen S9 **零重训、零 Teacher、不改 normalization/schema**；上海样本是 covariate shift，只报告、不 refit。
7. `coordinate_system` 已参数化（上海 `EPSG:32651`，新加坡/赫尔辛基默认 `EPSG:32648`）。

## 真实 GTFS（论文级，可选）

```powershell
set TRANSITLAND_API_KEY=你的key
python scripts/shanghai/download_shanghai_gtfs.py      # -> data/shanghai/gtfs/raw/shanghai-gtfs.zip
python scripts/shanghai/build_shanghai_transit.py data\shanghai\gtfs\raw\shanghai-gtfs.zip
```
