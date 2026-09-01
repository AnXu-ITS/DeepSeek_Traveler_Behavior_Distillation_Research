# Reference MATSim Integration Pipeline

> 把本仓库的「Student → MATSim」部署流程封装成可复用的 Reference Pipeline：
> **陌生用户 clone 后：装依赖 → 准备 travelers CSV → 改一个 YAML → 跑一条命令 →
> 得到 MATSim-ready population.xml（可选继续跑 MATSim）。**
>
> 定位：**Reference MATSim Integration Pipeline**（本项目部署流程的参考实现），
> **不是** Universal MATSim Adapter，也不声称适配任意 MATSim 项目。

---

## 1. What this pipeline does

- 读取 **traveler population CSV**（static 出行者属性 + trip 属性），严格校验字段；
- 读取 **scenario YAML**（weather / congestion / fare / delay / disruption 等动态环境）；
- 加载 **frozen S9 Student**（`releases/s9_supply_aware_v2/`，只读，绝不重训、不改权重）；
- 在 **真实供给**（OSM 路网 + GTFS 时刻表，本仓库新加坡 Tampines + Pasir Ris）上：
  1. 用生产 PT 规划器计算每个 OD 的可达性向量（`plan_accessibility`）；
  2. 构造 car/pt/bike/walk 备选项（`build_real_alternatives`，S9 修正速度）；
  3. **batch inference**（默认 bs=256，已审计：决策与逐态 `decide()` 完全一致）；
  4. 用生产 leg 路由构造 MATSim plan（`_build_legs` / `_plan_pt`，含 pt fallback 规则）；
  5. 写出 `population.xml` + `config.xml` + 决策 manifest + 运行摘要；
- 把 **route/accessibility 结果持久化到磁盘缓存**（网络/时刻表/路由状态绑定，见 §9）；
- 可选：用现有 `RunMatsimPreloaded` launcher 直接运行 MATSim（`lastIteration=0`，
  MATSim 只执行预生成的 plan，不重规划）。

**关键语义**：Student 只产生 **behavioral decisions**（mode / departure shift）；
**MATSim 执行** Student 决策生成的 plan。Student 不做 route planning。

## 2. What it does NOT do

- 不重训 Student、不改权重、不调用 Teacher LLM/API（不需要 `.env` 密钥）；
- 不修改冻结 release、不写 `evidence/` / `reports/` / 既有 `outputs/` 正式结果
  （数据隔离见 `docs/DATA_ISOLATION_CHECK.md`）；
- 不做 MATSim 在线 replanning / event-triggered runtime 决策 / JVM 内嵌 ONNX；
- 不做任意城市自动识别——供给必须由本仓库的 `src/traveler_distillation/singapore/`
  工具链准备（OSM + GTFS → network_with_transit.xml + transitSchedule.xml + …）；
- 不声称适配其他 MATSim 项目（定位为 reference，非 universal adapter）。

## 3. Requirements

- Python ≥ 3.12；`pip install -e .[dev]`（torch / pydantic / networkx / PyYAML /
  zstandard 等，CPU 即可）
- Java 25 + MATSim 2026.0（仅 `--run-matsim` 需要；官方
  matsim-libs release 解压为 `tools/matsim-2026.0-release/matsim-2026.0/`）
- **冻结 Student checkpoint**：`releases/s9_supply_aware_v2/checkpoint/model.pt`
- **已准备的 MATSim 供给**（网络 + 时刻表 + 停靠站索引；新加坡供给的复现命令见
  `data/singapore/DATA_README.md`——本仓库默认路径已指向它）

## 4. Installation

```powershell
# 1. 依赖
.venv\Scripts\python.exe -m pip install -e .[dev]

# 2. 供给与 checkpoint（若本机还没有）
#    - releases/s9_supply_aware_v2/  随仓库提供（git 跟踪）
#    - data/singapore/transit/*      由 data/singapore/DATA_README.md 的复现命令生成
#    - tools/matsim-2026.0-release/  官方 matsim-libs GitHub release 解压（仅 --run-matsim）

# 3. 测试（可选）
.venv\Scripts\python.exe -m pytest
```

## 5. Input schema

输入是一个 CSV，**每行 = 一个 traveler 的一次 trip**（同一 traveler 多行 = 多 trip）。
列名按 canonical 名使用；如果用户的 CSV 列名不同，用 YAML `feature_mapping` 显式映射
（不自动猜字段）。

| field | required | type | unit | allowed values | missing behavior |
|---|---|---|---|---|---|
| `persona_id` | yes | str | - | any non-empty string | error |
| `age_group` | yes | str | - | 18-24 \| 25-34 \| 35-44 \| 45-64 \| 65+ | error |
| `income_group` | yes | str | - | low \| medium \| high | error |
| `occupation` | yes | str | - | student \| office_worker \| service_worker \| manual_worker \| retired \| unemployed \| other | error |
| `household_size` | yes | int | persons | >= 1 | error |
| `has_children` | yes | bool | - | true/false (1/0, yes/no) | error |
| `car_ownership` | yes | bool | - | true/false (1/0, yes/no) | error |
| `driving_license` | yes | bool | - | true/false (1/0, yes/no) | error |
| `bike_ownership` | yes | bool | - | true/false (1/0, yes/no) | error |
| `transit_pass` | yes | bool | - | true/false (1/0, yes/no) | error |
| `habitual_mode` | yes | str | - | car \| pt \| bike \| walk \| mixed | error |
| `schedule_flexibility` | yes | str | - | low \| medium \| high | error |
| `mobility_limitation` | yes | str | - | none \| mild \| significant | error |
| `trip_id` | yes | str | - | any non-empty string | error |
| `purpose` | yes | str | - | commute \| education \| shopping \| leisure \| healthcare \| escort \| other | error |
| `origin_type` | no | str | - | any non-empty string | `home` |
| `destination_type` | no | str | - | any non-empty string | 按 purpose 的生产模板约定（commute→work, education/escort→school, shopping→shop, leisure→leisure, healthcare→healthcare, other→other） |
| `distance_km` | yes | float | km | > 0 | error |
| `desired_departure_min` | yes | int | minutes after midnight | 0 .. 1439 | error |
| `desired_arrival_min` | yes | int | minutes after midnight | 0 .. 1439 | error |
| `time_constraint` | yes | str | - | soft \| medium \| hard | error |
| `origin_node` | no | str | - | network node id（car 可达且在 walk 最大连通分量内） | 生产行为：persona_id 确定性哈希到 activity-node 池 |
| `dest_node` | no | str | - | 同上 | 生产行为：persona_id + trip 序号确定性哈希 |

`origin_node` / `dest_node` 必须成对出现（或都不出现）；显式节点会被校验为合法
activity node（car 可达 ∩ walk 最大连通分量），非法直接报错。

## 6. Configure YAML

完整示例即 `configs/reference_example.yaml`。最少只需改两个地方：

```yaml
paths:
  population_input: my_population.csv        # ← 你的输入
  output_dir: outputs/reference_pipeline/my_run
scenario:
  weather: {condition: clear, intensity: 0.0}
  road_congestion: 0.3
  fare_multiplier: 1.0
```

未知键会被拒绝（防 typo）。`scenario` 里的字段就是生产 `DynamicContext` 的字段，
没有多余项；`student.batch_size` 默认 256（审计最优）；`cache` 默认开启。

## 7. One-command quick start

```powershell
# 输入/配置自检（不加载模型、不跑管线）
python scripts/run_reference_pipeline.py --config configs/reference_example.yaml --validate-only

# 完整构建：CSV → Student 决策 → routing → population.xml + manifest + summary
python scripts/run_reference_pipeline.py --config configs/reference_example.yaml

# 构建 + 直接运行 MATSim
python scripts/run_reference_pipeline.py --config configs/reference_example.yaml --run-matsim
```

首次运行耗时参考（本机实测，CPU-only；构建时间随人口规模线性增长，
MATSim 执行时间由供给规模主导）：

| 规模 | 首次 build（cold cache） | 二次 build（warm cache） | MATSim 执行 |
|---|---|---|---|
| 25 trips（sample） | ~7 s | ~1 s | ~115 s（含 events 解析 ~43 s） |
| 500 trips | 153 s | 16 s（9.8×） | 未测（与 10k 同量级运行模式） |
| 10,000 trips | 3,224 s（≈54 min） | 470 s（≈7.8 min，6.9×，hit rate 100%） | Phase C 实测 137 s（E3） |

## 8. Output files

`<output_dir>/`：

| 文件 | 内容 |
|---|---|
| `population.xml` | MATSim population（person attributes + plan + leg routes），population_v6 DOCTYPE |
| `config.xml` | MATSim 配置（供给路径、capacity 因子、`lastIteration=0`、transit 模块） |
| `adapter_manifest.json` | 生产格式 manifest（每决策的 student/outbound/return mode、OD、leg 信息 + fallback 计数） |
| `decision_manifest.csv` | 决策表（含 per-mode 概率、departure shift、outbound/return mode、OD 节点） |
| `decision_manifest.json` | 同上（JSON） |
| `run_summary.json` | num_travelers / mode distribution / 各阶段耗时 / cache hit rate / MATSim 记录 |
| `pipeline.log` | 运行日志 |
| `cache/route_cache.pkl(.meta.json)` | 绑定式 route/accessibility 缓存（见 §9） |

`run_summary.json` 至少包含：`num_travelers`、`mode_distribution`、
`student inference time`、`feature preparation time`、`routing time`
（plan+xml）、`cache hit rate`、`total build time`；运行 MATSim 时另有
`MATSim wall-clock`、`events parse time`、`end-to-end wall-clock`。

## 9. Cache behavior

- 缓存内容：可达性向量（OD, 出发时刻）、各 mode 旅行时间、leg 最短路径。
- **绑定键（元数据头，加载时硬校验）**：network SHA256、schedule/vehicles/stops/
  snapping/trips_by_stop SHA256、routing state（access radius、buffer、换乘规则、
  walk 1.34 m/s / bike 4.17 m/s 速度常量、代码 git commit）。
- 命中 = 直接复用；miss = 用生产函数计算后写回；`rebuild: true` 强制重建。
- 日志/摘要输出：cache hit rate、routing saved（估算秒数）、cache size。

## 10. How to reuse the route cache

同一 `output_dir`（或同一 `cache.directory`）+ 相同供给 → 第二次运行自动加载
（`reuse: true`，默认）。跨情景（只改 `scenario` 不动供给）同样复用——可达性只依赖
(OD, 出发时刻, 供给)，与情景扰动无关（与生产 Phase C 共享缓存的语义一致）。

## 11. When cache becomes invalid

以下任一变化 → 元数据不匹配 → 旧缓存自动失效（日志明确告警，绝不静默复用）：
- 网络文件变化（`network_with_transit.xml` 重建/修改）
- PT 时刻表 / 车辆 / 停靠站 / trips_by_stop / snapping 变化
- 路由规则变化（access radius、buffer、换乘规则、walk/bike 速度、代码变更）

## 12. Batch inference

- 默认 `batch_size: 256`（latency 审计实测最优；`--config` 可改）。
- 正确性：batch 与逐态 `decide()` **mode 决策 100% 一致**、概率差 ≤ 1.8e-7、
  shift 差 ≤ 5.7e-6 min（浮点 GEMM 噪声），见 `docs/REFERENCE_PIPELINE_VALIDATION.md`。
- 决策吞吐 ~19.5k decisions/s（bs=256 实测）；推理只占端到端 ~0.35%，
  系统瓶颈在特征构造与 leg/XML 序列化（`docs/PIPELINE_LATENCY_AUDIT.md`）。

## 13. Run MATSim

```powershell
python scripts/run_reference_pipeline.py --config configs/reference_example.yaml --run-matsim
```

- 用 `tools/java/RunMatsimPreloaded.java` 启动器（自动编译到 ASCII 临时目录；
  Windows 非 ASCII 路径经 Temp junction，同生产脚本的处理）；
- `-Xmx6g` 可经 `matsim.java_xmx` 调整；exit code + events 指标进 `run_summary.json`。

## 14. Troubleshooting

| 现象 | 原因/处理 |
|---|---|
| `unknown key(s) in ...` | YAML 键名拼错；对照 `configs/reference_example.yaml` |
| `population input is missing required columns` | 列名不符；用 `feature_mapping` 映射（§5） |
| `origin_node ... not a valid activity node` | 显式 OD 节点必须 car 可达且在 walk 最大连通分量内；两列须成对给出 |
| `SUPPLY CHANGED since cache was built` | 正常：供给变了，缓存失效重建（§11） |
| `student checkpoint not found` / arch mismatch | checkpoint 路径错误或非 S9（`student_s8_v1`）；用 release 内 `checkpoint/model.pt` |
| `MATSim jar not found` | `matsim.matsim_dir` 指向未解压的 release（§3） |
| Java 侧 config 解析失败 | population/config 的 DOCTYPE 已写入；确认供给路径可读 |

## 15. Reproducibility

- 决策确定性：模型 eval 模式 + 固定输入 → 同输入同决策；population.xml 与原 pipeline
  **字节一致**（SHA256 相等，见 validation 报告）；
- 冻结资产：checkpoint SHA256、供给 SHA256、routing state 全部进入
  `run_summary.json` / cache 元数据；
- 本管线零随机性（persona/trip 来自用户 CSV；无 Teacher、无随机采样）。

## 16. Citation placeholder

```bibtex
@software{traveler_student_matsim_reference_pipeline,
  title = {Reference MATSim Integration Pipeline for the Distilled Traveler Agent},
  note = {Research artifact accompanying the traveler-behavior distillation study.},
  year = {2026},
}
```

---

## 附：真实链路（Reference Pipeline 封装的正是这条链）

```text
travelers.csv (static attrs)  +  scenario YAML (dynamic context)
        │                              │
        ▼                              ▼
 Persona / Trip / DynamicContext (生产 schema)
        │
        ▼  [供给: OSM 路网 + GTFS 时刻表 → SupplyIndex]
 OD 派生（确定性 persona hash） → plan_accessibility (可达性向量)
        │                              │
        ▼                              ▼
 build_real_alternatives (car/pt/bike/walk, S9 修正速度 + 情景效应)
        │
        ▼
 UniversalTravelerState → S8FeatureExtractor.encode（冻结 vocab/统计）
        │
        ▼
 TravelerStudentS8 masked-softmax → mode + departure shift   ← Student 决策（batch）
        │
        ▼
 _build_legs / _plan_pt（生产 leg 路由；car/bike SP、pt 直连/一次换乘、fallback）
        │
        ▼
 population.xml + config.xml + manifest
        │
        ▼
 java RunMatsimPreloaded config.xml（lastIteration=0）→ events → 指标
```
