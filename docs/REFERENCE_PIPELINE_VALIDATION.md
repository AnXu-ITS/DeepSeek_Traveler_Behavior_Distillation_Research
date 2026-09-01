# REFERENCE_PIPELINE_VALIDATION — Reference Pipeline 正确性验证报告

> 验证日期：2026-09-01 · 机器：单机 24C / 31.4 GB / CPU-only（torch 2.13.0+cpu）
> 原则（`docs/DATA_ISOLATION_CHECK.md`）：Reference 只读消费冻结资产；回归输出全部写入
> `outputs/reference_pipeline/`；frozen benchmark 只读，绝不改写。

---

## 1. 验证矩阵

| 层 | 验证内容 | 方法 | 结果 |
|---|---|---|---|
| D1 | 决策层：同输入下 Reference(batch) vs 原 pipeline(逐态 `decide()`) | 200 dev-seed 状态双侧构建，逐字段比对 | **200/200 全部字段一致** |
| D2 | 数值层：batch vs 逐态的概率/时刻差 | 200 状态 | max Δprob **1.79e-7**、max Δshift **3.81e-6 min** |
| D3 | 编码层：MemoizedEncoder(静态预计算) vs 生产 `extractor.encode` | 200 状态逐字段相等 | 见 §3（200/200） |
| D4 | 工件层：population.xml / config.xml / adapter_manifest.json | SHA256 / JSON 相等 | **字节一致** |
| D5 | 缓存层：cache-hit 路径 ≡ cold 路径 | 同输入 warm 重跑 | 决策与工件不变（§4） |
| D6 | 冻结基准层：10k 只读回归 vs frozen Phase C C0 manifest | 10,000 决策 × 7 字段 | §6（行为决策 10,000/10,000） |
| D7 | 运行层：fresh-user smoke（sample CSV + 一条命令 + MATSim） | 干净输出目录 | exit 0（§5） |
| D8 | 回归层：仓库既有测试 + 新增 reference 单测 | `pytest` | **172 passed** |

---

## 2. 方法（同输入、两条路径）

- **输入**：同一批 personas/trips（dev seed=0，N=200，`configs/generation_v0_1.yaml`
  生成器，与 DATA_ISOLATION_CHECK 的 dev-seed 约定一致）、同一 C0 context、
  同一冻结供给与 checkpoint（S9，SHA256 运行前校验）。
- **Path A（原 pipeline）**：`S8MATSimAdapter.build_real_scenario` +
  `run_phase_c.make_shared_idx / make_shared_factory` 的逐字副本（生产 Phase C 代码路径，
  进程内可达性缓存 + 旅行时间 memo + 逐态 `decide()`）。
- **Path B（Reference）**：`reference_pipeline.matsim_adapter.build_scenario` +
  `StudentAdapter.predict`（batch=256）+ 持久化 `RouteCache`（cold，rebuild）。

## 3. 结果（fixture，N=200，dev seed=0）

| 检查 | 结果 |
|---|---|
| student_mode 一致 | **200/200** |
| outbound_mode 一致 | **200/200** |
| return_mode 一致 | **200/200** |
| departure_min / departure_shift_min（2 位小数）一致 | **200/200** |
| home_node / dest_node 一致 | **200/200** |
| outbound / return leg info（route_links、pt fallback 原因等）一致 | **200/200** |
| batch vs 逐态 `decide()`：mode | **200/200** |
| batch vs 逐态：max \|Δshift\| | **3.81e-6 min**（浮点 GEMM 噪声） |
| batch vs 逐态：max \|Δprob\| | **1.79e-7** |
| MemoizedEncoder vs 生产 `extractor.encode` | **200/200 逐字段相等** |
| population.xml SHA256 | **相等**（`aa57754a2e51e185…`） |
| config.xml SHA256 | **相等** |
| adapter_manifest.json（JSON 语义） | **相等** |
| 原路径 build / Reference build | 72.0 s / 77.5 s（同数量级；Reference 额外持久化 2,316 条 SP 路由） |

**结论 D1–D5：Reference Pipeline 与原 pipeline 在同输入下决策 100% 一致，
population.xml / config.xml / manifest 字节级一致；XML 顺序与内容完全相同
（不是"语义一致"而是逐字节一致）。**

## 4. Cache 路径等价性（N=500 benchmark）

同一 config 连续两次运行（cache 目录相同）：

| | run 1（cold） | run 2（warm） |
|---|---|---|
| total build | 153.5 s | **15.6 s**（9.8×） |
| feature preparation | 105.6 s | **0.01 s** |
| student inference（batch 256） | 0.02 s | 0.02 s |
| plan + XML | 47.9 s | 15.6 s |
| cache 调用 | 8,608（15 hit） | **8,608（100% hit）** |
| cache 条目（acc / tt / sp） | 500 / 1,485 / 6,608 | 同左 |
| cache 体积 | 3.64 MB | 同左 |
| 估算 routing 节省 | — | **139.4 s** |
| mode 分布 | bike 176 / car 146 / pt 114 / walk 64 | **完全相同** |

- warm 运行时特征侧 105.6s → 0.01s（~10,000×），瓶颈转移到 leg 路由文本组装 + XML
  序列化（15.6s）——与 `docs/PIPELINE_LATENCY_AUDIT.md` 的预测一致
  （"缓存后瓶颈转移到 leg 路由 + XML 序列化"）。
- 缓存绑定元数据（网络/时刻表/停靠站 SHA256 + routing state + git commit）在加载时
  硬校验；供给或路由规则变化 → 缓存失效重建（`docs/REFERENCE_PIPELINE.md` §9–11）。

## 5. Fresh-user smoke test（D7）

严格按 `docs/REFERENCE_PIPELINE.md` 的陌生用户流程，干净输出目录
`outputs/reference_pipeline/example/`：

1. 读 README / 用户文档；
2. 使用 `examples/sample_population.csv`（20 travelers / 25 trips，0 显式 OD）；
3. 直接用 `configs/reference_example.yaml`（未改任何字段）；
4. 一条命令 `--validate-only` → PASS（0.004 s）；
5. 一条命令 build → 6.8 s（cold，含 25 条可达性规划 + 236 次路由调用）→
   `population.xml`（249 KB）、`config.xml`、`adapter_manifest.json`、
   `decision_manifest.csv`、`run_summary.json`、`pipeline.log`；
6. 二次运行（cache reuse）→ **0.94 s**，hit rate **100%**；
7. `--run-matsim` → MATSim **exit 0**，wall 114.5 s（供给主导：全天 20,966 班次）；
   events 解析 43.1 s（1,922 万事件）；四模式 leg 全部执行
   （car 24 / bike 14 / walk 15 / pt 5）；stuck persons 0；
   1 例 `no_route_car → walk` fallback 走生产 fallback 规则如实记录；
8. population.xml 校验：MATSim 成功加载并执行（最强校验）+ 字节级与原 pipeline
   一致（§3）。

用户全程不需要手动执行任何内部脚本；输入文件只有 1 个 CSV；配置只改
`population_input` / `output_dir`（示例即默认）。

## 6. 10k frozen-benchmark 只读回归（D6）

Reference 用 seed 2026 重建 Phase C C0 population（N=10,000，与冻结 C0 完全相同的
人-OD-情景），对冻结 manifest（`outputs/singapore_phase_c_s9/C0_baseline/adapter_manifest.json`）
逐字段比对；frozen 输入/结果全程只读（输出仅写 `outputs/reference_pipeline/`）。

| 字段 | 一致 | 说明 |
|---|---|---|
| `student_mode` | **10,000/10,000** | 行为决策 100% 一致 |
| `outbound_mode` | **10,000/10,000** | |
| `return_mode` | **10,000/10,000** | |
| `home_node` / `dest_node` | **10,000/10,000** | OD 派生一致 |
| `departure_shift_min` / `departure_min` | **9,998/10,000** | 2 例为 2-dp **舍入边界翻转**（1.88↔1.87、4.49↔4.48，0.01 min = 0.6 s）；对这两行实测原始 shift 差 **≤ 1.6e-6 min**（batch GEMM 浮点噪声，与 §3 D2 同量级）——原始值在 2 位小数边界两侧各 1e-6 min 处 |
| mode 分布 | **完全一致** | pt 2,531 / bike 3,399 / car 2,828 / walk 1,242（= 冻结 C0 的 25.3 / 34.0 / 28.3 / 12.4%） |
| leg info（route_links / pt fallback 原因） | 一致 | manifest `outbound`/`return` 全字段 |

运行量：cold build **3,223.6 s**（≈54 min；E3 原 pipeline 实测 3,063 s，差额为持久化
175,933 次路由调用）；warm 重跑 **469.5 s**（≈7.8 min，**6.9×**），cache hit rate
**100%**（179,063/179,063 调用），估算 routing 节省 **2,853 s**；缓存条目
acc 9,996 / tt 26,874 / sp 139,063。10k 决策本身仅 ~1 s（batch 256）。

**结论**：Reference Pipeline 在冻结 Phase C 基准上的 **行为决策（mode）与 OD 10,000/10,000
一致**；唯一的不一致是 2/10,000 行的出发时刻 2-dp 舍入边界翻转（0.01 min），原始差为
浮点噪声（≤1.6e-6 min）。这与 latency 审计的口径一致（审计仅承诺 mode 一致；
shift 差 ≤5.7e-6 min 为噪声级）。
