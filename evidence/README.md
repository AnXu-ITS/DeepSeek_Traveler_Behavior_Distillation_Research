# evidence/ — 论文证据快照（GitHub 可下载版）

生成于 2026-08-29（数字一致性核验与 provenance 修复后）。用途：合作者**仅凭 GitHub zip** 即可核对论文全部数字与结论，无需重新跑数小时~数天的实验。超过 ~20 MB 的原始工件不入库，清单与再生成命令见 §3。

## 1. 证据索引

| 子目录 | 实验 / 报告 | 一句话结论 | 关键文件 |
|---|---|---|---|
| `phase_c_s9/` | Phase C 主实验（frozen S9 × Singapore，六情景 C0–C5） | 雨→PT +20.0pp；延误→PT −15.1pp；道路中断→car −26.5pp、VKT −91%；六情景 gate 全过 | 6×`phase_c_result.json` + `phase_c_records.json`；报告 `reports/PHASE_C_SINGAPORE_REPORT.md` |
| `s9_eval/` | S9 模型侧评估（accessibility / regression / unseen OD） | PT-MAE −0.040\*、FVR 0.333→0.083、回归门禁全过 | `accessibility_eval.json`、`regression_eval.json`、`unseen_od_audit.json`；报告 `reports/EXPERIMENT_REPORT_S9_TRANSIT_ACCESSIBILITY_V2.md` |
| `e1_mnl/` | E1 MNL-B 基线（vs S9，决策层 + 六情景仿真镜像） | 静态相当（acc 0.863 vs 0.941）、动态/多条件大幅落后（T3 acc 0.668 vs 0.889、KL 0.386 vs 0.051）；C2 票价方向反转（β_cost=+0.257）如实报告 | `E1_REPORT.md` + 系数/评估 JSON |
| `e2_efficiency/` | E2 DeepSeek vs S9 速率/成本 | 时延 86.5s vs 0.33ms（≈3.2×10⁵ 倍）；10k 投影 240h vs 实测 3.3s；$21–351@10k vs $0；87/100 调用成功（13% 失败单列） | `E2_REPORT.md` + 6×逐态计时数组 + `deepseek_calls.jsonl` + `build_timing_100000.json` |
| `e3_scale/` | E3 人口扩展（1k×3 / 10k / 20k / 50k，C0） | decide 0.94–1.03 ms/态；系统瓶颈 = 可达性构建（57–63%），S9 推理与 MATSim 均非瓶颈 | `E3_REPORT.md` + 6×`e3_result.json`（含逐态计时数组） |
| `e4_multiseed/` | E4 multi-seed 稳健性（seeds 2026/42/7 × C0–C5） | 五情景响应 3/3 符号一致；C2 弱响应（Δpt +0.4/+0.5/+0.5）按预注册规则判定 stable | `E4_REPORT.md` + `e4_records.json` + 12 份逐运行 JSON |
| `e5_helsinki/` | E5 Helsinki zero-shot 迁移（frozen S9，零重训零 Teacher，C0–C5） | 五情景方向一致（C1 7/7、C2–C5 6/7）；A→E 可达性梯度方向复现但更弱（E−A −0.047 vs SG −0.153），如实报告 | `E5_REPORT.md` + `e5_records.json` + 6×情景结果 + 供给元数据 |

## 2. 冻结模型包（模型侧完整证据）

- `releases/s9_supply_aware_v2/`：S9 checkpoint（SHA256 `6af79b44bc699c00…`）、config、schema、normalization、provenance、`reports/reproduction_gate/`（accessibility/regression 复现 gate 评估 JSON）、`FINAL_S9_FREEZE.md`。
- `releases/s7_w3_generic_core_v1/`：generic baseline（24,370 参数）+ 三 seed 变体 + 冻结报告。
- `releases/s8_supply_aware_v1/`：**DEPRECATED**（walk/bike 速度数据错误），仅存档。
- 复现 gate 命令：`python scripts/freeze_s9_release.py verify-gate`（12/12 Δ=0.0000）。注意：verify-gate 比对的是 `outputs/s9_*` 重跑产物；本目录 `s9_eval/` 与 release 内 `reproduction_gate/` 提供冻结快照，重跑评估脚本后即可比对。

## 3. 未入库大文件（大小 · SHA256 · 再生成命令）

| 文件 | 大小 | SHA256 | 再生成 |
|---|---|---|---|
| `outputs/e2_efficiency/states_100000.jsonl` | 199,770,467 B | `744c589604686d41…`（与 E2 报告头一致） | `python scripts/bench_e2_efficiency.py`（状态池构建，4 workers，~88 min） |
| `outputs/e2_efficiency/decisions_100000.json` | 18,047,424 B | `5491dfffe11a76a7…` | 同上 |
| `outputs/e2_efficiency/g4_decisions_r0/r1.json` | 各 12,635,646 B | 两文件同为 `6dfb762fec28c0d2…`（哈希相同 = G4 逐位一致证据） | 同上 |
| `outputs/singapore_phase_c_s9/**/population.xml` + `adapter_manifest.json` + `output/events.zst` | ~94–160 MB/情景 + GB 级事件流 | 见 `E4_REPORT.md` §T5（部分工件 SHA） | `python scripts/singapore/run_phase_c.py`（可达性构建数小时） |
| `outputs/e5_helsinki/**`（MATSim 输出、pilot manifests） | GB 级 | 报告内已记录关键 SHA | `python scripts/helsinki/run_e5_helsinki.py` |
| `data/singapore/transit/transitSchedule.xml`（519 MB）/ `network_with_transit.xml`（47 MB）/ `trips_by_stop.json`（39 MB） | 合计 ~600 MB | 源数据 SHA 见 `releases/s9_supply_aware_v2/provenance/` | `src/traveler_distillation/singapore/` 管线从 OSM + GTFS 构建 |
| `hsl/`（HSL OSM PBF 97 MB + GTFS zip 77 MB） | 174 MB | PBF `8c59e968…`（见 `e5_helsinki/supply/helsinki_sub.meta.json`） | HSL 开放数据下载 |
| `tools/matsim-2026.0-release.zip` | MATSim 发行包 | — | 官方下载（见 §4） |

## 4. 外部数据获取

- **MATSim 2026.0**：官方 GitHub Releases（`matsim-org/matsim-libs`）→ 解压为 `tools/matsim-2026.0-release/`；需 Java 25（仿真）。
- **Singapore GTFS**（社区 feed，非官方 LTA DataMall）：来源与 SHA256 见 `data/singapore/gtfs/raw/source_metadata.json`（`1fcc6f5f…`）。
- **Helsinki HSL GTFS**：E5 供给元数据见 `evidence/e5_helsinki/supply/prep_meta.json`；OSM PBF 见 `helsinki_sub.meta.json`。
- **OSM**：Overpass bbox 下载或 Geofabrik；Singapore 研究区提取已入库 `data/singapore/osm/tampines_pasir_ris.osm`（16.8 MB）。

## 5. Provenance 关键事实（写论文引用前必读）

- 论文主模型 = **frozen S9**（Supply-Aware Traveler Agent v2.0，24,562 参数，SHA256 `6af79b44bc699c00…`）；S8 已废弃（walk/bike 速度数据错误，审计链见 `docs/S8_DEPRECATION.md`）。
- Phase C provenance 已修复：6 份 `phase_c_result.json` 均记录真实 S9 路径与 SHA256（2026-08-29，commit `10269cf`）。
- 数字一致性核验（2026-08-29，ccf-integrity-auditor）：E1–E5 报告 + 主报告全部数字与证据 JSON 逐格重算一致；1 critical + 8 minor 已全部修复。核验详情见会话记录；报告内 "No-Fabrication" 声明均已核实。

## 6. 论文写作口径（诚实边界速查）

- 表述上限："a controlled real-network experiment under calibrated effective capacity"；不得称 "calibrated reproduction of real Singapore congestion"、不得称 "real Singapore traveler behavior"。
- E5 跨城一致性为**定性方向证据**（无 Helsinki 行为标签，不做跨城统计检验）。
- E2 DeepSeek 价格为三档参考场景（真实账单口径未获得）；失败率 13%（87/100）如实单列。
- E1 MNL 拟合自教师蒸馏软标签，**不是**真实出行调查/revealed-preference 数据。
- C2 票价弱响应为稳定弱效应（E4 预注册判定），不夸大；30:00 transit 截断 artifact 全程如实标注。
