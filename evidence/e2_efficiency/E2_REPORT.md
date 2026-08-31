# E2 DeepSeek vs S9 速率/成本 — 实验报告

> 设计书：`TRC_AIT_5_E2_DEEPSEEK_S9_EFFICIENCY_DESIGN.md` v0.2（依据 `TRC_AIT_5_EXPERIMENT_PLAN.md` §E2）
> 本报告所有数字由 `scripts/make_e2_report.py` 从证据 JSON 生成，无手抄数字。
> 状态池：outputs\e2_efficiency\states_100000.jsonl（SHA256 `744c589604686d41…`，N=100,000，seed=2026）
> 冻结 S9 checkpoint：releases\s9_supply_aware_v2\checkpoint\model.pt（SHA256 `6af79b44bc699c00…`）

## 0. Gate 汇总

| gate | 内容 | 结果 |
|---|---|---|
| G1-S9(t=1) | 3 次重复 mean/P95 CV ≤ 10%（mean CV=1.4%，p95 CV=2.8%） | PASS |
| G1-S9(t=default) | 3 次重复 mean/P95 CV ≤ 10%（mean CV=0.9%，p95 CV=6.4%） | PASS |
| G1-DS | 全部调用有效 JSON+usage（13 失败：api_error×10, empty_content×2, timeout×1；统计基于 87 次成功调用） | FAIL |
| G2 | S9 在 10k 前缀的决策与冻结 Phase C C0 逐位一致（比对 10,000 条，0 处不一致） | PASS |
| G3 | DeepSeek 调用数 ≤ 预算（100/120，成功 87，失败 13） | PASS |
| G4 | 两次 time-s9 复跑决策逐位一致（100,000 条，0 处不一致） | PASS |

## 1. T1 — 单次决策时延（同 100 态前缀对拍）

| 决策者 | mean | P50 | P95 | min | max | 单位 |
|---|---|---|---|---|---|---|
| DeepSeek（100 次调用 / 87 次成功实测，cache_bypass，K=1） | 86.539 | 87.414 | 162.793 | 11.387 | 184.215 | s |
| DeepSeek mean 95% bootstrap CI（B=2000, seed=42） | [76.745, 96.550] | | | | | s |
| S9 CPU 顺序（threads=1，同 100 态前缀） | 0.272 | 0.250 | 0.351 | 0.231 | 1.019 | ms |
| S9 CPU 顺序（threads=1，100,000 态全量） | 0.332 | 0.265 | 0.797 | 0.218 | 59.652 | ms |
| S9 CPU 顺序（默认线程=24，100,000 态全量） | 0.334 | 0.264 | 0.815 | 0.215 | 52.899 | ms |

- 时延比（DeepSeek mean / S9 threads=1 mean，同 100 态前缀）: **318,534×**（数量级）
- DeepSeek 分档（同状态池前缀，实测）：

| N | mean | P50 | P95 |
|---|---|---|---|
| 1 | 116.093 s | 116.093 s | 116.093 s |
| 10 | 70.964 s | 79.500 s | 162.793 s |
| 50 | 90.645 s | 89.170 s | 166.717 s |
| 100（87 成功） | 86.539 s | 87.414 s | 162.793 s |

## 2. T2 — 吞吐

| 决策者 | 口径 | 吞吐 | 10k 等效墙钟 |
|---|---|---|---|
| DeepSeek | 顺序实测（N=100） | 0.7 states/min | 240.4 h（projection） |
| DeepSeek | 4-worker 投影（历史锚点 5–6 次/分钟） | ~5–6 states/min | ~28–33 h（projection） |
| S9 | CPU 顺序 threads=1（实测） | 3015 states/s | 3.3 s |
| S9 | batch 32（encode+forward，实测 n=20,000） | 21909 states/s | 0.5 s |
| S9 | batch 256（encode+forward，实测 n=20,000） | 25399 states/s | 0.4 s |
| S9 | batch 1024（encode+forward，实测 n=20,000） | 21049 states/s | 0.5 s |

## 3. T3 — Token 与成本

- DeepSeek per-state token（实测 usage，87 次调用）：
  prompt 中位 1,341 [IQR 1,341–1,343]；completion 中位 4,152 [IQR 1,940–5,745]；reasoning 中位 4,024；total 中位 5,494

| 行 | $/state（V3 / reasoner / hi-end） | cost@10k | cost@100k |
|---|---|---|---|
| DeepSeek（实测 token × 参考价格，按逐调用 token 平均） | $0.0021 / $0.0096 / $0.0351 | $21 / $96 / $351（projection） | $208 / $962 / $3515（projection） |
| S9（蒸馏后推理，零外部调用） | $0 / $0 / $0 | $0 | $0 |
| 一次性蒸馏投入参考行（1,502 次调用，真实 token：in 2,431,492 / out 5,309,636） | $2.91 / $12.97 / $47.34（总额） | — | — |

## 4. T4 — S9 人口级端到端（无 MATSim，实测）

| N | 共享输入构建 | 决策时间 | 总墙钟 | 输入 ms/state | 决策 ms/state |
|---|---|---|---|---|---|
| 1 | 0.1 s | 0.0 s | 0.1 s | 105.9 | 3.29 |
| 100 | 26.2 s | 0.1 s | 26.3 s | 262.3 | 1.04 |
| 1,000 | 258.7 s | 1.0 s | 259.7 s | 258.7 | 1.00 |
| 10,000 | 2215.8 s | 10.5 s | 2226.3 s | 221.6 | 1.05 |
| 100,000 | 20808.8 s | 98.4 s | 20907.2 s | 208.1 | 0.98 |

> 注：T4 的时间来自 100k 状态池构建过程（4 进程并行，共享输入构建含可达性规划；见诚实边界 §7）。DeepSeek 对应行：N=100 实测总墙钟 10598 s（顺序，含重试与调用间隔）；10k/100k 为线性投影（10k 顺序 ≈ 240.4 h；按历史 4-worker 锚点 ~5.5 states/min ≈ 30.3 h），见 T2。

## 5. T5 — 模型体量与部署形态

| 项 | S9 | DeepSeek |
|---|---|---|
| 参数量 | 24,562 | 未公开（remote API） |
| 模型文件 | 108,533 B（~106 KB，fp32） | — |
| 推理硬件 | CPU（本机实测，torch 2.13.0+cpu） | API 服务（无需本地硬件） |
| GPU 需求 | 无 | — |
| 外部调用/付费 | 无（蒸馏后） | 每决策 1 次付费调用（K=1） |

## 6. T6 — headline 比值（由 T1–T3 计算）

| 比值 | 值 |
|---|---|
| 时延比（DeepSeek mean / S9 threads=1 mean，同 100 态） | 318,534× |
| 吞吐比（S9 batch1024 / DeepSeek 顺序） | 1,821,566× |
| 10k 成本比（DeepSeek / S9） | ∞（S9=$0）；DeepSeek 绝对值为 T3 的 cost@10k 列 |

## 7. 诚实边界

- `deepseek-v4-pro` 公开定价未知：三档价格为参考场景（V3/reasoner/hi-end，cache-miss 口径），真实账单口径未获得——论文措辞用 "under reference pricing scenarios"。
- DeepSeek 时延含网关/网络/时段方差：测量窗口为逐调用 `ts_iso`（首/末见调用记录），非受控硬件基准。
- DeepSeek 实测失败率 13.0%（13/100：SSL 断连/超时/empty_content，网关间歇性不稳定与项目历史记录一致）：时延与 token 统计基于成功子集，失败单独成列；按设计 §7 R1 触发条件（失败率 >10%）建议执行 N=200 扩展，**待用户追加批准后执行**。
- S9 绝对时延绑定本机 CPU（Intel Core Ultra 9 275HX，torch 2.13.0+cpu，无 GPU）；论文报告数量级与比值，绝对值附硬件说明。
- 共享输入构建（可达性规划/备选项）单独计时（T4 输入列），不计入任何一方决策时延；两决策者同口径。
- DeepSeek 采用 K=1 直接部署口径（单次调用噪声 L1=0.117 为历史审计值，如实注明）；K=3 标签口径成本 3×，未执行（设计 R2）。
- 一次性蒸馏投入（1,502 次调用，真实 token）非零，T3 参考行如实给出；不声称零成本蒸馏。
- DeepSeek 的 10k/100k 行是线性投影（假设无 rate limit、时延稳定），S9 的 10k/100k 是实测——"projection" 与 "measured" 已分别标注。
- E2 只测效率，不涉及决策质量（E1 与主实验覆盖）。
- 原冻结 Phase C record 的 `checkpoint` 字段指向 s8 路径属 `run_phase_c.py` 的硬编码元数据缺陷（2026-08-29 已修复：默认 checkpoint 改为 S9、record 字段由实际 checkpoint 派生，6 份冻结 `phase_c_result.json` 的 provenance 已回写为 S9 SHA256 `6af79b44bc699c00…`）；G2 以 S9 checkpoint 复建逐位一致，证明冻结决策实际由 S9 生成。

## 8. No-Fabrication 状态

- 本报告所有数字由 `scripts/make_e2_report.py` 从证据 JSON 填入；价格场景与投影均已标注。
- 数字一致性建议由 `ccf-integrity-auditor` 复核；出版级图表交由 `ccf-visual-composer`（Fig-E2a/E2b 三态标注）。
