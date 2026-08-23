# 实验完成时间与 API 成本预估

> 更新：2026-08-20 · 依据：真实已生成数据（192 states / 576 calls）的实测统计 + 网关实测延迟
> 脚本：`scripts/estimate_dataset_cost.py`（离线估算）、`scripts/probe_api_cost.py`（实时探测）

## 1. 实测基础数据

| 项目 | 数值 | 来源 |
|---|---|---|
| 单次调用输入 | ~752 tokens（state JSON ~1609 字符 + system prompt） | 192 个真实 state 的平均长度 |
| 单次调用输出 | ~71 tokens（action JSON ~257 字符） | 576 条真实回复的平均长度 |
| 单次调用延迟 | 23–90 s，均值 ~54 s（近期网关波动时更长） | repeat_records.elapsed_seconds |
| 有效吞吐 | ~5–6 次/分钟（4 workers） | 192 states 用时 ~1h53m |
| 网关近期状态 | 出现间歇性 502 / 120s 超时（收尾段 11/206 次硬失败） | 生成日志 + `probe_api_cost.py` 两次 502 |

## 2. 已花费的 API 成本

576 次有效调用 + ~40 次失败重试 ≈ 616 次真实请求 ≈ 0.46M input + 0.044M output tokens。

| 价格场景（USD/M tokens） | 单次 | 已花费总计 |
|---|---|---|
| v3 级：$0.28 in / $0.42 out | $0.00024 | **≈ $0.15（≈ ¥1.1）** |
| reasoner 级：$0.55 in / $2.19 out | $0.00057 | ≈ $0.35（≈ ¥2.5） |
| 高端：$2 in / $8 out | $0.00208 | ≈ $1.28（≈ ¥9.2） |

> ⚠️ `deepseek-v4-pro` 走的是企业内部网关（corp-ai），公开定价未知；以上是**参考价格场景**，
> 精确账单以 LiteLLM 网关后台为准。就量级而言：**API 成本不是瓶颈**（人民币个位数）。

## 3. 剩余工作的时间与成本预估

### 3.1 数据集扩展（唯一花 API 的部分）

扰动轴状态数（每 persona×trip 对）：baseline 1 + weather 4 + congestion 4 + transit_delay 3
+ fare 3 + parking 3 + road_disruption 1 = **19 states**。

| 方案 | 规模 | states | 调用数 | 时间（4 workers，网关正常） | 时间（网关波动 +50%） | 成本（v3级 → 高端） |
|---|---|---|---|---|---|---|
| B. 最小够用（推荐下一步）：20 personas × 3 trips × {weather, fare} | 60 对 × 8 states | 480 | 1440 | ~4.5 h | ~7 h | $0.35 → $3.0 |
| A. 开发验证完整：24 personas × 4 trips × {weather, fare, congestion, delay} | 96 对 × 15 states | 1440 | 4320 | ~13 h | ~20 h | $1.0 → $9.0 |
| C. Phase 0–7 全轴：20 personas × 3 trips × 全部 6 轴 | 60 对 × 19 states | 1140 | 3420 | ~10.5 h | ~16 h | $0.8 → $7.1 |

- 8 workers 若网关允许，时间约减半。
- 支持 `--resume`，可分批跑；`--axes` 可按预算切子集。
- 方案 B 即足够做 **persona-holdout**（20 personas → 14/3/3 切分）检验未见人群泛化。
- K=3 保持不变（审计结论 B1：单次调用噪声 L1=0.117，K=3 已把噪声压到 0.047）。

### 3.2 训练与评估（无 API 成本）

- 单个 Student 训练：CPU 上 3–6 秒（实测 v0.2-A 3.5s / B 5.2s / C 4.5s）。
- A/B/C 三方训练 + 对比报告：分钟级。
- persona-holdout 切分 + 评估代码：已有 `group_field=persona_group_id` 支持，剩 ~1–2 小时开发。

### 3.3 Phase 8–10：MATSim（0 API 成本，纯工程）

| 阶段 | 内容 | 时间预估 |
|---|---|---|
| Phase 8 | MATSimAdapter（Student 输出 → plans/attributes） | 2–5 天 |
| Phase 9 | Population-scale 模拟（公开场景 + 蒸馏 agents） | 1–3 天（+模拟运行，每次数分钟–数小时 CPU） |
| Phase 10 | 网络反馈动态闭环 | 2–5 天 |

MATSim 部分**没有 API 开销**（Student 是 24k 参数 CPU MLP，群体规模推理近乎免费），
是"整个研究蓝图完成"里真正花时间的部分，按兼职节奏约 **1–3 周**。

## 4. 总体结论

1. **API 钱不是问题**：全实验（含所有扩展方案）在参考价格下最多 ~$10（几十元人民币），
   已花费部分约 ¥1–9。真正要花的是**墙钟时间**（网关每次调用 30–90 s）。
2. **完成 Phase 0–7 开发验证**：约 **1–2 天**（方案 B 数据 4.5–7h + 训练/评估分钟级 + 报告 2–3h）。
3. **完成整个研究蓝图（含 MATSim 闭环）**：约 **2–4 周**，瓶颈是 Phase 8–10 的工程，不是 API。
4. **当前风险**：网关正在间歇性 502/超时（刚探测两次均 502）。启动下一轮数据生成前，
   建议先确认网关恢复；生成脚本已支持 `--resume`，中断无损失。
5. **改进已落地**：生成脚本现在会把每次调用的 `usage`（真实 token 数）写入
   `repeat_records.jsonl`，后续运行可用 `estimate_dataset_cost.py` 得到精确（而非估算）成本。
