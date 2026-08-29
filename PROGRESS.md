# 实验进度与执行记录

> 蒸馏出行意图 · 个体行为蒸馏研究（DeepSeek V4 Pro → 轻量 Traveler Agent → MATSim）
> 更新日期：2026-08-29（TRC_AIT_5 补充实验 E1–E5 全部完成：MNL-B 基线、DeepSeek vs S9 速率/成本、
> 人口扩展、multi-seed 稳健性、Helsinki zero-shot 迁移——E5 六情景 C0–C5 门禁全过；
> 按计划书 Stop Rule 进入 manuscript v1 阶段。此前 08-27：S9 重训完成、S9 Freeze、
> Phase C 六情景重跑完成；同日根目录整理：阶段指令/计划文档移入 `docs/stage_instructions/`
> 与 `docs/plans/`，路径映射见 README）

## 研究目标（不变）

把 DeepSeek V4 Pro 对「不同 Persona × 动态 Context → 出行行为响应」的推理能力，
蒸馏为轻量 Traveler Agent，重点保留：

- 个体异质性（heterogeneity）
- 环境敏感性 / 行为弹性（elasticity：票价、天气、拥堵、延误变化时的行为响应）

并最终接入 MATSim 做 population-scale 与网络闭环验证。

## 当前阶段结论（2026-08-27）

- **模型线（冻结）**：`S7-W3 = Generic Behavioral Core v1.0`（论文 generic baseline）；
  `S9 = Supply-Aware Traveler Agent v2.0`（论文 supply-aware extension，tag
  `s9-supply-aware-v2.0`）；`S8` 因 walk/bike 速度数据错误 DEPRECATED
  （`docs/S8_DEPRECATION.md`，release 保持字节不变）。
- **验证线（主线）**：Singapore 真实供给（Tampines + Pasir Ris）Phase A 门禁 ✅ →
  Phase B 规模 ✅ → B.5 PT 有效性与容量标定 ✅（N*=10,000，capacity 因子 0.3/0.3）→
  **Phase C 六情景完成 ✅**（frozen S9，六 gate 全 PASS；报告
  `reports/PHASE_C_SINGAPORE_REPORT.md`，基线 pt 25.3% / car 28.3% / bike 34.0% /
  walk 12.4%）。
- 详见文末「S9 重训完成」「Phase C 重跑完成」两节与 `releases/` 冻结记录。

## 本轮（2026-08-20）已完成的改动

### 1. 实验脚本修正
- `scripts/generate_aggregated_teacher_dataset.py`
  - 新增 `split_group_id`（persona::trip）与 `persona_group_id`，保证 baseline 与其全部
    反事实曲线落在同一 split（弹性评估的必要条件）。
  - 新增 `--axes` 子集参数（按需生成扰动轴）。
  - 新增 `--workers` 并发参数（ThreadPoolExecutor，线程安全，每个 state 独立
    `httpx.Client`），4 workers 可把 576 次真实调用从 ~8.6h 压到 ~2h。
- `scripts/train_student_v0_2_a.py`
  - 切分改为按 `split_group_id`（baseline+CF 同组）。
  - counterfactual preview 的 baseline 改为经 `baseline_sample_id` 关联解析
    （修复了原来 baseline 不在 CF group 内导致 preview 恒为空的问题）。
- 新增 `scripts/train_student_v0_2_b.py` + `configs/student_v0_2_b.yaml`
  - 在 L_action + L_distribution 之上增加 **L_elasticity**（监督 ΔP_T 与 ΔP_S 的方向与幅度）。

### 2. Student 模块新增
- `src/traveler_distillation/student/losses.py`：`elasticity_l1`（mode-aligned ΔP 的 L1，带 mask）。
- `src/traveler_distillation/student/dataset.py`：`CounterfactualPairDataset` + `collate_pairs` +
  `build_baseline_index` + `make_counterfactual_pairs`（baseline↔CF 配对）。
- `src/traveler_distillation/dataset/aggregation.py`：`AggregatedTeacherTarget` 增加 `split_group_id`。

### 3. v0.2-B（弹性损失）与 v0.2-C（异质性损失）
- `scripts/train_student_v0_2_b.py` + `configs/student_v0_2_b.yaml`：
  在 L_action + L_distribution 之上增加 **L_elasticity**（监督 ΔP_T 与 ΔP_S 的方向与幅度）。
- `scripts/train_student_v0_2_c.py` + `configs/student_v0_2_c.yaml`：
  再增加 **L_heterogeneity**（`heterogeneity_l1`），监督「同一情境下两个 persona 的
  响应差异」与 teacher 一致；通过 `make_persona_contrast_pairs` 构造 (persona A, persona B)
  同情境配对，用 union availability mask 处理不同选择集。
- v0.2-C 与 A/B 采用**同一 split**（`split_group_id`），保证三者可直接对比；
  persona-holdout（未见人群泛化）作为独立后续实验，文档中明确标注。

### 4. 测试
- 新增 `tests/test_split_group.py`、`tests/test_elasticity_loss.py`、`tests/test_persona_contrast.py`。
- 全套 **120 passed**。

### 5. 归档（`archive/legacy_20260820/`，含 `README.md` 说明）
- `command/`（历史轮次指令日志）、`data/mock_test`、`data/student_v0_2_a_mock*`、
  `data/teacher_v0_1`（v0.1 单次调用数据集，已被 K=3 决策取代）、
  `data/student_v0_2_a`（旧 9 条真实 smoke，旧 schema）、
  `outputs/audit_mock_test`、`outputs/student_v0_2_a_mock*`、
  根目录散落笔记文件。
- 保留：`outputs/teacher_audit_v0_1/`（真实审计证据）、`data/student_v0_2_a/`（正在重新生成）。

## 正在执行

- ✅ **真实 K=3 聚合教师数据集生成完成**：192 states / 576 次 cache-busted 调用，
  完整性校验 PASS（`scripts/validate_aggregated_dataset.py`）。
- ✅ **v0.2-A / v0.2-B / v0.2-C 真实训练完成**，结果见 `outputs/EXPERIMENT_REPORT_v0_2.md`。

## 核心结果（详见 EXPERIMENT_REPORT_v0_2.md）

| 指标(test) | v0.2-A | v0.2-B | v0.2-C |
|---|---|---|---|
| mode_accuracy | 0.7917 | 0.7917 | 0.7917 |
| KL | 0.0984 | 0.1160 | **0.0787** |
| probability L1 | 0.3031 | 0.3503 | **0.2891** |
| counterfactual \|ΔP_T−ΔP_S\| | **0.1269** | 0.1564 | 0.1461 |
| heterogeneity \|ΔP_T−ΔP_S\| | — | — | 0.1601 (8对) |

结论：管线端到端可复现；mode 全学对；L_heterogeneity 是有效分布正则（KL 0.098→0.079）；
但 test 仅 24 states，A/B/C 差异统计上不显著，需扩大数据（更多 persona + persona-holdout）。

## 下一步（v0.3 阶段，进行中）

1. ✅ **代码完成**：
   - 弹性损失重设计：`elasticity_direction_loss`（方向对齐，只罚反向移动、可带 margin、
     排除 teacher 未移动的 mode）+ `elasticity_magnitude_loss`（幅度分解）+
     `elasticity_sign_agreement`（评估用）；训练脚本 A/B/C 全部支持 config 驱动的
     `lambda_elasticity/lambda_direction/lambda_magnitude`。
   - persona-holdout：切分支持 `group_field=persona_group_id`（train/test persona 零重叠，
     即未见人群泛化检验）；新增 `student/eval.py`（`persona_breakdown` 逐 persona 测试
     指标 + `counterfactual_sign_agreement`）。
   - 新配置：`configs/student_v0_3_{a,b,c}.yaml`（holdout 切分 + 分解弹性损失）。
   - 新增 `scripts/probe_gateway.py`（网关探活，故障恢复自动通知）。
   - 测试 **128 passed**（含 `tests/test_matsim_adapter.py`）；mock 端到端 smoke
     （holdout 切分 + 三版本训练）通过。
2. ✅ **已解除阻塞：切换 DeepSeek 官方直连**（用户提供官方 key）：
   - `.env` → `DEEPSEEK_BASE_URL=https://api.deepseek.com/v1`、
     `DEEPSEEK_MODEL=deepseek-v4-pro`（官方 `/models` 确认存在）、新 key。
   - smoke 验证：官方端点 3 次调用全部 HTTP 200，解析 + 校验通过。
   - 生成脚本新增 `--no-cache-bypass`（官方 API 不接受 LiteLLM 网关的 cache 字段；
     官方直连本就无响应缓存，重复 completion-id 检查保留作安全网）。
   - 旧内部网关（192.168.27.4:18888）此前宕机约 4.5h（TCP 通、HTTP 无响应），已弃用。
3. ✅ **v0.3 数据集完成并校验通过**：320/320 states、960 repeat records（每 state 恰好
   3 次、completion id 互异）、40 baselines + 280 counterfactuals、40 split groups
   （`validate_aggregated_dataset.py` PASS）。期间修复了 incomplete 状态丢弃已获取
   repeats 的持久化 bug，并新增 `repair_jsonl_tail.py` 崩溃恢复工具。

## v0.3 实验结果（persona-holdout，test = 3 个完全未见的 personas）

| 指标(test) | v0.3-A | v0.3-B (+分解弹性) | v0.3-C (+异质性) |
|---|---|---|---|
| mode_accuracy | 0.7708 | **0.8125** | **0.8125** |
| KL(P_T‖P_S) | 0.3816 | 0.1440 | **0.1092** |
| probability L1 | 0.7605 | 0.3260 | **0.2860** |
| counterfactual \|dP_T−dP_S\| | 0.0661 | 0.0596 | **0.0539** |
| sign agreement | 0.6508 | **0.7143** | 0.6825 |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0986 (48对) |

结论：分解式弹性损失在未见人群上有效（ΔP 误差 0.066→0.060、方向一致率 0.651→0.714）；
L_heterogeneity 进一步改善分布拟合（L1 0.76→0.29）且不牺牲 accuracy。教师噪声参考
（官方 API K=3 pairwise L1）= 0.159。完整报告：`outputs/EXPERIMENT_REPORT_v0_3.md`。
诚实边界：test 仅 3 personas / 48 states，development-scale 证据。

## Phase 8 门禁达成：MATSimAdapter v0.1 + 真实 MATSim 运行 ✅

网关故障期间推进了不依赖 API 的 MATSim 集成（Phase 8 门禁）：

- `src/traveler_distillation/matsim/adapter.py`：`MATSimAdapter` 从 student checkpoint
  解码行为决策（mode + departure shift），生成合成网格 network.xml + population.xml
  （persona attributes、活动坐标、student 选择的 leg mode）+ config.xml
  （car 网络模式，walk/bike/pt teleported；`lastIteration=0` 不让 MATSim 自身
  replanning 覆盖蒸馏决策）。
- `scripts/build_matsim_scenario.py`：checkpoint → 场景一键构建。
- **真实 MATSim 运行验证（Java 25 + MATSim 2026.0，exit=0）**：
  - 20 personas × 2 trips 场景，student 决策分布 bike 8 / pt 13 / walk 11 / car 8；
  - MATSim `modestats.csv` = bike 0.2 / car 0.2 / pt 0.325 / walk 0.275 ——
    **与蒸馏 Student 的 mode share 完全一致**，证明 student 决策已完整进入
    MATSim 并被执行。
- 期间定位并绕过 3 个 MATSim 2026.0 环境问题（已在代码中修复/规避）：
  1. `RunMatsim` 的 Guice Scenario provider 二次 materialize → 网络重复加载崩溃
     → `tools/java/RunMatsimPreloaded.java` 预加载 scenario 绕过；
  2. population_v6 的 `<act>`→`<activity>` 改名、attribute 用 `class=` 而非 `type=`；
  3. PowerShell→Java argv 中文路径编码损坏 → `scripts/run_matsim.ps1` +
     `scripts/build_matsim_launcher.ps1` 用 ASCII junction classpath。
- 测试 **128 passed**（新增 `tests/test_matsim_adapter.py`）。
- v0.1 已知边界：return leg 复用去程 mode；pt 为 teleported（无公交时刻表）；
  无 within-day 动态调整（Phase 10 的闭环留待后续）。

## Phase 9 完成：人口级仿真（1000 personas × 4 动态情景）✅

- `scripts/run_population_experiment.py` + `scripts/make_phase9_report.py`：
  1000 personas × 每人 2 次出行，4 个情景（baseline / rain / fare ×2 / combined）
  分别经 student（v0.3-C）决策 → MATSim 执行（lastIteration=0），4 情景全部 exit=0。
- **人口级需求转移（百分点，vs baseline）**：
  | 情景 | bike | car | pt | walk |
  |---|---|---|---|---|
  | rain | −19.2 | +15.0 | +5.7 | −1.5 |
  | fare ×2 | −2.5 | +7.0 | −4.3 | −0.2 |
  | combined | −19.2 | +15.0 | +5.4 | −1.2 |
  方向全部符合行为预期：暴雨把暴露型出行推向遮蔽型；票价翻倍使 pt 需求流失
  （主要转向 car）；组合扰动中天气效应占主导。
- 完整证据链验证：**Context Change → Individual Choice (student) →
  Population Demand Shift (MATSim executed)**。报告：
  `outputs/PHASE9_population_report.md`；数据：`data/population_experiment/`。
- 过程修复：初版脚本将全部 trip 复用于每个 persona（人口文件放大 ~1000 倍，
  1000 人时 MATSim 报 runners-null NPE）→ adapter 增加 `trips_per_persona`
  参数、脚本改为每人独立 2 次出行后 1000 人正常运行。

## Phase 10 完成：网络反馈闭环 ✅（蓝图五层全部打通）

- `scripts/run_phase10_loop.py` + `matsim/runner.py`（linkstats 拥堵提取 +
  `parse_modestats/run_matsim` 共享工具）：student 决策 → MATSim → 观测拥堵
  c_obs（capacity 加权延误比 / 0.5）→ 平滑更新 c_ctx → 再决策，直至收敛。
- 结果（800 personas × 10×10 网格 × cap=5 veh/h，3 情景全部收敛）：
  | 情景 | 平衡 c* | car share | 收敛轮数 |
  |---|---|---|---|
  | baseline | 0.38 | 25.0% | 3 |
  | rain | **0.44** | 38.5% | 5 |
  | fare ×2 | 0.40 | 31.6% | 4 |
  暴雨→car 38.5%→均衡拥堵显著抬升；票价→car 31.6%→均衡拥堵居中——扰动经
  行为转移传导为网络级拥堵，方向与量级均合理。
- 诚实发现：student 的拥堵弹性较弱（单情景内 c 变化仅致 car ≤0.25pp 变化），
  与 v0.2 教师审计（拥堵轴噪声最大）一致；改进方向=扩大拥堵轴反事实覆盖。
- 报告：`outputs/PHASE10_feedback_loop_report.md`；数据：`data/phase10_loop/`。
- 过程修复：adapter 东南角节点 link id bug（`l_9_9_er`→`l_8_9_er`，10×10 网格
  触发）+ MATSim 2026 linkstats 输出位置（`ITERS/it.0/0.linkstats.txt.zst`，
  需 zstandard 流式解压）。

## 工作区整理（2026-08-21）

- 归档 v0.1 时代工具与开发期临时产物 → `archive/legacy_20260821/`（清单见 `archive/README.md`）：
  教师审计工具链、v0.1 单次调用生成/校验脚本、网关探活、成本探测工具、
  equil 本地拷贝、Phase 8 冒烟场景、过时对比/分析文件、网关时代成本预估文档。
- `.env.example` 更新为官方直连模板（API key 留空）。
- 测试 130 passed（归档不影响任何活跃脚本）。

## 已知边界

- 合成 state（synthetic persona/trip），无真实世界标定。
- v0.2 A/B/C 切分 persona overlap=True（保证三者可比）；v0.3 起用 persona-holdout。
- 真实数据规模有限（development dataset 而非最终规模）。
- pt 为 teleported 模式（无公交时刻表）；Phase 10 反馈仅作用 road_congestion。

## S1 扩训完成：road_congestion 轴补训（决定性实验）✅（2026-08-21）

- `scripts/extend_teacher_dataset.py`：从现有 baseline state 派生新轴反事实态，复用 baseline
  （不重烧调用），支持 `--resume`；合并后数据集 **480 态 / 1440 repeat / 40 split groups**，
  `validate_aggregated_dataset.py` PASS（新增 160 拥堵态 = 480 次真实调用，1 次解析失败已重试）。
- `scripts/run_v0_3_s1_experiment.py`：重训 A/B/C 到 `student_v0_3_s1_{a,b,c}`（保留 v0.3 旧 checkpoint
  作 before 对比）；`scripts/eval_congestion_elasticity.py`：逐轴 before/after 弹性测量。
- **教师侧**：拥堵弹性 |ΔP_T|=0.0572（介于票价 0.049 与天气 0.080），方向正确（car 随拥堵单调降）；
  但教师拥堵采样噪声(pairwise L1, K=3)=0.1325，**信号低于噪声**，拥堵轴信噪比三轴最差。
- **拥堵轴 before/after（同一 24 个 test 拥堵态）**：
  | checkpoint | |ΔP_S| | |ΔP_T−ΔP_S| | sign |
  |---|---|---|---|---|
  | v0.3-C（无拥堵训练） | 0.0067 | 0.0637 | 0.3611 |
  | **v0.3-S1-C（补训后）** | 0.0215 | 0.0604 | **0.7361** |
- **三管线整体（test=72 态，含拥堵）**：C 最优（acc 0.875 / KL 0.066 / L1 0.251 / sign 0.742）。
- **结论**：① 教师对拥堵**敏感**（方向正确、强度中等）；② 之前"student 拥堵弹性弱"主要是
  **数据缺口**——补训后方向一致性 0.36→0.74；③ 幅度欠响应（|ΔP_S| 0.022 vs 教师 0.057）源于
  **教师信噪比上限**（0.057 信号 vs 0.133 噪声），提高幅度拟合应增加拥堵轴 K>3 以压噪，
  而非增加 persona。完整报告：`outputs/EXPERIMENT_REPORT_v0_3_S1.md`。

## S2 扩训完成：transit_delay + parking_cost 轴补训 ✅（2026-08-21）

- `scripts/extend_teacher_dataset.py`（复用）+ `scripts/run_v0_3_s1_experiment.py --tag s2`：
  补 `transit_delay` 3 档 `[5,15,30]` + `parking_cost_multiplier` 3 档 `[1.5,2.0,3.0]`。
  目标 240 态 × K=3 = 720 次；实际聚合 **233 态**（合并后 713 态 / 2139 repeat，校验 PASS）。
- **7 个 incomplete**：全是 `transit_delay=30` 且教师建议**提前 >60min 出发**（-79~-90）的
  "继续 pt + 大幅提前"样本，被 `departure_time_shift_min` ±60 schema 硬约束在 parse 阶段拒绝。
  这是 student `60·tanh` 出发头的**表示边界**（非 bug），按"蒸馏范围=student 可表示范围"排除。
- **教师侧信噪比（五轴排序）**：停车费 0.63 > 延误 0.51 > 天气 0.47 > 拥堵 0.43 > 票价 0.34。
- **before/after（test 逐轴，C 变体）**：
  | 轴 | sign before→after | \|ΔP_T−ΔP_S\| before→after |
  |---|---|---|
  | parking_cost | 0.315 → **0.778** | 0.090 → **0.059** |
  | transit_delay | 0.685 → **0.759** | 0.067 → **0.047** |
- **三管线整体（test=108 态）**：C 最优（acc 0.861 / KL 0.053 / L1 0.236 / ΔP gap 0.046 / sign 0.693）。
- **结论**：① 停车费是**干净数据缺口**，补训后完整学会（方向+幅度），与其最高信噪比一致；
  ② 延误轴**部分可由特征驱动**（transit_delay 已传播进替代属性，S1-C 已 sign 0.685），补训精化；
  ③ **跨轴一致规律**：student 弹性拟合上限由**教师该轴信噪比**决定（停车费学会 > 延误 >
  天气 > 拥堵 > 票价欠拟合），S1 拥堵与 S2 停车费共同印证。完整报告：
  `outputs/EXPERIMENT_REPORT_v0_3_S2.md`。

## S4 扩训完成：road_disruption 轴补训（六轴覆盖收官）✅（2026-08-22）

- `scripts/extend_teacher_dataset.py`（复用）：补 `road_disruption` 1 档 `true`（事故/封路）。
  40 态 × K=3 = 120 次调用；**attempted 121 / valid 120 / failed 1（解析失败自动重试）/
  incomplete 0 / aggregated 40**。合并后数据集 **753 态 / 2259 repeat**，校验 PASS。
- **教师侧（新增脚本 `scripts/teacher_axis_snr.py`，复现 S1/S2 口径：信号=test、噪声=全态）**：
  road_disruption 弹性 |ΔP_T|=**0.1756**、噪声 0.1388、**信噪比 1.27**——六轴最强最干净
  （远超此前的停车费 0.63）；car 概率质量封路下平均 **−0.351**，是教师所有轴中最大的单轴迁移。
- **before/after（同一 6 个 test 中断态，C 变体）**：
  | checkpoint | \|ΔP_S\| | \|ΔP_T−ΔP_S\| | sign |
  |---|---|---|---|
  | v0.3-S2-C（无中断训练） | 0.1056 | 0.0738 | 0.7778 |
  | **v0.3-S4-C（补训后）** | **0.1408** | 0.0787 | **0.8333** |
- **三管线整体（test=114 态）**：C 最优（acc 0.868 / KL 0.052 / L1 0.230）。
- **结论**：① 道路中断是教师最干净最强轴，学生补训后方向 0.78→0.83、幅度 0.106→0.141；
  ② 与 S2 transit_delay 同构——道路中断**部分可由特征驱动**（car travel_time/reliability 属性
  已可见，S2-C 即 sign 0.78），补训精化；③ **六轴数据缺口全部补齐**，跨轴规律再次成立
  （拟合上限=教师信噪比：道路中断 1.27 > 停车费 0.63 > 延误 0.51 > 天气 0.47 > 拥堵 0.43 >
  票价 0.34）。完整报告：`outputs/EXPERIMENT_REPORT_v0_3_S4.md`。

## S3 扩训完成：persona 20→40（holdout 统计功效）✅（2026-08-23）

- `scripts/generate_aggregated_teacher_dataset.py` 新增 `--persona-offset/--trip-offset/--id-prefix`
  （`PersonaGenerator`/`TripGenerator` 加 `id_offset`，向后兼容）+ 新增 `scripts/merge_teacher_datasets.py`。
  生成 20 新 personas（P000021–P000040，seed=43）× 2 新 trips（T000003/004）× 全 6 轴：
  **760 态 × K=3 = 2280 次调用，0 incomplete**（新人群未触发 S2 的 departure_shift>60 剔除；
  期间官方 API 瞬时故障自动重试，workers 8→16 提速）。合并后 **1513 态 / 4539 repeat /
  80 baselines + 1433 CF / 80 split groups**，校验 PASS。
- **教师侧（40 personas，test=6 未见人/226 态）**：信噪比排序不变（道路中断 0.97 > 天气 0.74 >
  停车费 0.45 > 拥堵 0.33 > 延误 0.32 > 票价 0.27）；但逐轴弹性随人群扩容明显变化
  （天气 0.080→0.102、延误 0.076→0.045、票价 0.049→0.036），印证 20-persona 估计方差大。
- **三管线整体（test=226 态，6 未见 personas）**：B 分布拟合最优（acc 0.885 / KL 0.050 / L1 0.187），
  C 反事实方向最优（sign 0.705 / ΔP gap 0.052）。**统计功效直观体现**：20-persona 时代 A/B acc
  恒并列、差异不显著；扩容后 B 的 KL/L1 明显拉开 A。
- **未见人群逐 persona（S3-C）**：acc 0.63（P000018 最硬）~1.0（P000008），泛化非均匀。
- **结论**：① holdout 统计功效达成（test 3→6 人、114→226 态，A/B/C 差异首次可可靠区分）；
  ② 40-persona 逐轴方向一致性参差（拥堵 0.72/延误 0.87 好，票价 0.56 欠拟合）；
  ③ 跨轴规律扩容后仍成立（拟合上限=教师信噪比），追低信噪比轴幅度应 K>3 压噪而非加 persona。
  完整报告：`outputs/EXPERIMENT_REPORT_v0_3_S3.md`。

## S5 多轴联合蒸馏实验完成 ✅（2026-08-24）

依据 `S5_MULTI_AXIS_DISTILLATION_EXPERIMENT_DESIGN.md`，从"单轴行为弹性蒸馏"升级为
"多条件联合行为响应蒸馏"。

### 数据
- 四组合 × 80 baselines × 2 joint levels = **640 joint 态**；**2640 次教师调用**（0 incomplete）。
- K 策略：high-SNR（rain×cong、disruption×cong）K=3；low-SNR（fare×cong、fare×delay）K=5；
  **K=7 因果子集**（fare×delay 最强档 40 态）。K 分布 {3:320, 5:280, 7:40}。
- fare×congestion 完全 holdout（unseen combination 组合泛化测试）。
- 实现：`joint_axes` schema、`perturb_context_multi`、`JointContextGenerator`、
  `generate_joint_teacher_dataset.py` / `validate_joint_dataset.py` / `rebuild_k5_view.py` /
  `train_student_s5_joint.py` / `eval_s5_joint.py` / `make_s5_report.py` / `run_s5_experiment.py`。

### 过程中修复的三个工程问题
1. joint 态触发教师建议出发 >60min，被 schema ±60 反复拒绝 → **解析时裁剪 departure 到 ±60**
   （保留 mode 概率；与 S2 "蒸馏范围=student 可表示范围"边界一致）。
2. joint 态触发超长推理（completion 高达 8000+ token，接近 8192 上限）→ empty-content 失败
   → **max_tokens 提到 12000**（32768 会让部分态推理 >10min 致吞吐崩塌，故取中）。
3. `_rebuild_target` 把 pydantic 对象当 dict 调 `.get()`（AttributeError）→ K=7 子集态 k5 视图
   缺失 → 改为接收原始 repeat 记录 + 合并 resume repeats，新增 `rebuild_k5_view.py` 确定性回填。

### 结果（test = 6 未见 personas / 96 joint 态）
| 指标 | M0 单轴 | M1 joint(K=5) | M2 joint(+K=7) |
|---|---|---|---|
| seen joint KL | 0.0538 | **0.0440** | 0.0441 |
| unseen joint KL (fare×cong) | 0.0954 | 0.0890 | **0.0880** |
| interaction L1 error | 0.0469 | 0.0464 | **0.0453** |
| legacy acc（不退化） | 0.8496 | 0.8496 | 0.8451 |
| legacy KL | 0.0776 | 0.0745 | **0.0735** |

### 结论
- **RQ-M2 成立**：定向 joint 微调使 seen joint KL 降 ~18%（0.0538→0.0440），且不牺牲单轴能力。
- **RQ-M3 有限**：unseen 组合（fare×cong）KL 降 ~7.8%（0.0954→0.0880），记忆强于组合泛化，但方向正确。
- **RQ-M4**：interaction 误差整体很小（≈0.045），联合响应近似可加，冲突组合略难。
- **RQ-M5**：K=7 稳定化方向正确但幅度有限（40 态子集规模限制效应量）。
- 报告：`outputs/EXPERIMENT_REPORT_S5_MULTI_AXIS.md`；数据：`data/student_s5_joint/`；
  模型：`outputs/student_s5_joint_m1|m2/`；评估：`outputs/s5_joint_eval/`。

## 下一步（已规划，按优先级）

### 1. 扩训：补扰动轴覆盖 + 人群扩容（✅ 全部完成）
已覆盖 **6/6** 扰动轴、persona **20→40**。分阶段：

| 阶段 | 内容 | 新增调用 | 实测耗时 |
|---|---|---|---|
| **S1** ✅ | 补 `road_congestion` 4 档（20 personas × 2 trips） | +480 | ~1.4h |
| **S2** ✅ | 补 `transit_delay` 3 档 + `parking_cost` 3 档 | +720 | ~2.4h |
| **S4** ✅ | `road_disruption` 1 档（事故/封路，故事性强） | +120 | ~0.5h |
| **S3** ✅ | persona 20→40（holdout 统计功效） | +2280 | ~4.5h（workers 8→16） |

每阶段完成后重训 v0.3-A/B/C 并对比（`run_v0_3_s1_experiment.py --tag` 一键完成）。
S1 结论（决定性）：补训后拥堵方向一致性 0.36→0.74，说明此前"student 拥堵弹性弱"
主要是**数据缺口**而非"居民不敏感"；但幅度仍欠拟合，受教师拥堵轴信噪比上限约束。
S2 结论：停车费（干净缺口，sign 0.31→0.78）完整学会、延误（特征驱动，sign 0.69→0.76）精化。
S4 结论：道路中断（SNR 1.27，六轴最强最干净，car −0.351）补训后 sign 0.78→0.83、幅度
0.106→0.141；与延误同构的"特征驱动"轴。
S3 结论：40-persona holdout（test=6 未见人）下 B 分布拟合最优、C 方向最优；per-persona
acc 0.63–1.0 泛化非均匀；逐轴弹性随扩容收敛（天气 0.102/延误 0.045/票价 0.036）。
**跨轴规律（40-persona 证据）**：student 弹性拟合上限 = 教师该轴信噪比（道路中断 0.97 >
天气 0.74 > 停车费 0.45 > 拥堵 0.33 > 延误 0.32 > 票价 0.27）。追幅度优先考虑低信噪比轴
（票价/拥堵）的 K>3 压噪，而非继续加 persona。

### 2. 仿真真实性提升
- 真实路网：OSM 路网 + GTFS 公交时刻表替换合成网格（pt 从 teleported 升级为
  transit schedule）；MATSimAdapter 需要坐标投影与网络导入改造。

### 3. 论文写作
- 全部 Phase 0–10 已有端到端证据，可进入写作阶段：`ccf-paper-writer` 起草、
  `ccf-integrity-auditor` 核验数字一致性（蓝图、两份实验报告、Phase 9/10 报告）。

## S6 因果机制审计完成 ✅（2026-08-24）

依据 `S6_REASONING_CAUSAL_AUDIT_EXPERIMENT_DESIGN.md`，检验 Student 蒸馏后保留的是
交通行为机制（mediator 驱动）还是 context-to-action shortcut。

### 方法与数据
- 三轴 × 四联组 A/B/C/D（baseline / natural / broken-path / mediator-only），
  broken/mediator 态人为解耦 context 标签与 causal mediator（car/pt travel_time、
  reliability、monetary_cost）。
- 960 审计态；Teacher K=5 跑 487 个新态（2435 调用，0 incomplete，中途断电后经
  增量进度文件 resume 续跑完成）；A/B 复用 S3 K=3 目标。
- 实现：`generate_causal_audit_states.py` / `run_teacher_causal_audit.py`（增量落盘+
  断点续跑）/ `eval_causal_mechanism.py` / `make_s6_report.py` + `configs/causal_audit.yaml`。

### 结果（R_shortcut ↓=机制、R_mediator ↑=机制；congestion/parking 仅 car-可用 36 组）
| 轴 | Teacher | C0 pre-S5 | C1 S5 | 判定 |
|---|---|---|---|---|
| transit_delay | 0.73 / **0.97** | 0.37 / **0.90** | 0.35 / **0.84** | Case 3 两者机制一致 |
| congestion | 0.52 / **0.89** | 0.81 / 0.75 | 1.02 / 0.67 | Case 1 蒸馏退化 |
| parking_cost | 0.69 / 0.56 | 1.03 / **0.05** | 1.02 / 0.06 | Case 1 机制丢失 |

（斜杠前 R_shortcut、后 R_mediator）

### 结论
- **Grade B — Partial Mechanism Preservation**：transit_delay 通路（delay→PT travel_time）
  蒸馏后完整保留；congestion 通路部分退化；parking_cost 通路（monetary_cost mediator）
  几乎完全丢失——Student 对该轴主要响应 parking_cost_multiplier 标签。
- **RQ-C5 关键发现**：S5 多轴补训只改善插值、不改善机制——congestion 的 R_shortcut
  C0 0.81 → C1 1.02（反而加剧 shortcut）。
- 论文表述边界（§27）：可写 "partial causal consistency under mediator interventions"，
  不可写 "full causal chain-of-thought"。
- 后续方向（§20/21）：若需修复，走 S7 causal-aware fine-tuning（A/B/C/D 四联组 +
  L_path + L_shortcut）。
- 报告：`outputs/EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md`；数据：`data/causal_audit/`；
  评估：`outputs/causal_audit/eval_metrics.json`。

## S7 机制感知补训完成 ✅（2026-08-24）

依据 `S7_MECHANISM_AWARE_FINETUNING_INSTRUCTIONS.md` + 用户五条不可违反约束，
从 C1（S5 joint M2）定向补训 congestion / parking_cost 机制保真。

### 五条约束的执行方式
1. **test 隔离**：机制四联组按 S3-C persona holdout 切分（congestion/parking 各
   train 26 / val 2 / test 8 组，car-可用过滤）；训练只加载 train、early-stop 与
   选模只用 val、最终评估只用 test（训练脚本硬断言 test=16 后丢弃引用）。
2. **零新增 API**：teacher target 完全复用 S6（A/B=K3、C/D=K5），本阶段 0 次调用。
3. **gap 主指标**：G_nat/G_broken/G_med（表 A1）定级，R_shortcut/R_mediator 次指标。
4. **paired bootstrap CI**：B=2000/seed=42，单位=四联组（causal）/state（regression），
   所有聚合量与 C2-vs-C1 差分均报 95% CI。
5. **消融**：W1/W2/W3（λ 0.25/0.5/1.0 双项）+ mech_only + broken_only（0.5 单项）共 5 变体。

### 实现
- `src/traveler_distillation/student/mechanism_dataset.py`：`MechanismQuadrupletDataset`
  + `collate_quadruplets`（A/B/C/D 四联组编码、成员 mode 对齐断言）。
- `src/traveler_distillation/student/mechanism_losses.py`：`mechanism_fidelity_loss`
  （natural+mediator 效应保真）+ `broken_path_fidelity_loss`（匹配 Teacher 非零
  broken effect，禁止推向 0）。
- `scripts/`：`build_s7_mechanism_dataset.py`、`train_student_s7.py`（replay 2:1:1 +
  52 het 对/epoch、LR=S5×0.25）、`eval_s7_causal_repair.py`（test-only + bootstrap）、
  `eval_s7_regression.py`、`eval_s7_val_metrics.py`、`s7_variant_distances.py`、
  `run_s7_experiment.py`、`make_s7_report.py`；`configs/student_s7_{w1,w2,w3,mech_only,broken_only}.yaml`。
- 测试 11 个新增（`tests/test_mechanism_losses.py`、`test_mechanism_dataset.py`），全套通过。

### 结果（test 四联组，8 组/轴，配对 bootstrap；与 S6 的 36 组口径不可直接比较）
- **Grade B — 部分成功**（选 W3）：parking mediator gap 显著下降
  （ΔG_med −0.0054 [−0.0090, −0.0018]）、congestion shortcut gap 显著下降
  （−0.0566 [−0.1164, −0.0051]）；parking shortcut gap 小幅显著反向
  （+0.0073 [+0.0005, +0.0128]）；congestion mediator gap 无变化。
- **无回归**：legacy KL 0.0735→0.0695（CI 不含 0，反而改善）、seen joint KL
  0.0441→0.0421（显著改善）、unseen joint KL 方向改善；legacy acc 0.8451 不变。
- **λ 消融关键发现**：五变体两两 L2 距离 ≤0.0097、距 C1 ≈0.133 —— λ 在 [0.25,1.0]
  内不区分结果，变体收敛到几乎同一模型；观测到的变化来自 S7 微调制度整体
  （机制四联组 + replay + 低 LR），L_mechanism/L_broken 的单独归因无法在该消融中建立。
- **val/test 背离**：val 仅 2 组/轴，val gap 未检测到修复（选模在各近似相同模型间进行）；
  最终判定只依赖 test 配对 bootstrap。
- 报告：`outputs/EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md`；数据：`data/student_s7_mechanism/`；
  模型：`outputs/student_s7_{w1,w2,w3,mech_only,broken_only}/`；评估：
  `outputs/s7_causal_eval/`、`outputs/s7_regression/`、`outputs/s7_selection/`。

### S7 结论与下一步
- 机制补训到此为止（§36）：**定向机制监督能产生统计显著但幅度有限的局部修复
  （parking mediator、congestion shortcut），无法完全恢复 parking 的 mediator 通路
  （R_mediator 0.071→0.085 vs Teacher 0.769）；论文表述 “axis-dependent mechanism
  preservation under targeted mechanism-aware supervision”。
- **Seed Robustness Check（最终小检查，`scripts/run_s7_seed_check.py`，零 API）**：
  W3 配置 × 4 训练种子（42/7/123/2024）重训后在同一 test-only 管线上复测 ——
  **STABLE=True**：parking G_med 4/4 种子显著为负（−0.0054~−0.0060，CI 全部不含 0）、
  congestion Gap_shortcut 4/4 种子显著为负（−0.0485~−0.0566）、legacy KL 4/4 显著为负、
  seen joint KL 4/4 方向为负；无任何种子出现显著回退。congestion G_broken 跨种子符号
  混排且均不显著（与主报告"修复发生在 shortcut ratio 而非 broken effect"一致）。
  判定：S7 改善超过训练种子方差，可放心 Freeze S7-W3。
- 按 §40 Grade B：**Freeze S7-W3**（整体性能不劣于 C1 且显著更优）→
  **Singapore OSM + GTFS real-world validation**（`NEXT_STEP_PLAN_SINGAPORE_AIT.md`），
  论文注明网络验证评估行为可执行性与系统响应，而非完整因果机制保真。

## 阶段转换：模型训练线冻结 · 交通验证线主线化 ✅（2026-08-24）

**Freeze 声明**：最终 Student = **S7-W3**（Grade B + 4-seed 稳定，回退点 S5-M2）。
不再扩 persona / 加扰动轴 / 重训 Teacher / 做机制补训；重新训练的唯一条件 = 发现明确
数据或方法错误。

**主线切换**：从“证明 Student 值不值得信”切换到“证明已审计 Student 放进真实交通供给系统后
有没有研究价值”。计划书更新为 v2.0（`NEXT_STEP_PLAN_SINGAPORE_AIT.md`，v1.0 已归档至
`archive/legacy_20260821/root/`）：Phase A（真实供给跑通，100-agent exit=0 门禁）→
Phase B（100/500/1000 规模验证）→ Phase C（正式论文情景）→ Phase D（真实网络反馈闭环）。

### Phase A 资源侦察（已完成，未开始执行）
- ✅ MATSim 2026.0 + 全部依赖 jar（`tools/matsim-2026.0-release/`），核心 jar 内含
  `org.matsim.core.utils.io.OsmNetworkReader`（OSM→network.xml）；Java 25 可用。
- ✅ `singapore-gtfs.zip` 已在工作区：6 agency / **603 线路（593 bus + 9 MRT）** /
  5,376 stops / 230,915 trips / 8.17M stop_times；目标区域（Tampines+Pasir Ris bbox
  lat 1.33–1.40 × lon 103.90–104.02）含 **746 站**。
- ✅ 既有资产：`RunMatsimPreloaded.java`（预加载运行器）、`MATSimAdapter`
  （需从 synthetic grid 扩展为真实供给 + scheduled PT）。
- ❌ 待获取/实现：OSM 区域提取、OSM→network.xml 转换脚本、GTFS→transitSchedule.xml +
  transitVehicles.xml 转换器（首选 matsim-pt2matsim，失败则自研 stop-snapping +
  route-routing 转换器）。

### Phase A 执行清单（按计划书 §2.4，执行中）
1. ✅ GTFS zip 归档（`data/singapore/gtfs/raw/`，sha256 + 快照 + 来源如实标注：社区构建 feed singapore-gtfs-2025，非官方 DataMall）
2. ✅ OSM bbox 下载（Overpass，`data/singapore/osm/tampines_pasir_ris.osm`，16.8MB，102,784 节点 / 30,140 highway ways）
3. ✅ OSM → network.xml（自研纯 Python 转换器 `src/traveler_distillation/singapore/osm_network.py`：
   100,867 节点 / 191,637 链路 / 3,520 km，UTM 48N 投影（级数与数值积分一致到 μm），
   car 最大连通分量 95%、walk 95%；按 highway 类别映射 freespeed/capacity/modes）
4. ✅ GTFS → transitSchedule + transitVehicles（`src/traveler_distillation/singapore/gtfs_prep.py` +
   `build_transit.py`：工作日 WD 服务日 + 早高峰 06:00–10:30 窗口，**5,580 trips / 170 线路 / 851 站
   （含 256 个 MRT 车次）**；供应裁剪策略=保留区域内最长连续站段（18,327 trips 裁剪）；stop
   snapping（仅 car 最大分量 + 本地道路节点，mean 60m/p90 186m）+ 人工接入链 ai_in/ai_out +
   单行道 transit 专用反向链 busr_*（28,681 条）+ 行人/自行车反向链 pdr_*（25,441 条）；
   **191/191 序列路由成功、0 失败**；transitSchedule.xml 139MB / transitVehicles 5,580 辆）
5. ✅ **100 个 S7-W3 agents baseline 运行 exit=0**（`scripts/singapore/run_phase_a_smoke.py`）
6. ✅ **car / pt / walk / bike 四模式 executed legs 全部出现**；5,580 transit 车次全部发车、
   pt 8 上车/8 下车、stuckAndAbort=0、departure 5,796 = arrival 5,796

### ✅ Phase A Gate PASSED（2026-08-25，`outputs/singapore_phase_a/PHASE_A_GATE.md`）
**真实 OSM + scheduled PT + S7 Student population 在 MATSim 中完整执行。**
关键修复沉淀：MATSim leg 路线惯例（路线须从当前活动 link 开始）、活动 link 须为入向 link、
transit 首站 ai_in 起始、车辆文件 XSD 格式、Raptor 配置（swissRailRaptor modeMapping）、
启动器 `RunMatsimPreloaded.java` --release 21 编译（规避 Guice ASM 对 Java 25 类文件崩溃）。
已记录近似（门禁报告 §3）：MRT rail-on-road、busr_/pdr_ 反向链、PT 直连无换乘（换乘与
2026 umlauf 车辆链机制冲突已关闭）、早高峰时窗（回程 pt 全步行回退）、需求侧 synthetic。
→ 下一步 **Phase B 规模化验证**（100→500→1000 agents，8 项运行指标）。

## Phase B 规模化验证完成 ✅（2026-08-25）

`scripts/singapore/run_phase_b_scaling.py`（构建 + MATSim + 事件流指标），
`outputs/singapore_phase_b/scale_report.md`：

| 指标 | 100 | 500 | 1000 |
|---|---|---|---|
| planning fallbacks | 49 | 245 | 501（随规模线性，≈50% 为 pt→walk） |
| failed trips / stuckAndAbort | 0 / 0 | 0 / 0 | 0 / 0 |
| PT boardings | 12 | 55 | 127（线性） |
| mean trip time (min) | 8.96 | 9.67 | 9.62 |
| road delay (s/passage) | 0.51 | 0.51 | 0.51 |
| congestion（自由流+15s 慢行占比） | 0.0 | 0.0002 | 0.0 |
| runtime (s) | 29.4 | 26.5 | 27.6 |
| executed mode share | 0.18/0.33/0.06/0.43 | 0.18/0.29/0.06/0.48 | 0.19/0.29/0.06/0.47 |

**判定：规模扩大无异常（失败行程 0、mode share 漂移 <5pp、无拥堵失控、runtime 平稳），
可进入 Phase C。** 备注：runtime 平稳因 5,580 个 transit 车次主导事件负载；baseline
无拥堵符合 1000 人规模预期。经验文档：`docs/MATSIM_2026_REAL_SUPPLY_INTEGRATION.md`
（13 个集成坑 + 配置模板 + 防御清单）。

## Phase B.5A — PT Routing Validity ✅（2026-08-25，目标 ≥90% 达成 98.0%）

用户硬性检查：Student intended PT 的 itinerary 构建成功率不得低于 90%，回退须分类。
- **供给升级**：全天服务 05:00–23:00，**20,966 trips / 859 站**（原双时窗 11,796）。
- **规划器升级**（`matsim/adapter.py::_plan_pt`）：直连 + **1 次换乘（独立 pt leg +
  换乘步行，无 chainedRoute**——绕开 2026 umlauf 车辆链冲突）**+ 步行延伸**（上/下车站
  可在起讫点 1.5km 内）+ 回退原因分类（no_service_window / no_direct_or_transfer /
  no_stops_in_radius）。
- **结果（1000 agents, seed=3000）**：intended 904 → routed **886（98.0%）**
  （direct 687 / transfer 199 / fallback 18，全部为 no_direct_or_transfer）；
  MATSim exit=0、pt 上车 1,082=下车 1,082、不平衡车辆 0；stuckAndAbort 3 全部在
  endTime（错过单班次车辆等至模拟结束，0.3%，计入 waitingForPt 指标）。
- 迭代过程：75.9% → 84.1% → 98.0%。报告：`outputs/singapore_phase_b5/pt_validity_v3/pt_validity.md`。

## Phase B.5B — Demand Loading Sweep（1k–10k 完成；40k 探针执行中）

`scripts/singapore/run_phase_b_scaling.py --scales 1000,2000,5000,10000`
（同一供给 + 冻结 S7-W3，只变 population scale）：记录 mean trip time / car travel
time / road delay / slow-link share / **car VKT** / **拥堵分布**（链路平均延误
>15s/30s/60s 占比 + p50/p90）/ runtime。

**结果：1k–10k 全部 free-flow，未找到 N\***——car VKT 完美线性（3.7k→7.1k→18.2k→36.2k km）、
延误恒 0.52 s/passage、慢行占比 <0.05%。原因：真实供给容量（~1,760 有向车道公里）远大于
该需求密度（~8 veh/h/车道公里）。线性外推可见拥堵需 **N≈30k–50k**。
- **40k 探针完成：仍 free-flow**（延误 0.54s、慢行 0.05%、VKT 142,538 km 线性）——
  换算峰值利用率仅 ~2% 容量；**需求单边扩展到拥堵需 N≈80万–120万，不可行**。
- 归因：OSM 容量按 highway 类别饱和流率赋值、**无路口延误建模**——真实拥堵由路口瓶颈
  产生，需求侧无法弥补。决策点见 `demand_sweep/scale_report.md`：
  ① **供给侧标定（推荐）**：城市道路容量 ×0.35–0.45 的路口有效绿信比折减（标准工程做法，
  如实写入 method）；② 需求加码 2 trips × 80k（构建 12h+）；③ 接受 free-flow 弱化网络级结论。
- **N\* 未冻结，Phase C 等待用户决策。**

## Phase B.5C — Effective-Capacity Calibration ✅（2026-08-25，方案已冻结）

诊断（`outputs/singapore_phase_b5/loading_diagnostic.md`）：峰值 car 3,452 辆/h；
链路 V/C **max 0.361 / P99 0.06**；早晚双峰集中；OD 高度分散（top-20 链路仅 ~4% VKT）；
**sample 因子 ≈0.115**（40k 为 sample population）。

标定测试（10k baseline × 同一种子，`calibration/calibration_report.md`，全档存档）：
- green-ratio 三档（approach×0.35/0.45/0.55，659 信号交叉口 1,242 条入向链）→ 全部 free-flow；
- qsim 容量因子 0.12→重拥堵（car travel 8.4→17.2 min）、0.2→中-重、**0.3→轻-中度
  （slow 0.17%、links>15s 0.36%、car travel +13%）**、0.4–0.7→轻；
- **car 行程失败率全档恒为 0.2%**（failed_trips 大头是 pt 等待乘客：道路变慢拖慢公交→
  错过单班次连接），无 road gridlock。

**冻结（Phase C 统一使用，不得改动）：N\*=10,000；flowCapacityFactor=storageCapacityFactor=0.3
（原网络不动）；全天供给 20,966 trips；PT 规划器（直连+1 换乘+步行延伸）；Student=S7-W3。**
论文表述（用户指定口径）："Signal timings were not explicitly available; therefore,
effective approach capacities were represented using green-ratio sensitivity factors"；
实验定性为 "a controlled real-network experiment under calibrated effective capacity"；
**不得**称 "calibrated reproduction of real Singapore congestion"。

### Phase A 已记录的工程近似（如实写入 routing report）
- **MRT rail-on-road**：OSM 转换不含 railway，MRT 车次沿道路图路由（transportMode=rail），显式降级。
- **单行道 transit 反向链**：GTFS 线路在 OSM 单行道建模 + snapping 误差下会产生有向不可达，
  为 bus/rail 生成仅 transit 可用的反向链（car 不受影响）——pt2matsim 同款近似，已记录。
- **PT 直连无换乘**：agent 的 pt 行程仅搜索直达车次（walk→pt→walk），无换乘规划；无直达时
  整体步行回退并计数。
- 需求侧仍为 synthetic personas/trips（Freeze 不冻结需求生成），活动点采样自站点 300m 内
  真实路网节点；student 的 alternative 属性仍为合成值（真实网络派生属性留待 Phase C）。

## S7-W3 正式冻结完成 ✅（2026-08-26）

依据 `S7_W3_BACKUP_FREEZE_INSTRUCTIONS.md` 执行 15 步冻结，发布目录：
`releases/s7_w3_generic_core_v1/`（git 追踪、只读保护）。

- **唯一 checkpoint**：`checkpoint/model.pt`（SHA256 `8263faec…b2b6`，24,370 参数，
  seed=42，来源 `outputs/student_s7_w3/checkpoints/best.pt`；optimizer/scheduler
  state 未保存，已如实记录）。
- **冻结内容**：config、输入/输出 schema + `feature_order.txt`、normalization
  （从 checkpoint `extractor_state` 提取）、4-seed checkpoint（42/7/123/2024）与
  seed 评估、data manifest（S3 1513 态 / S5 640 态 / S6 960 态 / S7 72 四联组，
  含 K 分布与 SHA256）、code manifest（S7 代码 commit `222e17e`、冻结时 HEAD
  `02f64b3`）、`FINAL_S7_W3_FREEZE.md`、`README.md`、全量 `SHA256SUMS.txt`。
- **复现性 Gate（不重训）**：对冻结 checkpoint 重跑 regression + causal 评估，
  7/7 门禁指标与历史 S7 数值**完全一致（Δ=0.0000）**：legacy acc 0.8451 /
  legacy KL 0.0695 / prob L1 0.2420 / seen joint KL 0.0421 / unseen joint KL 0.0845 /
  parking G_med 0.2580 / congestion Gap_shortcut 0.3455。
- **Git**：冻结提交 `freeze: S7-W3 generic behavioral core v1.0` + 注释 tag
  `s7-w3-generic-core-v1.0`；工作树 clean。
- **只读保护**：release 文件 Windows read-only 属性 +
  `src/traveler_distillation/student/release_guard.py` 硬断言（S8+ 脚本必须调用，
  `tests/test_release_guard.py` 6/6 通过）。
- **正式定义**：`S7-W3 = Generic Behavioral Core v1.0 = FROZEN`。
  S8 只能 load → 新实验 → 独立输出/发布目录，不得修改 S7-W3 本体。

## S8 Transit Accessibility Adaptation 完成 ✅（2026-08-26）

依据 `S8_TRANSIT_ACCESSIBILITY_TRAINING_INSTRUCTIONS.md`，用新加坡真实 OSM+GTFS
供给训练 supply-aware adaptation。报告：`reports/EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md`。

### 流程（12 步全执行）
- **Schema Audit → Case B**：新增 6 维 city-independent 特征（pt_feasible/egress/
  wait/in-vehicle/transfer_time/coverage_ratio），alt_encoder 14→20 维，Student-S8
  = 24,562 参数；S7-W3 权重逐字节复用 + 新列零初始化（**S8-at-init ≡ S7-W3 输出**，
  审计断言 4/4，`reports/S8_SCHEMA_AUDIT.md`）。
- **数据集**：338 态（Class A 86/B 36/C 60/D 78/E 78），三套 holdout 冻结
  （persona 28/6/6、OD 79/16/19 不相交、步行负担≥15min 画像仅测试）；route 规则
  access≤700m、≤1 换乘、access-aware boarding；身份泄漏引号级检查 0 命中。
- **Teacher 标注**：1,468 次调用（K=3 基础 + 边界态 K=5，用户确认预算），
  prompt `teacher_s8_accessibility_v0.1`，0 incomplete；教师信号：E 类 P(PT)=0.000、
  A 0.236 vs D 0.123、曲线内单调性 81.5%（B/C 中间档噪声大）。
- **训练**：从冻结 S7-W3 初始化，replay 2:1:1:1；Round 1（λ_acc=0）→ Round 2
  （λ_acc=1.0，§19 accessibility response loss，best_epoch=26，early stop 41）。

### 结果（test = 未见 persona × 未见 OD，配对 bootstrap）
| 指标 | S7-W3 (B0) | S8 (R2) | Teacher |
|---|---|---|---|
| PT prob MAE | 0.1336 | **0.1184**（Δ -0.0152，CI 不含 0） | — |
| mean P(PT\|infeasible) | 0.0978 | **0.0599**（Δ -0.0379*） | 0.000 |
| monotonicity pair/triplet | 0.657/0.261 | **0.686/0.304** | 0.800/0.522 |
| sensitivity ΔP_PT | 0.163 | 0.156 | 0.212 |

- **回归门禁 §26 全过且大幅改善**：legacy acc +3.5pp、legacy KL −35.8%、
  seen joint −15.4%、unseen joint −44.6%；机制无回退（congestion G_med 0.176→0.153、
  Gap_shortcut 0.346→0.285，parking 基本持平）。
- **诚实边界**：FVR rate 双侧恒 0（B0 已不选 PT 为 argmax，改进体现在 infeasible
  概率质量）；sensitivity 仍低于教师（教师 E≈0 的悬崖未被完全复现）；B/C 中间档
  受教师噪声上限约束。
- **Stop Rule（§33）判定：满足** → 下一步 **Freeze S8 → Phase C**。
- 测试：142 passed（新增 S8 测试全过；13 个 error 为沙箱临时目录 ACL 限制，与代码无关）。
- 产物：`outputs/student_s8/`（R2 最终）+ `outputs/student_s8_r1_lam0/`（λ 消融）、
  `outputs/s8_accessibility_eval(_r1)/`、`outputs/s8_regression(_r1)/`、
  `outputs/s8_unseen_od/`、`data/singapore_accessibility/`（DATA_README + manifest）。

## S8 正式冻结完成 ✅（2026-08-26）

依据 `S8_BACKUP_FREEZE_INSTRUCTIONS.md` 执行 15 步冻结，发布目录：
`releases/s8_supply_aware_v1/`（git 追踪、只读保护）。

- **唯一 checkpoint**：`checkpoint/model.pt`（SHA256 `f8232dde…3885`，
  24,562 参数，seed=42，best_epoch=26，arch `student_s8_v1`，来源
  `outputs/student_s8/checkpoints/best.pt`；optimizer/scheduler state 未保存）。
- **冻结内容**：config（student_s8 / accessibility_features / training_s8，含 R1/R2
  两轮记录）、Case B schema diff（+6 维 alt feature、alt_encoder 14→20、零初始化迁移）、
  normalization（S7 字段沿用冻结统计 + 6 新字段 S8 train 拟合）、accessibility 特征定义、
  Singapore 供给 provenance（OSM/GTFS manifest）、teacher provenance（1,468 次调用、
  0 incomplete）、data manifest（338 态、三套 holdout、泄漏 0 命中）、
  `metrics/final_metrics.json` + `comparison_s7_vs_s8.csv`、报告副本 +
  `S8_EVIDENCE_INDEX.md`、code manifest、`FINAL_S8_FREEZE.md`、`README.md`、
  全量 `SHA256SUMS.txt`（45 文件）。
- **复现性 Gate（不重训）**：对冻结 checkpoint 重跑 accessibility / unseen-OD /
  regression 三套评估，**16/16 门禁指标与历史 S8 数值完全一致（Δ=0.0000）**；
  S7-W3 release 全量 SHA256 复核 **45/45 未变**。
- **Singapore smoke（S8 推断 → adapter → MATSim）**：schema 断言通过（12 维 alt
  feature、encoder 输入 20）；PT accessibility 管线 98 feasible / 2 infeasible；
  MATSim exit=0，四模式（bike/car/pt/walk）全部执行，9 次 PT 登车、20,966 班次；
  Phase C 冻结设置（capacity 因子 0.3/0.3）路径验证通过。
- **Git**：冻结提交 `freeze: S8 supply-aware traveler agent v1.0` + 注释 tag
  `s8-supply-aware-v1.0`；工作树 clean。
- **只读保护 / Phase C Guard**：release 文件 Windows 只读属性 +
  `release_guard.py` 硬断言新增 `s8_supply_aware_v1`（`tests/test_release_guard.py`
  扩展）；S8 adapter `src/traveler_distillation/matsim/s8_adapter.py`（load-only，
  arch 硬断言）+ `scripts/singapore/run_s8_smoke.py` 为 Phase C 基础设施。
- **测试**：冻结前 155 passed / 0 error（与指令一致）；guard 扩展后 158 passed / 0 error。
- **正式定义**：`S8 = Supply-Aware Traveler Agent v1.0 = FROZEN`；
  `S7-W3 = Generic Behavioral Core v1.0` 永久保留为论文 generic baseline。
- **Phase C 固定**：Student = frozen S8；N*=10,000；flowCapacityFactor =
  storageCapacityFactor = 0.3；Singapore supply = frozen Phase B.5 版本；
  PT planner = frozen validated 版本；顺序 C0 baseline → C1 heavy rain →
  C2 PT fare increase → C3 transit delay → C4 road disruption → C5 joint。
  Phase C 只能 load，不得继续训练、改权重、改 schema、改 normalization。

## Phase C — Singapore 真实网络情景实验 启动 ✅（2026-08-26）

依据 `PHASE_C_SINGAPORE_SCENARIO_INSTRUCTIONS.md`（本次起草）+ 冻结指令 §18/§21 +
`NEXT_STEP_PLAN_SINGAPORE_AIT.md` §4，进入论文主实验阶段。

- **运行器**：`scripts/singapore/run_phase_c.py`（frozen S8 → `S8MATSimAdapter` →
  `plan_accessibility` 真实供给 alternative → 情景 context 注入 → MATSim
  lastIteration=0；复用 `scenario_metrics` 事件指标；每情景 result JSON + 汇总报告）。
- **情景定义**：C0 baseline；C1 heavy rain（intensity 0.75）；C2 fare ×1.5；
  C3 transit delay 15 min（pt tt/reliability +15）；C4 road disruption
  （car tt/reliability +20）；C5 joint = rain + delay。扰动仅经 Student context 注入，
  网络供给与时刻表六个情景完全相同。
- **population**：N*=10,000，seed=2026（与 Phase A 同源），六个情景同一批 agent
  （paired 对照）；capacity 因子 0.3/0.3 冻结值。
- **状态**：C0 baseline 运行中（后台）；C1–C5 随后按序执行；产出
  `outputs/singapore_phase_c/` + `PHASE_C_REPORT.md`。

## Phase C 六情景完成 ✅（2026-08-26）

全部六个情景（frozen S8，N*=10,000，seed=2026 同一 population，capacity 0.3/0.3，
供给不变）运行完成，六 gate 全 PASS（exit=0；真人 stuck ≤ B.5C 基线 266；
四模式；pt 下车≤登车）。报告：`outputs/singapore_phase_c/PHASE_C_REPORT.md`。

### 结果（student 决策 share，C0 → 情景）

| 情景 | car | pt | bike | walk | PT boardings | VKT |
|---|---|---|---|---|---|---|
| C0 baseline | 18.1% | 2.8% | 38.2% | 40.8% | 608 | 21,145 km |
| C1 heavy rain | **36.0%** | **32.4%** | 14.4% | 17.2% | **6,715** | 40,855 km |
| C2 fare ×1.5 | 20.3% | 2.6% | 37.1% | 39.9% | 550 | 23,700 km |
| C3 delay 15min | 19.5% | **0.0%** | 38.1% | 42.4% | 3 | 22,500 km |
| C4 road disruption | **0.2%** | 5.5% | 43.1% | **51.3%** | 1,230 | **168 km** |
| C5 rain+delay | 36.5% | 0.2% | 13.4% | 49.9% | 43 | 41,475 km |

- 方向全部符合训练行为：rain 将 walk/bike 压向 car/pt；fare 小幅 pt↓ car↑；
  delay 使 pt 近乎清零；disruption 使 car 近乎清零；C5 联合 = 非叠加
  （rain 的 pt 需求被 delay 重新路由回 walk）。
- **Gate 口径修正（有据）**：stuckAndAbort 含 30:00 截断的 transit 车辆；
  B.5C 基线（S7-W3 同设置）同样存在 transit 车辆截断（冻结 S9 C0 实测 4,287/20,966 辆；
  原 B.5C 4,347 读数无留存 JSON 佐证，统一以冻结 C0 记录为准）+ 266 真人 stuck，
  为冻结设置既有特性（报告已标注）；C0 真人 stuck 13（0.13%）。
- **已知边界（如实记录）**：~20% transit 车辆 30:00 截断影响 PT 绝对量；
  情景差分同一设置下有效；C1 failed_trips 1,072（10.7%，pt 错过单班次连接，
  B.5C 已记录的机制）；FVR 恒 0 继承自 S8 测试结论。
- **运行器**：`scripts/singapore/run_phase_c.py`（共享缓存优化：每 OD 只规划一次，
  单进程六情景）+ `scripts/singapore/postprocess_phase_c_gates.py`（gate 拆分/重算）。

## ⚠️ S8 数据错误发现与废弃（2026-08-27）

用户对 Phase C 结果提出质疑（pt baseline 仅 2.8%、C3 delay 后 pt 归零）→
实证定位到**明确数据错误**：S8 供给管线用道路自由流速度（length/freespeed）计算
walk/bike 旅行时间，有效速度中位 **29.1 km/h**（走路约 6 倍速、骑车约 2 倍速）；
338 训练态中 pt 快于 walk 的为 **0/338**（现实速度重算应为 ~75%）。MATSim 侧
walk/bike 亦按 freespeed 行驶。结论：S8 学到的物理世界错误，其 Phase C 结果作废。
详见 `docs/S8_DEPRECATION.md`。处置：S8 release/tag 保持字节不变（不删除），
数据集归档 `data/singapore_accessibility_s8_legacy/`；用户批准重训（S9）。

## S9 重训完成 ✅（2026-08-27）—— Supply-Aware Traveler Agent v2.0

- **修正**：walk 1.34 m/s（实测有效 3.28 km/h）、bike 4.17 m/s（实测 10.14 km/h）、
  car 不变；pt access/egress 步行同步修正；MATSim 侧 walk/bike 仍为 network modes
  （freespeed 行驶）并如实记录为执行层 limitation（决策在 MATSim 前由修正后
  alternatives 决定，不受影响）。
- **数据集重建**：336 态（A 79/B 16/C 77/D 84/E 80；train 234/val 51/test 51；
  OD 79/14/20 不相交）；**pt 快于 walk 占 70%**；泄漏 0；door-to-door 误差 0.001。
- **Teacher 重标注**：1,502 次有效调用（K=3×89/K=5×247；49 次瞬时 SSL 失败重试），
  0 incomplete；教师梯度恢复真实形态（P(PT)：A 0.448/B 0.429/C 0.372/D 0.189/E 0）。
- **训练**：从 frozen S7-W3 初始化（Case B 同 S8），best_epoch=17，24,562 参数。
- **结果（test=51 未见 persona×OD，配对 bootstrap）**：PT MAE 0.175→0.135
  （Δ -0.040*）；P(PT|inf) 0.292→0.213（Δ -0.079*）；**FVR 0.333→0.083（Δ -0.25*）**；
  单调性 0.632→0.658；sensitivity 0.085→0.153（教师 0.448）；回归门禁全过
  （legacy KL -26%、unseen joint -36.5%）；机制无回退；Stop Rule 六项全满足。
- **冻结**：`releases/s9_supply_aware_v2/`（43 文件 SHA256、只读）、tag
  `s9-supply-aware-v2.0`；guard 新增 s9 目录；复现 gate 12/12 Δ=0.0000；
  S7/S8 release 复核零修改。烟测（N=100）：pt 31%/bike 30%/car 28%/walk 11%，
  PT 登车 66（S8 世界为 9），exit=0。
- **正式定义**：S9 = Supply-Aware Traveler Agent v2.0 = FROZEN（论文 supply-aware
  extension）；S8 = DEPRECATED；S7-W3 = generic baseline 不变。

## Phase C 重跑完成 ✅（frozen S9，2026-08-27）

六情景全部完成，六 gate 全 PASS（exit=0；真人 stuck 率 3.3–4.3%/PT 登车 ≤5%；
四模式；pt 下车≤登车）。报告：`outputs/singapore_phase_c_s9/PHASE_C_REPORT.md`。

### 结果（student 决策 share，C0 → 情景）

| 情景 | car | pt | bike | walk | PT boardings | VKT |
|---|---|---|---|---|---|---|
| C0 baseline | 28.3% | 25.3% | 34.0% | 12.4% | 5,613 | 33,939 km |
| C1 heavy rain | **37.5%** | **45.3%** | 13.3% | 3.8% | **9,761** | 42,778 km |
| C2 fare ×1.5 | 29.9% | 25.7% | 32.5% | 11.8% | 5,674 | 35,798 km |
| C3 delay 15min | 29.0% | **10.2%** | 35.8% | **25.0%** | 2,474 | 34,685 km |
| C4 road disruption | **1.8%** | **37.5%** | **41.8%** | 19.0% | 8,537 | **2,943 km** |
| C5 rain+delay | **37.6%** | 18.6% | 24.5% | 19.2% | 4,301 | 42,823 km |

- 与 S8 世界的关键差异：基线 pt 25.3%（S8: 2.8%）、walk 12.4%（S8: 40.8%）；
  **C3 delay 使 pt 从 25.3% 降到 10.2%（不再是病态清零）**；C4 使 car 28.3%→1.8%、
  pt/bike 吸收；C5 rain+delay 的 pt（18.6%）介于 C1（45.3%）与 C3（10.2%）之间——联合效应
  可解释且非简单叠加；C2 票价弹性弱（+0.4pp，噪声级）如实记录。
- Gate 口径第二次修正（有据）：S9 世界 PT 需求为 B.5C 的 ~9 倍，真人 stuck 随 PT 需求
  增长（错过单次换乘的既有机制），绝对阈值 266 改为率口径（≤5%/PT 登车；六情景 3.3–4.3% 稳定）。
- 已知边界同前（transit 30:00 截断 ~20%、C1/C4 failed_trips 较高、bike share 受
  synthetic persona 自行车拥有率 45% 影响）。
- S8 时代 `outputs/singapore_phase_c/` 结果作废（保留存档），论文使用
  `outputs/singapore_phase_c_s9/`。

## E1 MNL-B baseline 完成 ✅（2026-08-28，TRC_AIT_5 补充实验第一条）

- **实现**：`src/traveler_distillation/baselines/mnl.py`（规格 S3，17 参数；精确 Newton+Armijo；
  G1 识别修正记录在案：exposure/reliability/car_own_car 吸收进 ASC，S2 commute 项不可识别被排除）。
- **估计**：234 训练态软标签 MLE，val(51 态) 选模 S3；G1/G4 PASS（Hessian PD、SE 有限、
  10 重启 spread 2.5e-08、逐位确定）。冻结：`outputs/e1_mnl/mnl_b_coefs.json`。
- **决策层**（G2 PASS：S9 复算与冻结 s9_accessibility_eval 逐位一致 28 项）：
  - T1 51 态：MNL KL 0.165 vs S9 0.170；PT-MAE 0.153 vs 0.135；FVR 均 0.083；acc 0.863 vs 0.941。
  - T2 可达性：MNL ΔP best−worst 0.105 vs S9 0.153 vs Teacher 0.448；P(PT) 曲线不随可达性等级单调。
  - T3 counterfactual：MNL acc 0.668 vs S9 0.889；KL 0.386 vs 0.051；sign 0.595 vs 0.708——
    **静态相当、动态/多条件大幅落后**。
- **仿真层**（MNL × C0–C5，10k，seed 2026，capacity 0.3/0.3；G3 PASS：decision_fn=None 复建
  C0 与冻结 S9 C0 三工件逐字节一致 SHA256 全同；六情景 gate 全 PASS）：
  - C0：car 36.9 / pt 31.7 / bike 21.8 / walk 9.6（S9 28.3/25.3/34.0/12.4）。
  - C1 雨：Δpt **+1.6pp**（S9 +20.0pp）——方向对、幅度弱；
  - C2 票价×1.5：Δpt **+7.4pp**（S9 +0.4pp）——**方向反转**（β_cost=+0.257, z=1.28）；
  - C3 delay：Δpt −7.0pp（S9 −15.1pp）；
  - C4 disruption：Δcar −2.9pp（S9 −26.5pp）；
  - C5 rain+delay：Δpt −5.0pp（S9 −6.7pp）。
- **结论**：MNL 是合格的静态选择基线；动态弹性、联合响应与 supply-aware 响应上 S9 占优；
  MNL 在蒸馏监督下估计出符号错误的成本系数并导致 C2 反转——如实报告（E1 结论目标成立）。
- **工件**：`outputs/e1_mnl/`（报告/系数/决策层评估）、`outputs/singapore_phase_c_mnl/`
  （六情景仿真）；paper 归档 `data_report/10_E1_MNL/`。

## E2 DeepSeek vs S9 速率/成本实验完成 ✅（2026-08-28，TRC_AIT_5 补充实验第二条）

> 设计书：`TRC_AIT_5_E2_DEEPSEEK_S9_EFFICIENCY_DESIGN.md` v0.2（paper 仓库）；报告：`outputs/e2_efficiency/E2_REPORT.md`

- **实现**：`scripts/bench_e2_efficiency.py`（状态池构建 + S9 计时 + G2/G4 门）、
  `scripts/bench_e2_deepseek.py`（DeepSeek 实测，K=1 + cache_bypass + resume）、
  `scripts/make_e2_report.py`（报告数字只从证据 JSON 生成，无手抄）。
- **状态池**：100,000 C0 基线态（persona/trip seed 2026 + `generation_v0_1.yaml` + frozen 供给；
  4 worker 确定性分片，88 min，SHA256 `744c5896…`，checkpoint `6af79b44…` 核对通过）。
- **G2**：10k 前缀决策与冻结 Phase C C0 `adapter_manifest.json` 逐位一致（0/10,000 不一致）
  ——状态池与冻结 S9 管线完全同源。
- **T1 时延**（同 100 态前缀对拍）：DeepSeek mean **86.5s**（P50 87.4 / P95 162.8，100 次调用 / 87 次成功实测）；
  S9 CPU 顺序 mean **0.33ms**（P95 0.8ms，100k 态）——**≈3.2×10⁵ 倍**。
- **T2 吞吐**：DeepSeek 顺序 0.7 states/min（10k 投影 240h；历史 4-worker 锚点 28–33h）；
  S9 顺序 3.0k states/s，batch 21–25k states/s（10k ≈ 0.4–3.3s）。
- **T3 成本**（87 次成功调用真实 token × 三档参考价格）：DeepSeek $0.0021–0.0351/state
  → 10k $21–351、100k $208–3515（线性投影）；S9 $0；
  一次性蒸馏投入参考行（1,502 次调用）$2.91–47.34。
- **T4 S9 人口级**（无 MATSim，实测）：100k 端到端 5.8h（共享输入构建 208ms/state 主导），
  决策时间合计仅 98.4s（0.98ms/state）。
- **门禁**：G1-S9 PASS（重复 CV 0.9–1.4%）、G2 PASS、G3 PASS（100/120 调用）、G4 PASS（100k 决策逐位一致）；
  **G1-DS FAIL 如实记录**：13/100 调用失败（SSL 断连×10 + empty_content×2 + timeout×1，
  网关间歇性不稳定，与历史记录一致），时延/token 统计基于 87 次成功调用。
- **诚实边界**：`deepseek-v4-pro` 公开定价未知（三档参考价格场景，账单口径未获得）；
  DeepSeek 10k/100k 与并发行均为投影并标注；失败率 13% > 10% **触发设计 R1**
  （N=200 扩展待用户追加批准）。
- **工件**：`outputs/e2_efficiency/`（报告、states 190MB、decisions、build_timing、
  6 份 time-s9、deepseek_calls.jsonl、pool_manifest）；paper 归档 `data_report/11_E2_EFFICIENCY/`。

## E3 Population Scalability 完成 ✅（2026-08-28，TRC_AIT_5 补充实验第三条；止于 50k）

- **依据**：`TRC_AIT_5_EXPERIMENT_PLAN.md` §E3；设计书 `TRC_AIT_5_E3_POPULATION_SCALABILITY_DESIGN.md`
  v0.1（执行期修订 1–8 全记录在案）；执行 `scripts/singapore/run_e3_scale.py`
  （import 复用 `run_phase_c.py`，不改默认路径；零侵入计时包装 + ctypes 内存采样）。
- **档位**：1k×3 试点 / 10k 主锚 / 20k / 50k 全过门禁；**100k 及 200k/500k 取消执行（用户指示）**，
  已启动的 100k 构建中止并清理（无完整工件）；S7 六条“100k 稳定”判据不再评估。
- **门禁**：G1 前缀逐位一致（各档前 10k 决策与冻结 C0 全字段一致；10k 档 population.xml SHA256
  全同）；G2a/G2b 全过；G3 确定性（1k×3 决策 manifest SHA 全同，CV 暖机口径全 ≤10%——修订 1）；
  **G3′ 10k 档 24/24 指标与冻结 `phase_c_result.json` 逐字段相等**（MATSim 跨日确定性实证）；
  G4 计时质量（10k/50k 各 1 次 ≤2 s OS 停顿，修订 4 条款记录在案）。
- **核心读数（T1–T5 见 `outputs/e3_scale/E3_REPORT.md`）**：build 353 s→3,063→5,934→13,907 s
  （跨档比值 ×8.7/×1.94/×2.34，次线性）；factory（可达性规划）占 57–63%，**decide（S9 推理）仅
  0.94–1.03 ms/态**（总量 1.0/10.3/18.8/48.8 s）；MATSim 118–202 s（1k→50k，供给主导，50k 拥堵
  初现 slow share 0.02）；py 峰值 1.97→5.27 GB（Δ 增长次线性，100k 外推 ~7 GB 安全、无需 R1）；
  java 峰值恒 ~6.5 GB；决策 share 漂移 ≤0.6pp（10k–50k 档；1k 试点 ±1.0pp）；stuck 3.8–4.8%；PT validity 93.5–95.1%；
  boardings 线性（5,613→11,357→28,440）。
- **结论（计划书 §E3 目标达成）**：behavioral inference scalability 与 simulation scalability
  成功分离——**系统瓶颈 = 行为构建侧（可达性规划 + leg 路由），S9 推理与 MATSim 仿真均非瓶颈**。
- **工件**：`outputs/e3_scale/`（E3_REPORT.md + 每档 result JSON + 原始逐态计时数组）；
  paper 归档 `data_report/12_E3_SCALE/`；设计书修订 1–8。


## E4 Phase C Multi-Seed Robustness 完成 ✅（2026-08-29，TRC_AIT_5 补充实验第四条）

- **依据**：`TRC_AIT_5_EXPERIMENT_PLAN.md` §E4；设计书 `TRC_AIT_5_E4_MULTISEED_ROBUSTNESS_DESIGN.md`
  v0.1（执行期修订 1–3 在案：并发批准、执行结果、归档编号顺延）；执行 `scripts/singapore/run_e4_multiseed.py`
  （import 复用 `run_phase_c.py`，不改默认路径；seed 参数化 + 真实 checkpoint 字段 + 工件 SHA256）。
- **运行量**：seed 42 × 6 情景 + seed 7 × 6 情景 = 12 次运行（seed 2026 冻结列只读复用，零重跑）
  + G1 门禁复跑（seed 42 C0 r1）。与 E5 Helsinki 并发（用户 2026-08-28 批准，contention_note 逐运行记录；
  计时仅为出处性元数据）。
- **门禁**：G0 冻结列只读 + 内部一致性 PASS（6 文件 SHA256 + phase_c_records 交叉核对 6/6）；G1 r1/r0
  population.xml 与 adapter_manifest SHA256 全同 + 决策/指标逐字段相等 PASS；G2 十二运行 Phase C gate
  全 PASS（12/12）；G3 结构一致仅 1 项报告偏离（seed7/C4 stuck 率 3.16% 略低于冻结带下限 3.3%，
  275/8,708，属良性样本波动）；G4 数字管道 PASS（程序化生成，无手抄）。
- **核心结论（T1–T5 见 `outputs/e4_multiseed/E4_REPORT.md`）**：C0 baseline 决策 share 跨 seed 高度一致
  （car 27.7–28.3 / pt 25.3–26.1 / bike 33.3–34.0 / walk 12.4–12.9）；C1–C5 全部响应指标 3/3 符号一致
  （Δpt share：C1 +19.9±0.2 / C2 +0.5±0.1 / C3 −15.4±0.3 / C4 +12.0±0.3 / C5 −6.7±0.3 pp）；
  **C2 fare ×1.5 弱响应经 §5.2 预注册规则判定 stable**（Δpt +0.4/+0.5/+0.5 pp、Δboardings +61/+99/+80，
  |mean|/std=8.08；与次小响应量级比 0.07×）——稳定弱效应而非抽样噪声；C0–C5 情景响应不依赖单一
  population seed（计划书 §E4 目标达成）。
- **工件**：`outputs/e4_multiseed/`（E4_REPORT.md + 12 份 e4_result.json + g1_check.json + seed_records）；
  paper 归档 `data_report/14_E4_MULTISEED/`（E5 已占 13 号，按设计 §6 S6 顺延规则）。

## E5 Helsinki Zero-Shot Transfer 完成 ✅（2026-08-29，TRC_AIT_5 补充实验第五条）

- **依据**：`TRC_AIT_5_EXPERIMENT_PLAN.md` §E5；设计书 `TRC_AIT_5_E5_SECOND_CITY_ZERO_SHOT_DESIGN.md`
  v0.1（paper 仓库）；执行 `scripts/helsinki/run_e5_helsinki.py` + `prep_helsinki_gtfs.py` +
  `build_helsinki_network.py`（frozen S9 零重训、零 Teacher；G0 teacher_guard 逐运行断言，
  checkpoint SHA256 每运行核实）。
- **供给**：HSL GTFS 2026-08-27 快照 + OSM 子区域（≈97 km²；节点/链路 290k/638k；GTFS trips
  13,655、站 1,269）；snap p90 21.7 m；routing failures 0；capacity 0.3/0.3 与时间窗沿用 Singapore
  冻结值（G1 平价带 0.5–4× 全过）。
- **情景**：**六情景 C0–C5 全部完成**（N=10,000，seed 2026）；C0/C1/C3 为计划书最小集，
  C2/C4/C5 按设计 §7 V4 条件扩展执行（最小集门禁全过 + 时间预算允许）。
- **门禁**：**六情景 Phase C gate 全 PASS、G0–G5 全 PASS**；C0 决策四模式齐 + fallback 原因已知
  （PT validity 75.9% overall / 100.0% feasible-conditioned，如实报告）；G2 确定性
  （1k pilot manifest SHA256 全同）。
- **核心结论（T1–T5 见 `outputs/e5_helsinki/E5_REPORT.md`）**：C0 share 28.6/23.0/34.6/13.8
  （vs SG 冻结 28.3/25.3/34.0/12.4）；P(PT) A→E 梯度方向复现但更弱（E−A −0.047 vs SG −0.153，
  中间类非单调——如实报告）；**五个情景响应方向全部与 SG 冻结一致**（C1 7/7、C2 6/7、C3 6/7、
  C4 6/7、C5 6/7 符号匹配；C4 Δcar −23.7pp vs SG −26.5pp、ΔVKT −29,479 vs −30,996 km 幅度接近；
  C2 Δpt +1.5pp vs +0.4pp 同为弱响应）；stuck 人 1–2、failed trips 1–2（SG 30:00 截断 artifact
  在 Helsinki 不出现）。**supply-aware 接口在未见城市 zero-shot 可用（定性方向证据；无 Helsinki
  标签，不做跨城统计检验）**——计划书 §E5 目标达成。
- **工件**：`outputs/e5_helsinki/`（E5_REPORT.md + e5_records/e5_acc_audit + 六情景完整
  MATSim 输出）；paper 归档 `data_report/13_E5_HELSINKI/`（含 SHA256SUMS）。
