# 实验进度与执行记录

> 蒸馏出行意图 · 个体行为蒸馏研究（DeepSeek V4 Pro → 轻量 Traveler Agent → MATSim）
> 更新日期：2026-08-20（v0.2 实验接手）

## 研究目标（不变）

把 DeepSeek V4 Pro 对「不同 Persona × 动态 Context → 出行行为响应」的推理能力，
蒸馏为轻量 Traveler Agent，重点保留：

- 个体异质性（heterogeneity）
- 环境敏感性 / 行为弹性（elasticity：票价、天气、拥堵、延误变化时的行为响应）

并最终接入 MATSim 做 population-scale 与网络闭环验证。

## 当前阶段结论

Teacher 侧（Phase 0–4）已完成并审计通过；Student 侧（Phase 5–7）已建立
v0.2-A baseline、v0.2-B 弹性损失、v0.2-C 异质性损失三条训练管线，正在用
**真实** DeepSeek K=3 聚合数据集训练验证。MATSim（Phase 8–10）尚未接入。

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
