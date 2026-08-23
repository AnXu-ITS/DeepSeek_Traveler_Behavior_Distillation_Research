# 下一阶段研究执行计划  
## Singapore Real-World Validation × Traveler Agent Distillation × MATSim

**项目主线**：DeepSeek V4 Pro → 轻量 Traveler Agent → MATSim  
**建议投稿方向**：Artificial Intelligence for Transportation（AIT）  
**计划状态**：从“模型开发阶段”切换到“真实性验证 + 论文收口阶段”  
**依据文件**：`Task_Phase.txt`、`PROGRESS.md`  
**计划版本**：v1.0  
**日期**：2026-08-23

---

# 1. 当前研究状态

## 1.1 已完成的核心工作

现有项目已经完成完整的端到端技术链：

1. 研究行为边界与 Universal State/Action；
2. DeepSeek V4 Pro Teacher Prompt；
3. Scenario Generator；
4. K=3 聚合 Teacher Dataset；
5. 蒸馏设计：
   - Choice / Distribution；
   - Decomposed Elasticity；
   - Heterogeneity；
6. 约 24k 参数的 lightweight Traveler Agent；
7. Individual-level evaluation + persona-holdout；
8. MATSimAdapter；
9. 1000-person population-scale simulation；
10. Network feedback loop。

目前六个动态扰动轴已经全部覆盖：

- weather；
- fare；
- road congestion；
- transit delay；
- parking cost；
- road disruption。

最终训练数据规模已经扩展到：

- **40 personas**；
- **1513 aggregated states**；
- **4539 DeepSeek repeat records**；
- **80 baselines**；
- **1433 counterfactual states**；
- persona-holdout test = **6 unseen personas / 226 states**。

现阶段最重要的实验发现包括：

- Student 可以保留 Teacher 的 mode choice 与概率分布；
- elasticity-aware training 能改善动态扰动响应；
- heterogeneity-aware training 能进一步改善分布拟合；
- 六个扰动轴的蒸馏难度与 Teacher signal-to-noise ratio 存在稳定关联；
- Student 决策已经成功进入 MATSim；
- population-level mode shift 和 network feedback loop 已经跑通。

因此，**后续不再以继续扩大模型、增加扰动轴或增加 persona 数量为主线。**

---

# 2. 当前论文真正的剩余缺口

目前项目的主要短板不是“AI 模型还不够强”，而是以下四类验证仍然不足。

## 2.1 Supply realism 不足

当前 Phase 8–10 仍主要依赖：

- synthetic grid；
- teleported public transport；
- 人工设置道路容量；
- road congestion 单一网络反馈。

下一阶段需要将其替换为：

> **Singapore OSM real road network + GTFS-based public transport network**

---

## 2.2 Demand / behavioral realism 不足

当前 persona 和 trip 仍然属于 synthetic behavioral states。

现有实验已经能够证明：

> Student ≈ DeepSeek Teacher

但 journal 论文还需要进一步回答：

> Teacher / Student 的行为响应是否与真实交通行为规律一致？

因此需要增加 **external behavioral plausibility validation**。

---

## 2.3 Transportation baseline 不足

现有 A/B/C 主要属于内部 ablation：

- A：基础 choice/distribution；
- B：+ elasticity；
- C：+ heterogeneity。

还需要增加至少一个交通行为领域基线：

- **Multinomial Logit (MNL)**。

可选增加：

- conventional MLP / XGBoost。

---

## 2.4 Teacher uncertainty 目前主要是相关性证据

当前已经观察到：

> Student elasticity fitting ceiling 与 Teacher axis SNR 高度一致。

但目前主要属于 observational evidence。

下一阶段应设计：

> **K=3 → K=5 / K=7**

来验证：

Teacher noise ↓  
→ distilled behavioral response quality ↑

从而把这个发现从“现象”提升为论文中的独立机制性贡献。

---

# 3. 下一阶段总目标

下一阶段不再进行开放式模型开发，而是完成以下五个目标：

### G1 — Singapore real-world multimodal environment
将现有 synthetic grid 替换为真实新加坡道路 + 公共交通网络。

### G2 — Transportation-domain comparison
增加 MNL 等传统交通行为模型 baseline。

### G3 — External behavioral plausibility
验证 Teacher / Student 对票价、天气、停车费、延误、中断等变化的响应是否与经验规律一致。

### G4 — Teacher-noise causal validation
验证增加 Teacher repeat sampling 能否降低噪声并改善 Student elasticity。

### G5 — Scalability / deployment evidence
正式量化 DeepSeek Teacher 与 lightweight Student 在速度、成本、参数量和大规模仿真可执行性上的差异。

---

# 4. Singapore Case Study 设计

## 4.1 推荐主研究区域

### Primary case study

**Tampines + Pasir Ris**

推荐原因：

- car / bus / MRT / walk / bike 均存在；
- 居住功能明显；
- 公共交通体系完整；
- 路网规模可控；
- 比 CBD 更容易解释；
- 适合研究 rain、fare、parking、congestion、transit delay、road disruption。

建议不要直接模拟整个新加坡。

推荐先裁剪约 **10–15 km 级别区域**，以保持 MATSim 网络规模和数据转换可控。

### Backup case study

**Jurong East + Clementi**

当 Tampines / Pasir Ris 的公交线路匹配或 GTFS route mapping 出现严重问题时，可作为备用区域。

---

# 5. Singapore 数据方案

## 5.1 Road network

使用：

> **OpenStreetMap Singapore road network**

目标产物：

```text
network.xml
```

需要保留：

- OSM 下载日期；
- 原始 `.osm.pbf`；
- 裁剪 bbox；
- OSM → MATSim 转换配置；
- 坐标系信息；
- network statistics。

---

## 5.2 Public transport

第一阶段直接使用当前已经找到的：

> `singapore-gtfs-2026 / singapore-gtfs.zip`

暂时**不要求申请 LTA DataMall API**。

但论文中必须准确描述数据性质：

- Bus services / routes / stops 基于 LTA DataMall；
- bus travel times 含估算；
- MRT schedules 为 synthetic / frequency-based schedules；
- 不能将整个 ZIP 表述为“官方实测 GTFS timetable”。

建议保存：

```text
data/singapore/gtfs/raw/singapore-gtfs.zip
data/singapore/gtfs/README_snapshot.md
data/singapore/gtfs/source_metadata.json
data/singapore/gtfs/checksum.sha256
```

这样可保证论文数据版本可追溯。

---

## 5.3 Optional external data

后续用于 external plausibility 时，再考虑加入：

- Singapore public transport ridership；
- LTA passenger-volume data；
- published Singapore mode share；
- published fare elasticity；
- rain / weather-related travel behavior literature；
- parking-price sensitivity literature。

这些数据主要用于：

> **sanity check / behavioral plausibility**

而不是声称已经完成真实人口 calibration。

---

# 6. Phase 11 — 当前模型与数据冻结

## 目标

正式结束“模型继续扩展”阶段。

## 工作

- [ ] 冻结 40-persona / 6-axis 数据集；
- [ ] 冻结当前 A/B/C checkpoint；
- [ ] 保存最终 config；
- [ ] 保存所有关键 metrics；
- [ ] 为当前版本建立 Git tag / release snapshot；
- [ ] 建立 `FINAL_DEVELOPMENT_BASELINE.md`；
- [ ] 记录 1513 states / 4539 repeats 的 checksum；
- [ ] 不再默认扩展 persona 数量；
- [ ] 不再增加第七个 perturbation axis。

## Gate

只有发生明确的数据错误或方法错误时，才重新训练主模型。

---

# 7. Phase 12 — Singapore OSM + GTFS 数据接入

## 目标

构建真实 Singapore multimodal MATSim supply network。

## 12.1 OSM pipeline

- [ ] 下载 Singapore OSM；
- [ ] 裁剪 Tampines + Pasir Ris；
- [ ] 转换为 MATSim `network.xml`；
- [ ] 检查：
  - connected components；
  - one-way links；
  - road class；
  - free speed；
  - capacity；
  - link length；
  - coordinate system。

## 12.2 GTFS pipeline

- [ ] 解压 `singapore-gtfs.zip`；
- [ ] 检查基本 GTFS integrity；
- [ ] 提取目标区域 stops / routes / trips；
- [ ] GTFS → MATSim：
  - `transitSchedule.xml`
  - `transitVehicles.xml`
- [ ] stop 坐标投影；
- [ ] stops → MATSim links snapping；
- [ ] transit route → road network routing；
- [ ] 修复 disconnected / invalid routes。

## 12.3 必须重点检查的问题

最可能出现的工程问题：

1. GTFS stop 不在 road network link 上；
2. bus route 与 OSM 单行道不兼容；
3. route stop sequence 可用，但 road routing 不可达；
4. CRS 不一致；
5. 某些线路跨出研究区域；
6. transit vehicle / departure 配置不完整；
7. MRT schedule 与 road-based bus routing 混合时产生配置问题。

## Gate

完成一个最小 baseline：

> 100 agents + real OSM + scheduled PT → MATSim exit=0

并满足：

- 没有大量 broken transit routes；
- bus / MRT / car / walk / bike 均能执行；
- PT 不再完全 teleported。

---

# 8. Phase 13 — Singapore Real-Network Smoke & Scaling

## 目标

先证明真实网络运行稳定，再做科学实验。

## 实验规模

依次运行：

- 100 agents；
- 500 agents；
- 1000 agents。

暂时只跑 baseline。

## 输出

记录：

- MATSim runtime；
- number of trips；
- mode share；
- failed / unroutable trips；
- PT boardings；
- road travel time；
- network delay；
- average trip duration；
- link congestion statistics。

## Gate

1000 agents baseline 稳定运行后，才能进入主实验。

---

# 9. Phase 14 — Singapore Main Scenario Experiment

## 目标

将已有 behavioral intelligence 投射到真实城市网络。

个人层面仍保留完整六轴实验。

网络层面不需要机械地把六轴全部做成复杂 supply interventions。

推荐把以下四类作为 **main Singapore case-study scenarios**：

### S0 — Baseline

真实 OSM + GTFS，正常环境。

### S1 — Heavy Rain

Traveler context：

- rain / severe weather。

主要观察：

- bike / walk ↓；
- car / PT ↑；
- network congestion change。

### S2 — PT Fare Increase

Traveler context：

- public transport fare multiplier。

观察：

- PT demand loss；
- mode substitution；
- road demand / congestion change。

### S3 — Road Disruption

真实关闭一个具有较高交通作用的 road link / corridor。

同时更新：

- route travel time / availability；
- road disruption context。

观察：

- mode switching；
- route/network delay；
- behavioral adaptation；
- equilibrium response。

### S4 — Transit Delay

在 GTFS/MATSim schedule 层引入：

- additional delay；
- increased travel time；
- 或降低 service reliability。

观察：

- PT demand change；
- car substitution；
- multimodal network response。

---

# 10. Phase 15 — Singapore Network Feedback Loop

## 目标

将现有 Phase 10 的 synthetic network feedback loop 迁移到真实网络。

现有逻辑继续保留：

```text
Student decision
    ↓
MATSim
    ↓
observed congestion / travel time
    ↓
context update
    ↓
Student re-decision
    ↓
new population demand
    ↓
MATSim
```

推荐至少跑：

- baseline；
- rain；
- fare increase；
- road disruption。

## 主要指标

- equilibrium congestion；
- equilibrium car share；
- equilibrium PT share；
- convergence iterations；
- mean travel time；
- VKT；
- network delay；
- link-level congestion distribution。

## 目标

验证：

> Disturbance → heterogeneous behavioral adaptation → population demand shift → real-network response → behavioral feedback

而不仅仅是 mode-choice accuracy。

---

# 11. Phase 16 — Transportation Baselines

## 目标

回答：

> 为什么不直接使用传统 travel behavior model？

## Baseline 1 — MNL

训练 / 拟合输入：

- persona attributes；
- trip attributes；
- mode attributes；
- dynamic context。

输出：

- mode probabilities。

与 Student 比较：

- mode accuracy；
- KL；
- probability L1；
- counterfactual ΔP error；
- sign agreement；
- heterogeneity preservation；
- unseen-persona performance；
- inference speed。

## Baseline 2 — Standard ML（可选）

优先顺序：

1. plain MLP；
2. XGBoost。

注意：

当前 Student-A 本身已经接近 plain neural baseline，因此不必为了“数量多”堆很多 ML baseline。

---

# 12. Phase 17 — Teacher Noise Causal Experiment

## 目标

验证当前最有潜力的机制性发现：

> Teacher uncertainty limits behavioral distillation.

## 推荐轴

优先：

> **road_congestion**

原因：

- 与最终 MATSim feedback loop 直接相关；
- 当前属于较低 SNR；
- 已经观察到 Student magnitude under-response。

备选：

> fare。

## 实验

固定同一组 counterfactual states：

```text
K=3
K=5
K=7
```

不要重新增加 persona。

只增加同一 state 的 Teacher repeated sampling。

## 检验链

### Step 1

验证：

```text
Teacher pairwise noise:
K=3 > K=5 > K=7
```

### Step 2

分别聚合 Teacher targets。

### Step 3

训练相同 Student。

### Step 4

比较：

- elasticity magnitude error；
- sign agreement；
- ΔP gap；
- KL；
- probability L1。

## 成功结果

如果出现：

> Teacher noise ↓ → Student elasticity fidelity ↑

则可以形成独立论文贡献：

> Behavioral distillation is constrained by teacher response consistency rather than merely student capacity or dataset size.

---

# 13. Phase 18 — External Behavioral Plausibility

## 目标

补上：

> Student ≈ Teacher

之外的第二层证据：

> Teacher / Student response ≈ empirical behavioral expectation

## 推荐验证轴

优先选 3–4 个经验文献较成熟的变量：

1. PT fare；
2. parking cost；
3. heavy rain；
4. transit delay。

## 两级验证

### Level A — Directional plausibility

例如：

```text
fare ↑      → PT ↓
parking ↑   → car ↓
rain ↑      → walk/bike ↓
PT delay ↑  → PT ↓
```

### Level B — Magnitude plausibility

当能够找到可比较的 empirical elasticity 时：

比较：

- published elasticity；
- Teacher elasticity；
- Student elasticity。

注意：

如果数据定义、城市、时期不同，只做区间或量级比较。

不得把不同来源的 elasticity 当作严格 ground truth。

---

# 14. Phase 19 — Efficiency & Scalability Benchmark

## 目标

正式回答：

> 为什么需要 distillation？

## 对比

### DeepSeek Teacher

记录：

- API latency；
- token usage；
- monetary cost；
- repeated-sampling cost；
- ability to scale to 1k / 10k agents。

### Student

记录：

- parameter count；
- model size；
- CPU inference latency；
- GPU inference latency（如需要）；
- memory；
- 100 / 1000 / 10000 decisions runtime；
- offline availability。

## 核心表

| Metric | DeepSeek Teacher | Student |
|---|---:|---:|
| Model access | API | Local |
| Parameters | Large proprietary model | ~24k |
| 1 decision latency | measured | measured |
| 1000 decisions | measured/estimated | measured |
| 10000 decisions | measured/estimated | measured |
| API cost | measured | ~0 marginal API cost |
| Offline execution | No | Yes |
| MATSim population execution | impractical | Yes |

---

# 15. Phase 20 — Statistical Robustness

## 最低要求

### Student training

- [ ] 3–5 random seeds；
- [ ] report mean ± std。

### Individual-level evaluation

对：

- KL；
- L1；
- ΔP gap；
- sign agreement；

增加 bootstrap 95% CI。

### Singapore simulation

至少对主要 scenario 使用：

- 3 个 population / trip random seeds。

观察：

- mode share shift；
- network delay；
- equilibrium congestion；

是否稳定。

---

# 16. Phase 21 — Final Experiment Matrix

建议最后论文形成以下证据层：

| Evidence level | Question | Experiment |
|---|---|---|
| E1 | Student 能否复制 Teacher？ | A/B/C + holdout |
| E2 | 能否保留动态 elasticity？ | 6-axis counterfactual |
| E3 | 能否保留 heterogeneity？ | persona contrast |
| E4 | 是否优于传统行为模型？ | MNL baseline |
| E5 | Teacher noise 是否限制蒸馏？ | K=3/5/7 |
| E6 | 是否能进行 population-scale execution？ | 1000 agents |
| E7 | 是否能进入真实交通系统？ | Singapore OSM + GTFS |
| E8 | 是否产生网络后果？ | real network feedback loop |
| E9 | 行为是否具有外部合理性？ | empirical plausibility |
| E10 | 为什么必须蒸馏？ | latency/cost/scalability |

这十层证据完成后，论文故事将不再是：

> “我们把 LLM 做小了。”

而是：

> **We distill heterogeneous and context-sensitive travel behavior from a large language model into an executable traveler agent, quantify when such behavioral intelligence can be reliably distilled, and demonstrate its population-scale and network-level consequences in a real multimodal transport environment.**

---

# 17. 推荐论文核心 Research Questions

## RQ1 — Fidelity

Can a lightweight Traveler Agent preserve the choice distributions of a large language model for unseen traveler personas?

## RQ2 — Behavioral responsiveness

Can distillation preserve heterogeneous responses to dynamic transportation contexts rather than only static mode choices?

## RQ3 — Teacher uncertainty

How does teacher response consistency affect the fidelity of distilled behavioral elasticity?

## RQ4 — Simulation executability

Can the distilled agent be executed at population scale inside MATSim?

## RQ5 — System consequence

Do context-induced behavioral adaptations propagate into measurable network-level outcomes in a realistic multimodal urban network?

---

# 18. 推荐最终 Contributions

论文最后建议收敛为四个核心贡献。

### C1 — Behavioral distillation framework

提出从 large LLM teacher 向 lightweight executable traveler agent 蒸馏：

- choice；
- probability distribution；
- elasticity；
- heterogeneity。

### C2 — Counterfactual / heterogeneity-aware learning

证明仅学习 choice 不足以保持动态出行行为，加入 elasticity / heterogeneity supervision 能改善未见 persona 的行为响应。

### C3 — Teacher uncertainty as a distillation bottleneck

揭示不同 context axis 的蒸馏质量受到 Teacher signal-to-noise ratio 限制，并通过 K-sampling 实验验证该机制。

### C4 — Real-world transportation execution

在 Singapore OSM + GTFS multimodal environment 中，将 distilled individual behavior 扩展到 population demand 和 network feedback。

---

# 19. 明确不做的事情

为了避免项目再次无限扩张，下一阶段默认**不做**：

- [ ] persona 40 → 80 / 100；
- [ ] 增加第七、第八个 context axis；
- [ ] 再换新的 Student architecture；
- [ ] 增加大量 Teacher 模型进行横向比较；
- [ ] 直接模拟整个 Singapore；
- [ ] 为了追求“真实”重新建设完整 Singapore travel demand model；
- [ ] 一开始就接 LTA realtime API；
- [ ] 把社区 GTFS 描述成完全官方 timetable；
- [ ] 把 synthetic personas 描述成 Singapore representative population；
- [ ] 在当前论文加入超出主线的新模块。

---

# 20. 推荐目录结构

```text
data/
  singapore/
    osm/
      raw/
      clipped/
      network/
    gtfs/
      raw/
      processed/
      README_snapshot.md
      source_metadata.json
      checksum.sha256
    population/
    scenarios/

outputs/
  singapore_smoke/
  singapore_population/
  singapore_feedback/
  baselines/
  teacher_noise/
  efficiency/
  external_validation/

reports/
  FINAL_DEVELOPMENT_BASELINE.md
  SINGAPORE_DATA_AUDIT.md
  SINGAPORE_NETWORK_VALIDATION.md
  SINGAPORE_SCENARIO_REPORT.md
  BASELINE_COMPARISON.md
  TEACHER_NOISE_CAUSAL_REPORT.md
  EFFICIENCY_REPORT.md
  EXTERNAL_PLAUSIBILITY_REPORT.md
  FINAL_PAPER_EVIDENCE_MATRIX.md
```

---

# 21. 推荐执行顺序

## P0 — 立即执行

1. Freeze 当前 40-persona / 6-axis 主实验；
2. 下载并裁剪 Singapore OSM；
3. 接入现有 `singapore-gtfs.zip`；
4. 完成 OSM + GTFS → MATSim；
5. 100 → 500 → 1000 agent smoke；
6. 跑 Singapore baseline；
7. 跑 rain / fare / disruption / transit-delay 主场景；
8. 将 Phase 10 feedback loop 迁移到真实网络。

## P1 — Journal 必须补强

9. MNL baseline；
10. efficiency / scalability benchmark；
11. external behavioral plausibility；
12. 3–5 seeds / bootstrap CI；
13. K=3/5/7 Teacher-noise experiment。

## P2 — 只有论文仍明显不足时再做

14. 第二个 Singapore region；
15. 第二城市 transfer；
16. LTA API 重建最新版 feed；
17. 更大人口规模。

---

# 22. 预计三周收口路线

## Week 1 — Singapore Supply Realism

目标：

> real OSM + GTFS MATSim baseline 跑通

完成：

- Phase 11；
- Phase 12；
- Phase 13。

## Week 2 — Main Scientific Experiments

完成：

- Phase 14；
- Phase 15；
- Phase 16；
- Phase 17。

## Week 3 — Journal Evidence Closure

完成：

- Phase 18；
- Phase 19；
- Phase 20；
- final figures / tables；
- paper evidence matrix；
- manuscript drafting。

如果 Singapore GTFS route mapping 花费时间明显超过预期，应优先保证：

> real OSM + usable scheduled PT + one stable study region

而不是追求整个 Singapore 全线路完美复现。

---

# 23. 最终停止条件

当以下条件全部满足时，停止新增实验并进入论文写作：

- [ ] real Singapore OSM MATSim network 可稳定运行；
- [ ] GTFS public transport 已接入；
- [ ] 1000-agent baseline 成功；
- [ ] ≥4 Singapore dynamic scenarios 完成；
- [ ] real-network feedback loop 成功；
- [ ] MNL baseline 完成；
- [ ] Teacher-noise K experiment 完成；
- [ ] efficiency benchmark 完成；
- [ ] external behavioral plausibility 完成；
- [ ] statistical robustness 完成；
- [ ] 所有最终数字有统一 evidence report；
- [ ] limitations 明确区分：
  - supply realism；
  - behavioral plausibility；
  - population calibration。

满足这些条件后，项目应**停止继续开发**，直接进入 AIT manuscript preparation。

---

# 24. 一句话执行原则

> **当前项目已经不缺新的模型和新的扰动轴；下一阶段的全部工作应围绕“真实新加坡交通环境、交通学基线、外部行为合理性、Teacher 噪声机制和规模化价值”收口。**

最终目标不是继续证明 Student “能学”，而是证明：

> **它学到的行为值得被交通仿真系统使用，而且能够在真实城市网络中产生可解释、可扩展、可验证的系统级结果。**
