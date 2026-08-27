# 研究蓝图：面向城市级交通仿真的个体出行行为蒸馏

## 1. 研究定位

本研究关注一个核心问题：

> **能否将大型语言模型对“不同类型城市居民在动态环境变化下如何调整出行行为”的推理能力，蒸馏为可在城市级交通仿真中大规模运行的轻量化 Traveler Agents？**

研究对象不是车辆的微观驾驶行为，而是**城市居民的个体出行决策**。

研究核心可以概括为：

**Persona × Dynamic Context → Behavioral Response**

即同一个居民在天气、拥堵、交通中断、价格、道路状态等外部条件变化时，如何改变自己的出行选择。

---

## 2. 研究目的

本研究希望实现：

1. 建立一个同时表示**居民长期个体属性**与**动态城市环境**的出行行为建模框架；
2. 使用 **DeepSeek V4 Pro** 作为统一的 foundation-model teacher，生成不同居民在不同环境扰动下的行为监督；
3. 将这种 **context-sensitive travel behavior** 蒸馏到轻量化 Traveler Agent 中；
4. 将蒸馏后的 agents 接入 MATSim，在 population scale 下运行；
5. 检验蒸馏后是否仍然保留个体异质性、环境敏感性和不同人口群体之间不同的行为响应；
6. 研究微观个体行为变化如何聚合成宏观的 mode share、traffic demand、congestion 和 network performance 变化。

---

## 3. 核心研究对象

### 3.1 Persona：相对稳定的个体属性

可能包括但不限于：

- 年龄；
- 收入；
- 职业；
- 家庭结构；
- 是否拥有汽车；
- 通勤特征；
- 出行习惯；
- 可达性或 mobility-related characteristics。

具体采用哪些属性由数据可得性和后续实验决定，本蓝图不预先锁定。

### 3.2 Dynamic Context：动态外部环境

重点考虑能够扰动居民出行行为的城市环境变化，例如：

- weather；
- 实时交通拥堵；
- 公交或轨道交通中断；
- transit delay；
- 道路封闭；
- 事故；
- 票价变化；
- 停车成本；
- 拥堵收费；
- 其他交通政策或异常事件。

核心不是单独研究某一种扰动，而是研究：

> **不同 Persona 面对相同 Context Change 时为什么产生不同的 behavioral response。**

---

## 4. 与现有研究的主要区别

### 4.1 与 MATSim 的区别

MATSim 本身已经是成熟的 agent-based transport simulation framework，可以表示大量独立 persons、plans 和交通方式，并通过评分与 replanning 等机制模拟居民出行选择。

因此，本研究**不声称 MATSim 缺少个体居民建模能力**。

本研究希望增加的是：

> **由 foundation-model behavior knowledge 蒸馏得到的、Persona 与 Dynamic Context 联合驱动的个体行为决策层。**

MATSim 主要承担城市级 population 和 transport system simulation，而本研究重点发展居民面对变化环境时的行为响应模型。

### 4.2 与 GTA 类 LLM Traveler Agents 的区别

GTA 等研究已经证明，可以利用人口统计 persona 和 LLM 生成活动计划、mode choice，并将这些行为用于大规模交通仿真。

本研究不重复“LLM 能否模拟 traveler”这一问题。

重点转向：

> **LLM Traveler 的行为能力能否被压缩到 lightweight agents 中，同时保留个体异质性和环境响应能力。**

### 4.3 与 MobCache 的区别

MobCache 的核心目标是提高 LLM-based human mobility simulation 的扩展性，通过 reasoning reuse、cache reconstruction 和轻量 decoding 等方式降低大规模 mobility generation 的成本。

本研究同样关注 scalability，但研究重点不同：

> **重点不是高效复用或生成 mobility trajectories，而是蒸馏 Persona × Dynamic Context 所决定的 behavioral response function。**

特别关注：

- 同一个居民在环境变化前后的行为差异；
- 不同 population groups 面对相同扰动时的响应差异；
- 蒸馏后 behavioral responsiveness 是否被保留；
- 个体响应如何进入交通系统闭环并影响 network state。

因此，本研究更强调：

**Behavioral Response Distillation / Behavioral Elasticity Preservation**

而不仅是 mobility generation efficiency。

---

## 5. DeepSeek V4 Pro 的角色

本研究所有 foundation-model teacher reasoning、behavioral labeling 和 teacher data generation **统一使用 DeepSeek V4 Pro**。

DeepSeek V4 Pro 的作用是：

> **离线提供复杂的人类出行行为推理监督。**

其输入由：

**Persona + Dynamic Context + Travel Situation**

构成，并输出相应的行为选择、选择概率或结构化行为信息。

随后将其行为知识蒸馏到轻量化模型中。

运行城市级仿真时，不再为每个居民实时调用 DeepSeek V4 Pro。

整体逻辑为：

**DeepSeek V4 Pro teaches → Lightweight Traveler Agents execute → MATSim simulates**

---

## 6. 使用工具

### 核心工具

- **DeepSeek V4 Pro**
  - foundation-model teacher；
  - behavioral reasoning；
  - teacher dataset generation。

- **MATSim**
  - synthetic population；
  - activity-based / agent-based transport simulation；
  - multimodal travel；
  - network interaction；
  - population-scale experiments。

- **Python**
  - 数据生成与整理；
  - Persona / Context 构造；
  - DeepSeek 调用；
  - 数据分析；
  - 实验自动化。

- **PyTorch**
  - knowledge distillation；
  - lightweight traveler model training；
  - behavioral policy learning。

### 可选数据与辅助工具

根据后续案例需求，可使用：

- OpenStreetMap；
- GTFS / public transit data；
- census / travel survey；
- weather / disruption / policy data；
- MATSim 公开场景。

具体城市和数据源在正式实验设计阶段确定。

---

## 7. 大致实验思路

实验不预先限定某一种具体模型架构，而围绕以下逻辑展开。

### 第一层：个体行为生成

构造具有不同 Persona 的 synthetic travelers，并为其施加不同 Dynamic Context。

使用 DeepSeek V4 Pro 获取 teacher behavioral responses。

### 第二层：行为蒸馏

利用 teacher data 训练 lightweight traveler model，使其学习：

**Persona × Context → Travel Behavior**

重点不是只复制单次 choice，而是尽可能保留环境变化前后的行为响应关系。

### 第三层：个体级验证

比较 teacher 与 distilled agent 在相同 Persona / Context 下的：

- choice consistency；
- choice distribution；
- context response；
- individual heterogeneity；
- counterfactual behavior。

重点检查：

> **当只改变环境变量时，student 是否能够产生与 teacher 类似的行为变化方向和响应幅度。**

### 第四层：Population-scale 验证

将 distilled agents 接入 MATSim。

观察大量居民面对同一城市扰动时形成的 population response：

**Dynamic Context Change → Individual Choice Changes → Population-Level Demand Shift**

### 第五层：交通系统闭环验证

进一步观察个体行为变化如何改变网络：

**Environment Change → Traveler Adaptation → Mode / Departure / Route Changes → Network State Change → New Travel Conditions → Further Traveler Adaptation**

从而研究 individual behavioral adaptation 与 city-scale traffic dynamics 之间的联系。

---

## 8. 重点评价内容

本研究的评价不应只停留在“student 与 teacher 的准确率”。

重点包括：

### Behavioral Fidelity

蒸馏模型是否能保留 DeepSeek V4 Pro 的基本出行选择规律。

### Behavioral Heterogeneity

不同 Persona 是否仍然表现出不同的出行偏好。

### Contextual Responsiveness

环境变化是否会合理改变同一个居民的行为。

### Behavioral Elasticity Preservation

当票价、延误、拥堵、天气强度等连续变化时，student 是否能够保留 teacher 的行为响应趋势和敏感程度。

### Population-Level Fidelity

大量个体聚合后，population-level mode choice、departure pattern 和 demand response 是否保持合理结构。

### Scalability

蒸馏后的 traveler agents 是否能够以显著低于直接 foundation-model inference 的成本运行于大规模 MATSim population。

### Network Impact

个体行为变化是否能够在城市交通网络中形成可解释的交通需求和拥堵变化。

---

## 9. 整体研究流程

```text
Research Question
      ↓
Define Persona + Dynamic Context
      ↓
Construct Behavioral Perturbation Scenarios
      ↓
DeepSeek V4 Pro Teacher
      ↓
Behavioral Teacher Dataset
      ↓
Knowledge Distillation
      ↓
Lightweight Traveler Agent
      ↓
Individual-Level Validation
      ↓
MATSim Integration
      ↓
Population-Scale Simulation
      ↓
Dynamic Context Perturbation
      ↓
Individual Behavioral Adaptation
      ↓
Population Demand Shift
      ↓
Network Traffic Response
      ↓
Evaluation & Research Findings
```

---

## 10. 预期研究贡献

### Contribution 1 — Context-Responsive Traveler Modeling

将稳定的 Persona 与动态的城市 Context 联合建模，使个体 traveler 能够针对城市环境变化产生不同响应。

### Contribution 2 — Behavioral Response Distillation

探索如何将 DeepSeek V4 Pro 中隐含的个体出行行为推理压缩为城市级可运行的轻量化 traveler agents。

### Contribution 3 — Behavioral Elasticity Preservation

从“是否复制 teacher choice”进一步推进到：

> **是否保留不同人口群体面对环境变化时的行为响应规律。**

### Contribution 4 — Population-to-Network Propagation

研究环境扰动如何经过：

**Context → Individual Behavior → Population Demand → Traffic Network**

形成城市级交通影响。

---

## 11. 本研究不强调的内容

本研究现阶段不以以下内容作为核心创新：

- 创造新的 LLM；
- 单纯提高 trajectory generation speed；
- 单纯证明 LLM 可以模拟居民；
- 单纯增加更多人口属性；
- 单纯提高 mode-choice prediction accuracy；
- 开发新的微观车辆驾驶模型。

核心始终是：

> **将 foundation-model 中 Persona-conditioned、Context-sensitive 的出行行为响应能力蒸馏为可大规模部署的 traveler population，并研究这种行为能力在动态城市交通环境中的表现。**

---

## 12. 一句话研究定义

> **DeepSeek V4 Pro learns how different people respond to changing urban environments; knowledge distillation compresses this behavior into lightweight traveler agents; MATSim evaluates whether these agents preserve heterogeneous behavioral adaptation at city scale.**

中文：

> **让 DeepSeek V4 Pro 教会模型“不同的人面对不同城市变化会怎么调整出行”，再把这种行为能力蒸馏成可以在 MATSim 中大规模运行的轻量居民 Agent。**

---

## 13. 当前研究定位参考

- **MATSim**：成熟的 multi-agent transport simulation framework，以 persons、plans、scoring 和 replanning 为核心。
- **GTA: Generative Traffic Agents for Simulating Realistic Mobility Behavior (2026)**：基于 persona 的 LLM traveler simulation，并与 SUMO 结合验证城市级 mobility behavior。
- **MobCache: Mobility-Aware Cache Framework for Scalable LLM-Based Human Mobility Simulation (2026)**：聚焦 LLM-based human mobility simulation 的 scalability，通过 reasoning cache/reconstruction 与 lightweight decoding 提升效率。

本研究将在这些工作的基础上，将重点进一步放到：

> **dynamic environmental perturbation × individual persona × behavioral response distillation × population-scale transport feedback**

这一完整链条上。
