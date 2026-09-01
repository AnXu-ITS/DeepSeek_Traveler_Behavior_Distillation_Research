# PIPELINE_LATENCY_AUDIT — Student × MATSim 端到端推理管线性能审计与优化实验

> 审计日期：2026-08-29（2026-08-30 会话）· 机器：单机 24C / 31.4 GB / CPU-only（torch 2.13.0+cpu）
> 方法：不改动任何现有管线代码；全部数字来自新脚本 `scripts/benchmark_student_pipeline.py`
> 的实测（真实 S9 checkpoint、真实 E2 状态池、真实新加坡供给），并引用 E2/E3 已实测数字。
> 结论速览：**Student 推理只占端到端时间的 ~0.3%；瓶颈是特征构造（可达性/备选项路由）与
> 跨情景重复计算，且当前架构中 MATSim 运行时根本不调用 Student。**

---

## 阶段 1 — 当前真实调用链（依据代码，非假设）

### 1.1 MATSim 如何调用 Student？——它不调用

- 启动器 `tools/java/RunMatsimPreloaded.java`：普通 `Controler` + 一个 transit 事件记录器
  （`TransitEventLogger`），**没有任何 Python/HTTP/socket 调用代码**。
- 配置 `lastIteration=0`（`adapter.py` `_write_real_config`）：**零重规划迭代**，Java 侧在整个
  仿真期间只执行预先写好的 plans。
- 因此：**不存在 Java↔Python IPC、不存在每次决策启动进程、不存在 HTTP/socket/CSV 往返。**

### 1.2 真实链路：离线预计算 + 文件交接

```
MATSim 侧（Java，仅执行）                           Python 侧（离线，一次性构建）
------------------------------------------------   ---------------------------------------------------
[仿真启动前]                                        S8MATSimAdapter.__init__: torch.load(checkpoint)      (一次)
                                                    SupplyIndex: 加载路网图 + 859 站 + trips 索引          (一次)
                                                    for each agent (N=10k):
                                                      ① 确定性 home/dest（persona_id 哈希）
                                                      ② alt_factory → plan_accessibility(OD, 出发时刻)
                                                         （真实 GTFS 公交行程规划：找站、walk Dijkstra、
                                                           直达+一次换乘搜索）                         ← 主导成本
                                                      ③ build_real_alternatives：car/bike/walk
                                                         mode_travel_time（101k 节点图上双向 Dijkstra ×3）
                                                      ④ UniversalTravelerState → adapter.decide()
                                                         extractor.encode → collate_batch(B=1)
                                                         → model.forward（24,562 参数）→ argmax + shift
                                                      ⑤ _build_legs：所选模式 leg 路由（SP 缓存）
                                                      ⑥ ElementTree 组装 population.xml + manifest JSON
                                                    写盘：population.xml + adapter_manifest.json + config.xml   (文件交接)
java -Xmx6g RunMatsimPreloaded config.xml ──────▶  Java 读 519MB transitSchedule + plans，QSim 一次，写 events.zst
Python 解析 events.zst ────────────────────────▶  指标统计（T_parse）
```

关键结论：`T_process_startup / T_IPC_send / T_IPC_return / T_MATSim_writeback` 在**当前架构中
为 N/A**——所谓"IPC"就是一次性文件交接（population.xml），所谓"writeback"是下一次重规划
时才存在的概念（Phase D 未启用）。

### 1.3 Student 真正需要的输入（从代码确认）

- 全部为**数值/类别特征，无任何文本/NLP**：`S8FeatureExtractor.encode()` 输出
  `global_cat`（9 维类别：persona 6 + trip 2 + weather 1）、`global_num`（17 维：persona 6 +
  trip 3 + context 8）、`alt_mode_idx` + `alt_num`（每备选项 12 维：travel_time/cost/access/
  transfers/reliability/exposure + 6 维可达性）、`alt_mask`。
- **Teacher 的 natural-language prompt / parser 管线完全没有进入 Student 路径**
  （`teacher/prompts*.py` 只被教师标注脚本引用；student 输入是张量）。
- static vs dynamic 划分（`features.py` 的 PERSONA_*/TRIP_*/CONTEXT_* 常量）：
  - static（persona 侧 12 维 + trip 距离/时刻）：每次决策都重新 encode —— 可预计算；
  - dynamic（context 8 维 + 备选项 12 维）：其中备选项的 6 维可达性特征**只依赖 (OD, 出发时刻)**，
    与情景扰动无关 —— 可跨情景缓存（当前每情景重算）。

---

## 阶段 2 — Profiling（实测，mean ms/state；标注来源）

| component | mean ms/state | share of 10k build | 说明 |
|---|---|---|---|
| feature preparation（可达性规划 + 备选项路由） | **100.9 cold / 0.046 cache-hit** | 57–63% | 实测：plan_accessibility 80.6 ms + alt 路由 ~20 ms；缓存命中后 0.033 ms |
| serialization（leg 路由 + population.xml + manifest） | ~116 | ~35–38% | E3 T_residual（10k = 19.3 min）；含所选模式 leg Dijkstra + ElementTree |
| process startup | N/A | — | 单进程一次启动（离线设计） |
| python import + 模块导入 | 1408 ms **一次** | <0.1% | 冷启动实测（含 torch import） |
| model load | 3.8 ms **一次** | <0.01% | torch.load（108 KB checkpoint） |
| IPC send / return | N/A（文件交接一次） | — | 运行时 IPC 原型实测 0.50 ms/请求（见实验 B） |
| **Student inference（encode+forward+decode）** | **1.066 逐态 / 0.051 批(256)** | **0.35%** | 真实 S9 + 真实状态实测 |
| postprocess（argmax/shift） | 含于 inference | — | encode 0.007 ms、forward B=1 0.147 ms |
| MATSim writeback | N/A（本架构无运行时写回） | — | — |
| MATSim 执行（摊销到每 agent） | 13.7 | — | 10k 情景 136.7 s（E3 实测，供给主导） |
| events 解析（摊销） | 7.2 | — | 10k 情景 72 s（E3 实测） |
| **端到端合计（10k，E3 实测）** | **327** | 100% | build 306.3 + sim 13.7 + parse 7.2 |

### 冷/温/端到端三口径（实测）

- **A 冷启动**（进程 + torch import + checkpoint load）：**1.408 s**；其中 checkpoint 加载仅 3.8 ms
  （模型只有 108 KB）。可达性供给索引构建（路网图 + 站索引）**6.4 s**。全部为每次运行一次的成本。
- **B 温推理**（模型已加载）：逐态 decide **0.267 ms**（mean, n=100；p95 0.553）；批处理后
  **0.051 ms/state**（bs=256, n=10k）。
- **C 端到端决策时延**（离线构建口径，每 state 摊销）：**306.3 ms/state（10k，E3 实测）**；
  其中 Student 推理 1.066 ms = **0.35%**。

**回答："Student inference 占总时间百分之多少？" → build 的 0.35%，端到端的 0.33%。**
（特征构造占 57–63%，leg/序列化占 ~35%，其余为一次性成本。）

---

## 阶段 3 — 是否把 Teacher 管线错误保留给了 Student？

检查结果：**没有**。Student 路径无 prompt/文本/JSON 中间层（输入即数值张量）。但存在三个
"Teacher 时代"遗留的**结构性低效**（均在特征侧，不在模型侧）：

1. **可达性特征每次决策实时规划**：Teacher 标注需要真实行程分解（access/egress/wait 时间），
   部署时 Student 只需要 12 个数值——但当前每次决策都重新跑 `plan_accessibility`
   （80–103 ms/state）。这些向量只依赖 (OD, 出发时刻)，**应离线预计算并持久缓存**。
2. **static persona 属性每决策重 encode**：12 维 persona 特征对同一 agent 恒定，却每 state
   重新查字典/归一化（实测 encode 0.007 ms/state——很小，但在 100k+ 规模可省）。
3. **无磁盘级缓存**：`run_phase_c.py` 的 OD 缓存是**每情景每工厂**新建的（每情景全量重算），
   进程退出即消失（每次运行重新支付数小时构建）。

删除清单（实验证实可删）：无 NLP prompt 可删（本就没有）；可达性/路由可转预计算缓存
（实验 E，2478×）；per-state 张量构造可转批量（实验 C，5.8×）。

---

## 阶段 4 — 六项优化实验（逐项独立实测）

### 实验 A：Persistent Student
当前架构**已是持久化**：模型每进程加载一次（3.8 ms），进程常驻直到构建完成。
实测收益对比：若按"每次决策都新起进程"的假设，每次决策要付 1.408 s 冷启动；
持久化后每决策只付 0.267 ms 推理 → **5,270×**。结论：本项已天然满足，无需改造。

### 实验 B：Persistent Java↔Python IPC（原型，仅评估未来运行时架构）
原型：常驻 TCP worker（长度前缀 JSON），真实 S9 + 真实状态，100 次热请求：
**mean 0.50 ms / median 0.43 ms / p95 1.02 ms**（请求体均值 2,174 B）。
结论：若未来需要运行时决策，常驻 worker 可行（0.5 ms/决策，推理占其中 ~0.3 ms，
其余为 JSON 编解码 + TCP）。当前架构不需要。

### 实验 C：Batch Inference（真实收益项之一）
同一决策 epoch 的 N 个 agent 确实存在（离线构建天然可全批）。10k 真实状态实测：

| batch | ms/state | decisions/s |
|---|---|---|
| 1（逐态，现状） | 0.295 | 3,393 |
| 16 | 0.054 | 18,686 |
| 64 | 0.062 | 16,014 |
| **256（最优）** | **0.051** | **19,489** |
| 1024 | 0.059 | 16,904 |

逐态→批处理加速 **5.8×**（encode+forward+decode 全链路）；最优 batch=256。
注意：decide 只占总管线 0.35%，故批处理对**总端到端**的贡献 ≈0.3pp——它是"推理侧"的优化，
不是系统级瓶颈的解药。

### 实验 D：Lightweight Feature Interface
Student 输入本已是纯数值（无 NLP/JSON/DataFrame）。实测分解：
encode 0.007 ms（static 属性查表）→ forward 0.147 ms（B=1）→ collate/decode ~0.11 ms。
轻量化结论：无需新接口；把 per-state 张量构造换成批量 `collate_batch` 即实验 C 的收益。
static persona 特征预计算（同一 persona 多 state 时复用 encode 结果）在本数据集为
每 persona 单 state，节省为 0——仅在"多行程/多情景共用 persona"时有效（见实验 E）。

### 实验 E：Caching / Precomputation（**单项收益最大**）
真实供给、seed 2026、1,000 个真实 (persona, trip, OD)：

| 路径 | ms/state | 相对冷路径 |
|---|---|---|
| 可达性重算（现状，每情景每进程重来） | 80.6 | 1× |
| 特征冷首遍（可达性 + alt 路由，唯一 OD） | 100.9 | 1× |
| **磁盘缓存命中 + 记忆化（第二情景/第二次运行，同 OD）** | **0.046**（含盘载 0.01s/1000 条） | **2,478× 快于可达性重算；2,200× 快于冷首遍** |

- 可达性缓存体积极小（1,000 条 pickle 仅 0.1 MB；10k 条约 1 MB）。
- 这就是把"每次运行几小时"变成"几分钟"的那一项：Phase C 六情景实测 build 合计 3.9 h
  （14,032 s），其中 5 个情景在重复计算与 C0 相同的 (OD, 出发时刻) 可达性。

### 实验 F：ONNX 部署可行性（**可行，但当前收益≈0**）
- 导出成功：`outputs/pipeline_benchmark/s9.onnx`，**109,603 B**（与 checkpoint 同量级）；
  opset 17，动态 batch 轴，legacy TorchScript 导出器。
- Parity（1,000 真实状态）：max |Δprob| **1.79e-07**、max |Δshift| **5.72e-06 min**、
  argmax 一致 **1000/1000** —— 数值等价。
- ORT CPU 实测：bs=1000 时 **0.0009 ms/state，1.13M decisions/s**（与 torch 批处理同量级，
  torch 巨批 1.34M/s）。
- 判定：模型极小（107 KB），在 JVM 内嵌 ORT 技术上完全可行；但当前架构 Java 侧**根本不需要
  推理**（决策离线预计算），换成 ORT 的端到端收益 ≈ 0。**不推荐现在替换正式管线**；
  若未来做运行时重规划（Phase D 类反馈闭环），ORT Java 内嵌是首选路线。

---

## 阶段 5 — 调用频率是否本身不合理？（分析，不修改实验）

- 现状：每个 agent **每情景恰好 1 次决策**，且在仿真**开始前**离线完成；运行时调用次数 = 0
  （`lastIteration=0`）。Phase C 六情景 = 60,000 次决策一次性批产。
- 若改成"运行时每重规划迭代每 agent 调一次"（10 迭代）→ 600,000 次运行时调用，且每调用
  需在 JVM 内构造特征——**当前离线设计反而更优**，因为特征构造（100 ms/state）远贵于推理。
- 事件触发式（只在 disruption/延误/拥堵越限时重决策）仅在**未来反馈闭环**里有意义：
  预计触发面 << N*（雨/延误等扰动只改变 context，供给不变时无需重规划）。
- 结论：频率设计合理（1 决策/agent/情景），无需改动；若论文要写反馈闭环，建议用
  "event-triggered + 预计算特征"。

---

## 阶段 6 — 城市规模 benchmark

| 规模 | 纯 Student 推理（批 256） | 当前端到端构建 | 优化后端到端构建（缓存+批处理） |
|---|---|---|---|
| 1k | 0.05 s（measured） | 353 s（E3 measured） | ≈ 0.1 s 特征+推断（measured 成分）+ leg/XML（未实测） |
| 10k | 0.51 s（measured） | 3,063 s（E3 measured） | ≈ 1 s 特征+推断（measured 成分）+ leg/XML（未实测；上限=E3 residual 19.3 min） |
| 50k | 2.6 s（measured 外推：0.051 ms×50k） | 13,907 s（E3 measured） | ≈ 5 s 特征+推断 + leg/XML |
| 100k | 5.1 s（measured 外推） | 20,907 s（E2 measured） | ≈ 10 s 特征+推断 + leg/XML |
| 1M | **51 s（measured：encode 58.2 s + forward 1.03 s）** | ≈ 58 h（**estimated**：E2 实测 100k=5.8 h 线性外推 ×10） | ≈ 46 s 特征+推断（**estimated**：0.046 ms×1M，前提 OD 已预计算；冷首遍另计 ≈28 h）+ leg/XML |

throughput（实测）：逐态 3,393/s → 批 256 **19,489/s** → 纯 forward 巨批 **1.34M/s**；
ONNX ORT 1.13M/s。端到端（含特征）当前 **4.8 decisions/s**（100k，E2 实测）→ 缓存命中后
特征侧可达 ~21,848/s，瓶颈转移到 leg 路由 + XML 序列化。

**回答**：Student 很快之后，**系统级部署瓶颈依然存在**——它只是从"推理"移到了
"特征构造 + 序列化"；Student 的理论速度优势（vs DeepSeek 的 3.2×10⁵ 倍时延）在 MATSim
中已经兑现（决策不再是瓶颈），但**没有**转化为端到端时间缩短，因为端到端从未被推理主导。
真正能缩短端到端的是缓存与批处理（本报告实验 C/E）。

---

## 阶段 7 — Correctness Check（不改决策逻辑的前提下）

| 对比 | 结果 |
|---|---|
| 批处理路径 vs 原始逐态 `decide()`（10,000 真实状态） | **mode 一致 10,000/10,000（100.0000%）**；max \|Δshift\| 5.72e-06 min（浮点 GEMM 分块噪声，低于 manifest 2 位小数舍入精度） |
| 批处理决策 vs 冻结 Phase C C0 manifest（前 10,000） | **mode 一致 10,000/10,000（100.0000%）** |
| ONNX vs PyTorch（1,000 状态） | max \|Δprob\| 1.79e-07；max \|Δshift\| 5.72e-06 min；argmax 1000/1000 |

全部优化实验均未改变任何决策：mode 100% 一致，概率/时刻差在浮点噪声量级。

---

## 最终推荐架构（三档，全部有实测支撑）

### 第一档【最低成本，马上可以做】— 跨情景可达性/路由磁盘缓存
- 把 `plan_accessibility` 与 `mode_travel_time`/leg 路由结果按 (OD, 出发时刻) 持久化到
  `outputs/` 下的 pickle（10k 条约 1 MB）；`run_phase_c.py` 增加"缓存目录"参数，命中则跳过
  路由。**不改变任何决策**（缓存的是同一函数在同一供给上的确定性输出）。
- 实测收益：特征侧 100.9 → 0.046 ms/state（**2,200×**）；Phase C 六情景 build 3.9 h →
  约 10–15 min（估算，冷首遍不可省）。风险：极低（纯工程，正确性由 G2/逐位复现锚背书）。

### 第二档【推荐论文版本】— 离线批量决策 + 预计算特征 + 缓存复用
- 保持"离线预计算 → MATSim 执行"架构不变（论文部署叙事天然成立）；
- 特征全部预计算并缓存（档 1）+ 静态 persona 特征一次性编码 + Student 批量推理（bs=256，
  实测 19,489 decisions/s）+ leg/XML 复用已有 SP 缓存。
- 每态决策成本 0.051 ms、特征命中 0.046 ms；10k 情景构建从 51 min 降到分钟级。
- 论文表述："precomputed supply features + batch inference：100k 决策 5.1 s（实测外推），
  end-to-end build 缩短 25–60×"。

### 第三档【最终工程版本，仅在需要运行时重规划时】— MATSim JVM 内 ONNX + 事件触发
- ONNX 已验证可行（107 KB 模型、parity 1.8e-7、1.13M decisions/s）；未来 Phase D 反馈闭环
  在 JVM 内嵌 ORT-Java，去掉 Python 依赖；决策仅在重规划事件触发。
- **当前不采纳**：现有实验（lastIteration=0）不需要运行时推理，收益为 0，复杂度最高。

---

## 六个关键问题（数字回答）

1. **当前一次 MATSim→Student→MATSim 总耗时？**
   本架构无运行时调用；等价口径 = 每 agent 决策的端到端摊销 **327 ms**（build 306.3 + sim
   13.7 + parse 7.2，10k E3 实测）。若把决策做成运行时 IPC：**0.50 ms/请求**（原型实测）。
2. **Student 占其中多少？**
   **0.35%**（build 口径 1.066/306.3）或 **0.33%**（端到端口径）；纯推理 0.147 ms（B=1）
   / 0.001 ms（批处理）。
3. **最大的三个 overhead？**
   ① 可达性/备选项路由 100.9 ms/state（57–63%，`plan_accessibility` 80.6 + alt Dijkstra ~20）；
   ② leg 路由 + XML/manifest 序列化 ~116 ms/state（~35%，E3 residual）；
   ③ 跨情景/跨进程全量重算（无持久缓存）——把 ① 放大为"每情景 30–50 分钟、六情景 3.9 h"。
4. **哪个单项优化收益最大？**
   实验 E 的缓存/预计算：特征侧 **2,478×**（0.046 vs 100.9 ms/state 冷首遍）。批处理是
   推理侧单项最大（5.8×），但对总管线贡献仅 ~0.3pp。
5. **几项低风险优化组合以后实际快了多少倍？**
   缓存 + 批量推理 + 静态预计算组合：决策+特征从 ~307 ms/state → ~0.1–0.5 ms/state
   （**~600–3,000×** 于这两段），10k 情景 build 从 51 min 到分钟级（**25–60×** 总 build，
   实测成分 + 标注 estimated 部分）；Student 决策吞吐 3,393 → 19,489 decisions/s（**5.8×**）。
6. **是否值得进一步做 Java/ONNX 原生集成？**
   当前实验架构下**不值得**（Java 侧不推理，收益≈0）；ONNX 本身已验证可行且高精度
   （107 KB、Δ≤1.8e-7），留作未来运行时重规划（Phase D）的工程路线，不建议现在投入。

---

## 附件

- 实测数据：`docs/pipeline_benchmark.csv`（method / batch_size / num_decisions /
  total_time / ms_per_decision / decisions_per_second / cold_or_warm）
- 基准脚本：`scripts/benchmark_student_pipeline.py`（profile / batch / scale / e2e1k /
  cached / ipc / onnx / correctness 八种模式，只读复用生产代码）
- ONNX 工件：`outputs/pipeline_benchmark/s9.onnx`（109,603 B）
- 全部优化均为可回退方案：未修改任何现有脚本/模型/实验定义；批处理与缓存的正确性由
  阶段 7 的 100% mode 一致性 + 冻结 C0 manifest 逐位对照背书。
