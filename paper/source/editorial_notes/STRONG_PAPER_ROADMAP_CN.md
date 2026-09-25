# AIT 强稿路线：从响应失配诊断到可验证、可复用的方法贡献

**适用稿件**：Beyond Static Imitation: Auditing LLM-Derived Traveler Responses for Transport Simulation  
**证据核查日期**：2026-09-24  
**版本状态**：已完成有据可依的文字重构与一个解析机制补强；未执行新训练、新 LLM 查询、新受试者研究或新 MATSim 实验。此文件中的新增实验均为建议方案，不是已完成结果。  
**仓库核查基准**：`AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research`，commit `4e67286ce72680a9e3a916a3fb7e97b1548b53f7`。仓库仍为访问受控状态。

## 1. 核心判断

把论文做成 AIT 中有辨识度的强稿，最有价值的路线不是把 4.97% 的相对改善包装得更响亮，也不是无限增加城市、模型和网络迭代。应建立一条完整证据链：

**定义决策相关的响应 → 在严格划分下检验泛化 → 定位失配发生在哪一层 → 在独立证据上验证一个可行的修复 → 发布可复用的基准与评估程序。**

现有稿件已经有跨 Teacher、stated choice 与 execution 的独特组织方式；缺口是“诊断之后的可验证改进”以及“新干预下是否仍成立”。这两点比再加一项内部指标更能改变论文层级。

“Overqualified”不是 AIT 的正式分类，也不能由已发表样本推导录用概率。这里的目标是：核心结论不依赖单一 API 实例、单一划分、单一默认输入或一项选择性结果；读者能够复用评估程序，并能依据结果作出不同于只看 imitation accuracy 的模型选择。

## 2. 对标证据：哪些做法实际存在，哪些不能被推断

### B1. ORThought：基准、受控方法、错误分类和成本一起构成贡献

用户提供的版本：arXiv:2508.14410v3，2026-04-18，PDF 32 页；AIT 发表记录对应 2026，100059。主要定位：PDF 第 10–14 页，Tables 2–6，以及 benchmark 与 error-analysis 小节。

可观察做法：LogiOR 与其他数据集共同评估；按问题类别与复杂度拆分结果；控制模块消融；报告 token 成本；给出错误分类。Table 3 的 LP 项、Table 5 的部分类别并非全胜。

对本稿的启示：不需要声称某个 Student 在所有层级获胜。需要把“为什么排序变了、怎样据此调整模型”做成系统性结果。不可推断：这些特定数据集数量或改善幅度是 AIT 接收门槛。

来源：https://arxiv.org/abs/2508.14410 ；DOI 10.1016/j.ait.2026.100059。

### B2. TrajFlow：评价目标与指标匹配，优势与成本同时报告

用户提供版本：arXiv:2501.14266v2，2025-08-02；核对的 AIT 记录为 100076。不要与另一个同名 TrajFlow 论文混淆。主要定位：PDF 第 12–17 页，尤其第 16 页 Table 2。

可观察做法：比较 GRU/CDE 与离散/连续 flow 的组合；同时考察预测与占用密度；密度结果配 RMSE、CRPS；与六个基线比较，并列训练和推理成本。论文没有掩盖连续配置的高成本。inD 的这项实验使用单个 recording 的划分，不应改述为已验证所有交叉口的跨域迁移。

对本稿的启示：将 static KL、response error、human response bias 和 execution gap 分别对应明确用途；报告取舍，而不是合成一个任意“总分”。

来源：https://arxiv.org/abs/2501.14266 ；https://www.sciencedirect.com/science/article/pii/S3050860626000311 。

### B3. Applying large language models to travel satisfaction analysis：诊断后给出修复

用户提供的是出版 PDF：AIT 8 (2026), 100079。主要定位：Section 3.1、Section 3 的模型与 few-shot 设计、Section 4.1、Tables 3–4。

可观察做法：分析样本 874 人；四个 LLM；zero-shot/few-shot 与传统基线比较；改变支持集规模；五次重复。研究明确披露样本偏向及其范围，但仍将文章组织为“问题—干预—验证”，而非一份局限清单。

对本稿的启示：Teacher/human 失配的发现需要一个在未参与修复的数据上评估的矫正实例。不能直接照搬其随机划分作为本稿重复测量数据的充分划分方式；本稿仍需 respondent/persona 分组隔离。

来源：https://www.sciencedirect.com/science/article/pii/S3050860626000347 ；DOI 10.1016/j.ait.2026.100079。

### B4. Driving as a screening aid：推断单位决定划分，用途措辞决定承诺

用户提供的 arXiv:2507.05463v3 标题为 “Driving as a Diagnostic Tool”；核对的 AIT 发表标题为 “Driving as a screening aid”，100067。主要定位：PDF 第 15 页，driver-level leave-five-drivers-out 与 random-split 对照。

可观察做法：按驾驶员分组测试，报告重复划分。标题变化是可观察事实；没有审稿信证据，不能声称是编辑要求或导致录用。

对本稿的启示：persona-disjoint 回答新 persona，intervention-disjoint 才回答新干预。execution audit 与 disruption forecasting 也应在题目和论断层级上区分。

来源：https://arxiv.org/abs/2507.05463 ；DOI 10.1016/j.ait.2026.100067。

### B5. Toward LLM-agent-based modeling…：贡献类型决定证据类型

AIT 1 (2025), 100001；用户版本 arXiv:2412.06681v2。该文是 conceptual framework 加 proof of concept，不应作为方法论文降低实证门槛的依据。

本稿可以回应其混合建模方向，但自己的价值必须具体到：压缩何种对象，怎样检验，失配怎样改变模拟解释。

来源：https://arxiv.org/abs/2412.06681 ；DOI 10.1016/j.ait.2025.100001。

### 其余上传文献的角色

Roadway-safety LLM 综述对应 AIT 2025, 100004，可用于界定领域而非推导实验阈值。Agentic Traffic Intelligence 对应 AIT 2026, 100057，适合作为可执行工作流与可追踪性对照。`ssrn-4913956` 和 `ssrn-5105959` 在本次核查中未确立 AIT 发表身份，应标为辅助预印本文献，不能统一称为“九篇 AIT 已发表强稿”。

### 相邻工作必须进入贡献边界

Liu, Li & Yin 的 Transportation Science 论文在 Swissmetro 上研究 persona-based empirical alignment；Natterer 等的 TR-C 论文代理的是 ABM 网络输出；Sobolev training 与 relational distillation 表明“保留响应/关系”不是全新的通用学习原则。本稿可识别的对象是 **LLM 数值决策函数的有限响应及其跨层级传递**。

另有 CALM 预印本（arXiv:2609.22252v1）已涉及校准、干预测试和可回放的 choice–network 系统。它是新的重叠线索，不是已确立的 AIT 标准。避免将“LLM + choice + network + intervention”本身写成首创。

## 3. 不应继续背负的错误负担

**仓库不是“后来实验目录不存在”了。** 当前 README、REPRODUCIBILITY、verification.json 明确列出 prepared benchmark、12 个受控 neural fits、modular records、September scripts 和 Helsinki 汇总。当前 manuscript 的对应句子过时，已经改掉。仍然受限的是公开访问、参与者级重抽样、完整大型事件档案与历史环境，而不是所有结果都无从核查。

**Teacher 重复性分析不是零。** Supplement S2 已有 373 states、350 pairs 的均值/中位数/省略重复及重抽样检查。下一步应补“同一 response estimand 的自差异参照”，而不是重做一遍已有分析。

**不能用 Teacher 单状态 SD 大于 0.34 pp，直接否定 loss 改善。** 两者统计对象、聚合方式及配对结构不同。是否稳定必须看同口径配对估计与目标重聚合敏感性。

**selected_mode 频率不自动构成 probability calibration。** 如果 selected_mode 是 argmax，那么重复频率本来就不需要等于自报概率向量。要区分输出重复性、模型内采样一致性和相对于独立人类选择的校准。

**多 iteration、多城市、大模型并非无条件越多越好。** 只做 execution audit 时，一次固定行为迭代是有效隔离设计；只有提出反馈、均衡或实际供给中断的结论时，才需要对应证据。

## 4. P0：投稿前必须关闭的事实与溯源问题

| 工作 | 应交付的证据 | 不能用什么替代 |
|---|---|---|
| Teacher 身份与时间 | 各采集批次的实际 UTC 时间、prompt/hash、请求字段、返回 model 字段及可用 provider revision；冻结 targets 与 checkpoint | 不能把随机种子日期当请求时间，也不能把 API alias 当 immutable model |
| 伦理与同意 | 适用机构/审批或豁免依据、日期与编号（如有）、电子同意流程、允许的数据使用/共享范围 | 不能杜撰编号、补写从未取得的批准，或将匿名问卷自动等同豁免 |
| 分析边界 | 哪些结果是探索性，哪些预先定义；新增修复数据的 train/validation/test 隔离 | 已反复查看的人类测试结果不能继续作为全新修复的未见证据 |
| 可复现范围 | pinned commit、依赖版本、checkpoint/target hashes、从新环境执行的 command + expected output | 仓库存在、manifest passed 与完整独立复现不是同一件事 |

DeepSeek 官方已公告该 alias 自 2026-09-14 04:00 UTC 起路由到 V4.1-Flash，直至后续 Pro 发布。因此可以确认 alias 不具不可变性；当前能读到的 campaign summary 仍返回 alias，不能据此确认实际执行后端。文字已经降为 contemporary API reference；更具体的 attribution 必须等真实日志。

伦理文本中保留了作者原有 statement，并加入不进入 PDF 的 `AUTHOR ACTION E1`。这不是问题已解决的声明。

## 5. P1-A：把 benchmark 从“新人”提升到“新干预”

### 设计

建立共享、冻结的状态与干预 manifest，将 persona 轴和 intervention 轴独立管理。至少区分：

- 现有 persona holdout：作为同分布参照。
- 未见强度：训练和测试的 delay/fare/access 强度区间不重叠；插值与外推分开。
- 未见组合：训练单因素，测试组合；直接报告已有的 interaction metric。
- 未见干预类型：leave-one-family-out；被留出的 endpoint、pair、auxiliary target 和 checkpoint adaptation 都不得泄漏。

不是每个“未见类型”都能合理学习。如果模型训练从未见过某个语义类别，其失败应解释为开放支持外推，而不是把一个不可识别的问题交给 Student 后宣布失败。优先选择模型输入中已定义、其属性有训练覆盖的干预轴。

### 对照与公平性

核心对照保留 Soft KL、已有 signed-response、direction+magnitude 与 MNL-S。SA-Student 继续作为部署模型单列，不进入只改变 loss 的 matched ablation。增加直接平方 response penalty 是用于检验 loss 形式的低成本可选项，不是默认必须。

训练和选模使用完整交叉：**endpoint/response objective × static/response validation criterion**。每个单元共享数据、初始化配对、训练预算和调参预算。响应选模可附带预先固定的 endpoint 容忍度，但不能用 test 决定容忍度。报告 static–response Pareto 关系。

新增一个可冻结、与原 Teacher 来源不同的 Teacher，先只复现 controlled benchmark；无需把所有 survey/MATSim 实验重复一遍。两个 Teacher 支持跨实例稳健性，不意味着普遍适用于所有 LLM。

### 样本与停止规则

不要凭空规定“必须 60 personas、10 seeds”。先用独立 pilot 得到 persona-level paired difference 方差，以具有实际意义的置信区间半宽制定预算，再冻结正式实验样本量、种子和分析方案。增加 seed 不能替代 persona 覆盖；增加大量同模板 synthetic persona 也不能提供人群代表性。

可用规划近似为 n ≈ (1.96 s_d / h)^2，其中 d 是 persona 层配对差、h 是目标半宽；它只用于 pilot 预算，正式分析保留分组和跨种子结构，不按这条近似公式宣称最终显著性。

**完成标准**：能完整说明哪些干预上改善、哪些无改善、哪些排序翻转，且结论不靠选择一个 seed/强度/Teacher。不是要求每格都显著。

## 6. P1-B：关闭优化解释的替代假说

当前 gradient cancellation 是真实数学性质，不是应删除的“不好看内容”。已存在 signed-response 对照，应回答拆成 direction+magnitude 是否值得。

正式协议应包含 response 权重曲线、loss 分量尺度/梯度量级、同预算调参，以及两个 validation-selection 规则。改变公式、归一化或 timing 权重后必须作为新训练版本，不能把旧模型结果贴到新公式下。

Teacher 自一致性应在同一 state pair 上比较独立重复聚合：Δp_T,A 与 Δp_T,B；两组使用匹配的重复数。已有 3/5 次重复可作有限诊断，但不宜宣称精确 noise ceiling。保持同一组目标重抽样供所有 Student 使用，分别报告 target uncertainty 与 training uncertainty。

MNL-S 检查应明确 utility identification 和正则化约定，报告有限、与任务相关的 coefficient contrasts、time/cost/access 方向与数值扰动曲线。不要因“线性”就自动声称经济可解释，更不要在无可识别时间/成本比时输出 VOT。

**完成标准**：排除“只是挑了一个权重/选模规则/噪声实现”的主要解释；即使 split loss 无独立优势，论文仍能以 robust response evaluation 成立。

## 7. P1-C：增加一个独立的人类响应修复实验

这是从“有趣诊断”升级为“可用方法”的高收益环节，借鉴 B3 的问题—改进—验证逻辑，而非照抄 few-shot 技术。

### 先把任务对齐

新任务中，人类、Teacher 和 Student 接收一致的数值 time/cost/access/waiting 信息、明确的 SGD/CNY、相同选项含义及可用性。明确 car 是私家车驾驶还是也包括 taxi/ride-hailing。随机化任务顺序并平衡 baseline/perturbation 顺序，记录实际执行配置。

对旧 Shanghai 数据补充 delay-intensity sensitivity，可以说明结论对编码的依赖；它不能补救原问卷未提供数值这一事实。若修改了 Student 的输入定义或归一化，需要相应重训，不能只在评估时改变含义。

### 修复必须可被证伪

建议选一个明确问题，例如 accessibility response 反转。比较原 MNL-S、一个有理论约束的 utility/calibration 版本，以及同等人类标注预算下的 conventional choice baseline。训练/校准样本与最终 respondent holdout 隔离；所有用到 human labels 的模型须明确标注，不能和零 human-training 模型混作纯粹方法公平比较。

修复成功的标准不是强迫模型继续最接近 Teacher，而是：在独立人类数据上改善预先指定的 response error，同时完整报告 Teacher fidelity、baseline calibration 和其他干预上的代价。若 Teacher 本身错了，更忠实不应是唯一目标。

既有 Singapore/Shanghai 数据已经参与问题发现。针对这些已观察到的失配调模型后，必须使用新受试者/未使用任务，或建立严格、事先固定的 nested procedure；不能把同一结果再次称为未见验证。

**完成标准**：至少一个通过审计定位的问题，在独立证据上得到可重复改善；同时保留失败与迁移边界。不能只靠 prompt 的几个展示案例。

## 8. P1-D：让 transportation 贡献超过“行为模型套模拟器”

保留目前 perceived-delay-only 作为控制臂，再做一个真实供给变化的四臂设计。

| 臂 | 行为输入 | 物理网络/时刻表 | 识别的部分 |
|---|---|---|---|
| 基准 | 基准 | 基准 | 比较起点 |
| 感知单变 | 扰动 | 基准 | 现有执行传递问题 |
| 供给单变 | 冻结基准行为 | 扰动 | 固定行为下的物理执行影响 |
| 联合变化 | 由扰动供给重新计算 LOS | 同一扰动供给 | 行为与供给耦合结果 |

供给单变臂是人为隔离控制，不应解释为现实出行者不会反应。联合臂必须避免 LOS 已增加 delay 后又额外添加相同 +15 minutes 的双重计数。

先选一个可验证的 headway/access disruption，不必同时做多城市闭环。共享人口、OD 与成对随机数，分别报告 raw prediction、feasibility-adjusted distribution、assignment、route、boarding、completion 和一个运行结果（如 journey time）。真实反馈/均衡迭代仅在相应 claim 存在时增加。

### 本次已补入的机制解释

设 F 是可行选项，Z 是原预测在 F 上的概率质量，q 是条件归一化。则

||q-p||₁ = 2(1-Z) = min_{r∈Δ(F)} ||r-p||₁。

这说明在排除正概率选项时，完全保留原分布不可能；但单状态 L1 最优也不保证状态间响应最优。补充材料给出了固定 F 的三选项反例。该式是已有规则的概率恒等式，不包装为全新理论。

同时将 total execution gap 分成：

**(adjusted response − raw response) + (simulated response − adjusted response)。**

这避免把所选可行性修正与后续执行误差混在一起。本次仅使用既有 rounded means 算出 SA-Student 的 +0.27 pp、−1.34 pp 与 −1.07 pp；未计算新的成分置信区间。

**完成标准**：知道误差是支持集修正、采样、路由还是执行造成；不能只报告最后一个 PT share。

## 9. P2：把结果变成会被复用的研究资产

提供独立于特定 Student 的 benchmark schema、frozen targets、intervention manifests、固定划分、评估函数和 per-scenario error taxonomy。输出一份模型诊断报告即可回答：在哪类干预失配、证据来自哪一层、哪些模式有可用性冲突。

仓库应支持分层复现：

1. 无 API、无 Java 的本地 inference smoke test。
2. 由冻结 target/checkpoint 重算主文核心受控结果。
3. 在允许的数据权限下重新进行 respondent bootstrap。
4. 由固定 supply/population 重跑一个最小 execution pair，再扩展完整矩阵。

Smoke test、artifact hash check、历史作者 verification record 和新的独立 end-to-end reproduction 必须分开。Windows 路径指纹问题应版本化解决，不应绕过后宣称 exact replay。人类原始数据不能公开时，明确申请流程、匿名化的可共享派生数据及限制，不应因此限制全部合成数据。

公开数据/代码是高价值强化项，AIT scope 鼓励开放与可复现；这不等于官方规定每个原始事件文件都必须公开。投稿时至少落实真正可工作的审稿访问。DOI 和公共 release 只能在完成后写入 paper。

### 成本：做可核查的 break-even，而非只报 Student 很快

令 C₀ 为监督获取、重试、训练和必要适配的一次性成本；同一工作负载下单次场景直接 LLM 与 Student 成本差为 C_L−C_S。仅当分母为正时，N*=C₀/(C_L−C_S)。分别计算金钱、wall-clock 和计算资源，不把不同单位混相加。共享网络成本在公平比较中一致处理，缓存假设和并发条件固定。不要用成功请求延迟直接当全部请求的实测总成本。

## 10. 写作结构：主文用发现推进，而非不断自我辩护

建议最终保留三个研究问题：

**RQ1**：什么训练/模型选择能在未见干预下保存冻结 Teacher 的响应？  
**RQ2**：这种 fidelity 是否预测 human agreement；审计能否指导独立验证的修复？  
**RQ3**：响应如何通过可行性、分配和实际供给执行，并在哪里改变？

目前版本完成了“现有证据叙事”，不声称已经完成新增 RQ1 OOD、RQ2 repair、RQ3 physical-supply 实验。等新增实验结束，再据真实结果替换对应段落。

主文图表建议以每张图回答一个判断组织：contrast concept；controlled/OOD + Pareto；human rank reversal + repair；stagewise support/execution decomposition；break-even/cost（必要时补充）。所有模型全量结果留 supplement，主文只按事先声明的角色展示，不能按结果好坏删模型。

结果段采用“结论句 → 绝对数值与不确定性 → 解释 → 推断边界”。不在每段开头重复 only/small/not representative/not complete；限制放在相应 estimand 定义处和集中 scope 段。核心限制不能只藏在 supplement。

## 11. 投稿停止标准

达到以下状态再停止加实验并锁稿：

- P0 事实问题关闭，主文、supplement、仓库、图表版本一致。
- 在完整、预先冻结的实验矩阵中，中心判断仍成立，且 null/failure results 被解释而非删除。
- 明确区分 training fidelity、temporal reference、human calibration 和 network execution；每个 claim 都能指向对应证据。
- 完成至少一个“审计定位 → 修复 → 独立验证”的闭环，以及一个真实供给测试（若保留实质性供给应用主张）。
- 第三方在新环境中完成核心结果重算；无法分享的材料有可执行的访问方案。
- 摘要、标题与 cover letter 只承诺真实完成的贡献，不用未来实验支撑现在的结论。

这个标准的重点是证据闭合与可复用性，不是永远补到所有指标都赢。若新增结果不支持 response-aware training 的泛化优势，转而把“何时失效及如何选模”写成被验证的诊断贡献，不应继续扩大试验直到挑出显著结果。

## 12. 参考与核查入口

**期刊 scope**：https://shop.elsevier.com/journals/artificial-intelligence-for-transportation/3050-8606  
**研究伦理政策**：https://www.elsevier.com/about/policies-and-standards/research-ethics  
**DeepSeek 路由公告**：https://www.deepseek.com/en/news/deepseek-v4-1-flash/  
**Persona alignment**：https://doi.org/10.1287/trsc.2025.0330  
**Network surrogate**：https://doi.org/10.1016/j.trc.2025.105360  
**Sobolev training**：https://arxiv.org/abs/1706.04859  
**Relational KD**：https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html  
**Concurrent preprint**：https://arxiv.org/html/2609.22252v1

仓库内部定位：`README.md`、`docs/REPRODUCIBILITY.md`、`evidence/paper_20260924/verification.json`、`evidence/paper_20260924/teacher/config.json` 与 `run_summary.json`。本次读取并核查其陈述，没有把历史 verification 的执行者改称为本次审稿者。
