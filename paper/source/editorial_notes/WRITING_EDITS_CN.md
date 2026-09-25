# 精修改动说明：哪些是表达问题，哪些不是

## 当前交付的范围

已修改标题、摘要、Introduction、Related Work、方法的目标定义与执行分解、实验设计的推断范围、关键 Results 表述、Discussion、Conclusion、Data/code availability，以及 Supplement S5/S7。其他实验方法、全部既有图和汇总数据保留。新增内容为解析式、反例及既有 rounded means 的算术分解，不是新实验。

## 1. 将“泛化承诺”改成可辨识贡献

原标题：`Preserving LLM-Derived Traveler Responses`。

现标题：`Auditing LLM-Derived Traveler Responses`。

区别：前者容易被理解为论文已普遍实现 preservation；后者准确覆盖实测中的成功、排序反转和执行失配。不是把论文降为无方法的评论，而是以 matched comparison、独立参照和可追踪执行为方法主体。

## 2. 去掉自我削弱，不去掉证据边界

原句：`This convenience and snowball recruitment does not yield representative city samples.`

改为：`The target of inference is response agreement within these convenience and snowball samples, not citywide mode shares.`

这是 estimand 的明确化；招募方式、样本规模、城市不具可比的任务格式仍保留。

原句：`Once both endpoint targets are given, paired supervision adds no new behavioral information.`

改为：`Paired supervision changes the geometry of the fitting objective: it prioritizes differences between states whose endpoint targets are shared by all controlled models.`

随后明确 improvement 不是 extra Teacher information 导致。相同信息成为对照公平性的理由，而不是先宣布方法没有价值。

原句：`The design therefore does not represent adaptation to an actual service disruption or a day-to-day equilibrium.`

改写顺序：先说明固定时刻表隔离 execution effect，再界定不是 physical disruption 或 equilibrium。限制没有消失，但读者先知道为什么这样设计。

## 3. 更正过时事实，而不是改得“看起来公开”

删除“后来实验目录不在仓库”“not all synchronized”等与当前版本不符的表述。新 availability 说明已含 benchmark、targets、splits、checkpoints、evaluation scripts 和 summaries；同时保留 access-controlled、participant-level 数据限制、full simulation supply/runtime 依赖。

精确 pinned commit 见 Supplement S7。没有假称公开 repository、Zenodo DOI 或完整一键重现。

## 4. 纠正一个逻辑错误

原句：`Response fidelity to the Teacher is necessary for a useful Student but not sufficient.`

改为：`Response fidelity is a criterion for faithful compression, not a guarantee of behavioral validity.`

一个不忠实的 Student 可能反而更接近人类。原句会与全文最重要结果冲突，不是只改文风。

## 5. Teacher 术语不再承担无法识别的归因

原来统一称 Teacher 的两类对象改为 archived training Teacher 与 contemporary API reference。保留现有图表中的 Teacher 标签，在图注中明确该标签含义；不重绘数值。

S−H=(T_now−H)+(S−T_now) 保留为描述恒等式，不叫识别出的 compression-error decomposition。实际后端要依据真实时间戳与 provider 信息，不能凭日期型随机种子判断。

## 6. 结果先报数值，不反复评价自己“small”

摘要使用 0.06930 → 0.06586 → 0.05644，而不是让 4.97% 单独承担 novelty。保留最有判别力的 human reversal：+13.85 pp 对 −24.40 pp。

Results 删除一次重复的“small but consistent”，保留绝对效应、persona-clustered interval 和第三个 seed 的不确定性。没有把“未显著分离”改写为胜利。

## 7. 失配有了机制，而不仅是坏消息

新方法段给出 feasibility renormalization 的 L1 最小位移及 response distortion 反例。新结果分解区分 support adjustment 与 downstream execution，不再把所有失配都归为 Student 没保存行为。

这是对已有算法的解析说明，不冒称首创理论；本次进行了数学推导和数值恒等式检查，没有训练新模型。

## 8. 哪些不能靠精修解决

- 未提供的伦理机构/豁免依据不能凭文字补出。
- 未记录的随机化不能写成“randomized”。
- 只测六个 persona 不能写成多样人群验证。
- 改了 loss、输入货币含义或 availability 后，旧实验结果不能继续对应新方法。
- 当代 API 不能替代已丢失的训练版本后端。
- 被反复检查和据此修模型的人类测试集不能再被称为 untouched confirmation。

这些在 `AUTHOR_ACTIONS.md` 单独登记；主文保留必要范围，不用大量管理性自辩填充。
