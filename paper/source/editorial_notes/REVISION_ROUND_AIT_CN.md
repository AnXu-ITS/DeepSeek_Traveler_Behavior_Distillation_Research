# 本轮修订说明（面向 Artificial Intelligence for Transportation）

## 一、核心诊断

1. 上一轮修订把几乎所有限定语放进了正文主叙事，导致论文读起来像审稿回复或审计文件。本轮把限定语集中到实验设计（出现一次）和 Discussion 4.4 Scope 小节，结果部分改为先给结论、再给证据和解释。
2. 原摘要以 4.97% 开头，这是全文最弱的结果。本轮把主线改为"三个层面的 fidelity 互不传递"，现有数据从四个方向支撑这一点：
   - MNL-S 的 Teacher 误差最小，却唯一预测反了新加坡可达性响应；
   - CE+KL 的 Teacher 误差最大，却最接近该响应（-23.37 对 -24.40，数值来自补充表）；
   - 新加坡 SA-Student 的 PT 水平偏差最大（-30.7），平均响应偏差却最小（4.0）；
   - Helsinki 中 PT 使用水平从预测到仿真下降 11 个点以上（27.74% 到 161/1000），响应只变化 0.8 个点。
3. 第二项贡献改写为"模型设定比损失函数更重要"：MNL-S 相对最佳目标函数再降 14.30%，原文把它写轻了。

## 二、Story 与文献定位

- 引言对接 AIT 自身发表的 Liu et al. (2025)：该文把可扩展性和行为对齐列为 LLM agent 建模的两大障碍，并提出 hybrid 方案；本文把 distillation 定位为其中一种 hybrid 方案。
- 新增 5 篇已核实元数据的文献：Argyle et al. 2023 (Political Analysis)、Aher et al. 2023 (ICML)、Bisbee et al. 2024 (Political Analysis)、Goli & Singh 2024 (Marketing Science)、Wang, Wang & Zhao 2020 (TR-C)。
- Bisbee et al. 关于模型版本漂移的发现，为 contemporary Teacher 可能随时间变化这一点提供了文献依据。

## 三、去编码化

- 删除的写法：supervision records、union mask、perturbed-endpoint availability mask、initial-population denominator、register PT use、Teacher lineage、remote model 等。
- 表格改动："Policy" 改为 "Assignment"，"Sampling SD" 改为 "Assignment SD"。
- 全文去除第一人称。
- 结果段落先给结论，再给证据和解释。解释性推断（例如 MNL-S 为何更好、CE+KL 为何更差）都标注为可能的解读。其中 CE+KL 的解释依据补充材料 S2 已报告的响应放大结果。

## 四、未改动的内容

- 所有实验设置、数值、统计结果和结论。脚本比对显示，主文与主表中的原数字均被保留。
- 新增数字只有三类：
  - 引自补充表的 -23.37；
  - 由 30.67 取整得到的 "about 31"；
  - 由已报告数字推导的 "more than eleven points"。
- 11 个编号公式内容不变。唯一变化是 eq:bound 句末标点从句号改为逗号，因为句子在公式后继续。
- 全部 label 保留，包括补充材料反向引用的三个。

## 五、建议补充的实验（未写成已完成）

1. **在人类数据上交叉验证估计的 MNL，作为参照上界。** 这回答审稿人最可能问的问题：LLM 路线相对传统问卷建模差距有多大。
2. **第二个 Teacher 模型。** 这能区分哪些结论是 DeepSeek 特有的。
3. **延误强度的 dose-response 测试（5/15/30 分钟）。** 这能检验响应的单调性与幅度外推。

## 六、需作者核实

- 方法部分已把 "service identified in the supervision records as deepseek-v4-pro" 改为直接陈述 Teacher 为 DeepSeek deepseek-v4-pro，请确认。
- 全文沿用美式拼写，以与图中文字保持一致。
- 补充材料本轮未改写，其中仍有部分工程化表述，可在下一轮处理。

## 七、编译

- 本沙盒中 microtype 的字体扩展会报错，因此编译时通过命令行传入 `expansion=false`，未修改源文件。
- 正常 TeX Live 环境可直接运行 `python build.py --clean`。
- 编译结果：正文 19 页，无未解析引用，无 overfull，无 float too large。原稿存在的 Figure 3 浮动组溢出已通过缩小组内间距修复。

---

# 第二轮：补充材料去编码化

## 改动范围

补充材料 S1–S7 的正文全部重写。30 余张补充表格的标题和脚注改为学术表述，表内数值未动。补充材料摘要已重写，原有的 "Guide to the supplement" 段落已删除。

## 主要去编码化处理

- **内部标识符。**
  - S7-W3、S9、teacher_s8_accessibility_v0.1 已替换为描述性表述。
  - 输入字段表中的 snake_case 变量名（如 `desired_departure_min`）已改为自然语言，并注明实现名称见补充数据文件。
  - `non_network_walk` 已改为物理含义描述。
- **工程词汇。**
  - legacy source 改为 single-context。
  - endpoint 改为 state。
  - Policy / Sampling SD 改为 Assignment / Assignment SD。
  - prefix、warm-up、hash salt、round-robin、updater、payload、ledger、source package、machine-readable、frozen、support conflict 等均已改写。
  - union mask 改为 "available in either state of the pair"。
- **生成记录注释。** 表格源文件中的 `% Generated from outputs/revision_...` 等构建注释已删除。它们不显示在 PDF 中，但会随源码提交。
- **S7。** 原来是审计式的材料边界说明，现改写为常规的材料与可复现性章节。所有"未提供"的事实如实保留，但每项只陈述一次，不再重复防御。
- **补充解释。** 新增两处解释，均直接由已报告数字推出：
  - Shanghai 速度实验中，延误使 90 人离开 PT（224 − 134）。
  - 独立神经时间预测器缩小了与联合模型的大部分时间误差差距。

## 核对结果

- 补充材料 5 个编号公式逐字一致。
- 补充正文新增数字只有 90，由 224 − 134 推得。
- 所有表格（含脚注）的数字集合与原稿一致。
- 编译：补充材料 24 页，正文 19 页，均无未解析引用、无 overfull、无 float too large。

## 本轮需作者核实

1. **ethics statement。** 原 S7 写明"招募渠道、抽样框和伦理审查信息本源码包未能确定"。本轮从补充材料中删去了这句自我否定式表述。投稿 AIT 前，正文必须补上 Ethics statement（伦理审批或豁免编号、招募方式）。这是投稿硬性要求，不能靠措辞解决。
2. **Table S（tab:pipeline）的 "Original (s)" 列。** 原稿没有定义这一列。本轮只把 cold/warm 在标题中解释清楚，Original 的含义请作者补一句，或删除该列。
3. **时间回退相关描述。** 两处描述依据原稿推定，请确认：
   - S1：class E 表示 "no connection was found"。
   - S5：出行测度截至 "first activity other than home or an interaction point created by routing"，对应原文的 "non-interaction activity"。
4. **Teacher 请求设置。** S3 原文 "no thinking-mode or reasoning-effort override" 改写为 "without changing the default reasoning settings"，请确认语义一致。
5. **未使用的表格文件。** `table_availability.tex` 未被任何章节引用（上一轮删除的扩展可用性比较），本轮未改动，可从源码包中删除。
6. **编辑说明文件。** `editorial_notes/` 中上一轮的中文报告、patch 和 QA 文件仍保留旧命名（S9 等）。它们仅供内部参考，投稿时不要上传。
