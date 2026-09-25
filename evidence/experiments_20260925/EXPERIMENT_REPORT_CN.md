# 实验进展与写作前报告

这是实验记录，不是论文正文修订。原问卷、原始人类回答及历史实验文件保持原样。

## 已完成：零 API 训练和复核

27 次完整重训，54 个按不同验证标准选择的检查点；各次均为 120 epoch / 6,360 updates。九个历史同配置模型参数和选中 epoch 完全复现。
另完成 4 个 MNL 正则化候选拟合；两个选择标准均选择 L2=0.0001。MNL choice 与 departure 的选择规则保持区分。

| 目标 | 权重 | 选择 | 平均响应误差 | 训练种子 SD | 静态宏 KL |
|---|---:|---|---:|---:|---:|
| soft_kl | 0 | static | 0.069301 | 0.003163 | 0.091533 |
| soft_kl | 0 | response | 0.070515 | 0.004985 | 0.102713 |
| signed_l1 | 0.25 | static | 0.068459 | 0.005230 | 0.091731 |
| signed_l1 | 0.25 | response | 0.069107 | 0.006174 | 0.100084 |
| signed_l1 | 0.5 | static | 0.068114 | 0.005963 | 0.089003 |
| signed_l1 | 0.5 | response | 0.069417 | 0.007090 | 0.099216 |
| signed_l1 | 1 | static | 0.066531 | 0.004970 | 0.087059 |
| signed_l1 | 1 | response | 0.066625 | 0.004721 | 0.088025 |
| signed_l1 | 2 | static | 0.065670 | 0.006534 | 0.085843 |
| signed_l1 | 2 | response | 0.065888 | 0.006674 | 0.087048 |
| direction_magnitude | 0.25 | static | 0.068568 | 0.005246 | 0.088491 |
| direction_magnitude | 0.25 | response | 0.068544 | 0.004878 | 0.090606 |
| direction_magnitude | 0.5 | static | 0.067222 | 0.004049 | 0.087446 |
| direction_magnitude | 0.5 | response | 0.066928 | 0.003843 | 0.087923 |
| direction_magnitude | 1 | static | 0.065856 | 0.004165 | 0.086086 |
| direction_magnitude | 1 | response | 0.068310 | 0.006149 | 0.097354 |
| direction_magnitude | 2 | static | 0.065966 | 0.004536 | 0.090433 |
| direction_magnitude | 2 | response | 0.065983 | 0.004343 | 0.090307 |

这些均值不是独立总体样本：原测试集只有 6 个独立人设。配对人设区间见 `controlled/paired_comparisons.json`，权重网格的比较为敏感性分析，不能事后选最优测试权重当作预注册主结果。

组合留出审计：训练/验证未引用 fare×congestion 留出端点；测试有 24 个该组合端点。构成它的单独因素在训练中存在，因此结论只适用于组合留出，不是全部干预类别未见。

重复 Teacher 分析覆盖 373 个端点、350 对响应。K=3 使用 1 vs 1，K=5 使用 2 vs 2，保留共享端点和方法间相同分配。30 个分配相互相关，不能视作 30 次独立试验，也不能当作精确噪声上限。

旧执行实验逐阶段统计：先在每个人内平均配对分配种子，再整体重采样人，分别计算概率调整、分配、路径、上车对响应的增量。

## 干预类别留出

另将正 transit_delay 的端点及整个 delay 机制四元组从训练、验证和所有关联单元移除，重新拟合特征统计，再训练 soft-KL / signed-L1 × 3 种子，共 6 个模型。各自 120 epoch / 5,280 updates；同一留出实验内预算一致，不能与全数据训练混称同更新数。历史测试的 delay-only 结果是回顾性诊断；新模型参考样本评估另列。

## API 与新增数据

当前 pilot 有效响应：0/432。请求实际模型 deepseek-v4.1-flash，提供方 OpenCode Go；不能记为 V4 Pro 或 DeepSeek 官方直供。
预检遇到 HTTP 400：Go 要求工作区 Privacy 使用 Global regions。已经停止自动重试，等待作者修改或选择暂缓。失败请求未报告 token 用量；这不能替代账户账单。
计划 pilot 约 1.84M tokens（包含 10% 余量）；按旧平均用量与 Go Flash 价格约 0.7–1.5 USD 的套餐额度。套餐订阅费与按 token 计算的额度价值分开报告。正式人数按 pilot 中逐人主要差值 SD 和半宽 0.01 计算，30–120 人，pilot 与正式人设完全分离。

独立人类修复验证按作者回复暂缓，现有问卷不能充当新增独立验证。

## 物理供给实验

已核验 40/40 个模型×情境×配对分配种子组合。
按线路与停站序列分组隔班删除，原 13,655 趟保留 6,924 趟。供给索引与 MATSim 时刻表共同修改。四臂为原行为/原供给、仅感知延迟/原供给、冻结原行为/受扰供给、按受扰 LOS 重算行为/受扰供给。
时间沿用历史网络运行口径，不能据此声称校准过的步行、骑行效率改善；主要判断仍是初始人群分母下的概率、分配、路径、实际上车和完成情况。

## 写作停止点

尚未开始把新结果写入论文。待其余可执行实验与完整性核验结束后，再向作者汇报；新 API 的阻塞单独列明，不能声称全部实验已经完成。

## 必须保留的类别留出负结果

static 选择下，58 个 delay 测试对：soft-KL=0.058267，signed-L1=0.061396；signed-minus-soft=0.003129，配对区间 [0.001818854288406808, 0.004942290550370889]。
response 选择下，58 个 delay 测试对：soft-KL=0.057867，signed-L1=0.063188；signed-minus-soft=0.005321，配对区间 [0.0036211654304440974, 0.007149197277125234]。
signed-L1 在这项类别留出诊断中更差；不能将已覆盖任务的响应改善写成未见干预上的普遍提升。区间仍条件于仅 6 个历史测试人设。

## 四臂供给实验已核验结果

| 模型 | 比较 | 原始 PT 概率变化 pp | 实际上车变化 pp | 完成率变化 pp |
|---|---|---:|---:|---:|
| s9 | perceived_delay minus baseline | -13.095 | -14.080 | 0.000 |
| s9 | frozen_disrupted minus baseline | -0.000 | 0.000 | 0.000 |
| s9 | recomputed_disrupted minus baseline | -0.600 | -0.640 | 0.000 |
| s9 | recomputed_disrupted minus frozen_disrupted | -0.600 | -0.640 | 0.000 |
| independent_neural_seed42 | perceived_delay minus baseline | -6.839 | -6.100 | 0.000 |
| independent_neural_seed42 | frozen_disrupted minus baseline | 0.000 | 0.000 | 0.000 |
| independent_neural_seed42 | recomputed_disrupted minus baseline | -1.143 | -1.000 | 0.000 |
| independent_neural_seed42 | recomputed_disrupted minus frozen_disrupted | -1.143 | -1.000 | 0.000 |

40 个组合包括 12 个逐人核验后复用的历史运行与 28 个新 MATSim 运行。每个组合的初始分母为 1,000 人，配对分配种子为 5 个；不能把 40,000 次执行记录视作 40,000 个独立人。区间、各阶段及行程时间见 `physical_supply`。
