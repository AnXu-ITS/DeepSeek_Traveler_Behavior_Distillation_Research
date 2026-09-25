# 本轮修订与逻辑核查报告（2026-09-24）

本轮以 `files (1).zip` 内的 `Beyond_Static_Imitation_AIT_revised_source.zip` 为唯一底稿，参考 `AIT_Reviewer_Reassessment_v2.md`，使用用户提供的 Scheme B Figure 2。结果已同步到实际存在的 `Elsevier_template`；没有改动 AIT、模型权重、调查原始回答或上游实验代码。原目标目录备份位于 `C:/Users/xuan1/Downloads/ait_revision_work_20260924/Elsevier_template_before_revision.zip`。

## 已完成的修改

| 审稿点 | 修订位置 | 实际处理 |
|---|---|---|
| 仿真替代规则与时序 | 主文 §3.4；补充 §S7.2 | 按保存实验协议匹配的代码补充：汽车、自行车、PT 搜索失败直接转步行；先预测出发调整，再在调整时刻筛选与路由；零可行概率质量会停止运行，不会删掉人；回程独立路由。 |
| 招募来源与伦理 | 主文 §4.2；补充 §S1.3 | 电子问卷、微信／WhatsApp、人际网络及转发、Google Forms／问卷星、资格筛选、隐私声明、同意与作者说明的伦理豁免。明确 convenience/snowball sampling，不声称城市代表性。 |
| 核心材料可取得性 | Data and code availability；§S7.2；材料索引 | 删除将“未放进投稿包”误写成“不可取得”的陈述，区分投稿包、私有远端仓库和本地保留数据。 |
| Teacher 历史归因 | 摘要；§4.3、§5.3、§6.2 | 保留 signed identity，但解释为当前同任务描述性比较，不用它唯一确定历史 Teacher 偏差或纯蒸馏误差。 |
| cannot help | §6.2 | 更忠实模仿可把该子样本绝对偏差从 24.04 降至 22.10 pp，但不能消除 Teacher–human 剩余偏差。 |
| Eq.7 的含义 | §6.1 | 不等式说明 endpoint 与 response error 的关系，不证明实测收益为何较小。 |
| 上海延误映射 | §5.2、§6.4 | 保留 reference specification 下的结论；八组敏感性都固定 15 分钟延误，未宣称覆盖延误强度变化。 |
| 统计聚合与措辞 | Table 4 注；§5.1 | 指标先在 seed 内取各 contrast 绝对值并平均，再跨 seed 汇总；Figure 4 为平均 signed bias。交互结论改为本 benchmark 未改善，timing 改为 comparable。 |
| Figure 2 | 方法页与图注 | 使用提供的径向 Scheme B，普通标注为 Arial，8 个公式为原生可编辑 Office Math，数学字形为 Cambria Math；原生公式由 PowerPoint 导出，PDF 无栅格图像。图注删除旧 (a)/(b)/(c)，解释 MNL-S endpoint fitting 与额外 SA-Student 的边界。 |

## 发现的逻辑问题及判断

1. **当前 Teacher 不等于历史 Teacher。** 原稿从当前 Teacher–Student–Human 差值直接推出偏差历史起源，超出了描述性分解的能力。已修正；不需要因此全面重新蒸馏。
2. **“更忠实的 Student 完全无帮助”与报告数字矛盾。** 24.04 pp 可降到 22.10 pp。已改成不能仅靠模仿消除剩余 Teacher–human 差异。
3. **三角不等式不能解释实测收益幅度。** 公式正确，原来的解释过度。已修正。
4. **平均绝对值不等于平均后取绝对值。** Table 4 与 Figure 4 的聚合顺序不同，并不是已证实的表格计算错误。已明确。
5. **有限搜索失败不代表不存在任何路线。** 全文相关表述统一到指定搜索的结果。
6. **原稿多处“不保留／不可取得”的判断不成立。** 本地存在 final targets、重复 Teacher 输出、S7/S9 checkpoint 和 training history、个体预测、逐人事件记录等。不能根据投稿 ZIP 的范围推断数据遗失。
7. **通用路由代码有未触发的断网边界隐患。** `_build_legs` 在步行最短路失败时使用空列表，可能仅生成起点 link 的步行方案；同模式返回值也不能证明步行连通。本批 140 次运行未发现这种路线（有 route_links 的出行记录最小为 26），故本次没有证据据此否定 Helsinki 结果。迁移到断开的路网前，应修复实现并定义无法完成出行的处理。这次只报告并解释，没有擅改实验代码或重跑。

综合判断：本轮针对源码、汇总数据和逐人账本的核查未发现足以推翻主要比较、要求全面重训或重跑 140 次仿真的根本错误。上述局部推断和材料描述问题已修正。这不是对全部训练过程、所有 bootstrap 区间或全部历史 API 请求的独立复现。

## 仿真规则核查证据

- 核对实验 driver 和 adapter 与历史 protocol 记录的源文件指纹：匹配。
- 140 个已完成 run；共 140,000 条 run-person 记录；每个 run 初始 1,000 人。
- 没有零可行概率质量或空可行集合；未发现只含起点 link 的已记录步行路线。
- 共 2,006 条 outbound PT→walk 和 940 条 car→walk fallback。这些是跨实验条件重复计数的记录，不是不同受访者人数。
- 未混入保留的 4 GB 启动失败尝试，没有运行新的模型或仿真。

## 必须如实保留的外部限制

- GitHub 仓库经认证查询为 **private**，当前 main 为 `de906ac8a3efae692797bd08e340375107b8e52a`。远端 728 个路径中没有 `outputs/matched_response_v1`、`outputs/revision_20260921` 或 `scripts/revision_20260921`。因此不能宣称“全部训练和实验数据已经公开供任何人使用”。稿件已写成受限访问和保留材料，可联系通讯作者。此任务的“推送”目标是本地目录；未改变 GitHub 可见性，未上传个人数据。
- 用户说明采用伦理豁免，但尚未提供豁免决定机构、编号或书面依据。稿件仅按作者提供的事实写 ethics-exemption basis，没有编造伦理委员会批准。投稿前应由作者确认实际豁免依据及是否属于机构确认或团队依规判断。
- 招募年龄门槛、报酬、配额和题目随机顺序尚无新增事实确认，所以没有虚构“无报酬／无配额／随机化”。
- Google Forms 是编辑链接，可能需要所有者权限；问卷星和 Google Forms 链接未通过无登录网页抓取完成核验。已标注 instrument access，不把它们当成已公开原始回答的入口。
- 上海延误强度敏感性仍未新增，本轮保持条件性结论。第二 Teacher、更多 persona、动态闭环和新调查均不属于本次必要修订。

## 验证结果

- 主文 19 页，补充 24 页；干净构建通过，没有 unresolved references/citations、overfull boxes 或 oversized floats。
- 全页文本框边界扫描：没有页面越界、空白页或 replacement glyph；已目视检查 Figure 2 独立图、正文嵌入页、调查说明页与材料表。
- 四个数值 CSV 与上传底稿逐字节一致。唯一修改的 generated table 是 Table 4 的解释性脚注，表中数字未改。
- Figure 4 的 78 行 identity 最大残差 0.01 pp；52 个 Teacher decomposition 组合最大残差 0.1 pp，均符合原发布精度。
- Helsinki 70 对执行差值 identity 最大残差约 1.33e-15 pp；14.30% 改善算术复核一致。
- PPTX：1 页、8 个原生公式；Arial 普通文字、Cambria Math 数学字形；PowerPoint 成功打开／规范化／导出；结构和版面校验 0 findings，Artifact Tool 导入通过。PPTX 内 Office 自动保留公式兼容预览，并非把公式替换为不可编辑图片；导出 PDF 的栅格图像数为 0。
- prose-quality 检查：0 findings。保留的历史审稿与修改报告仅供追溯，当前结论以本文件和 `QA_20260924.json` 为准。
