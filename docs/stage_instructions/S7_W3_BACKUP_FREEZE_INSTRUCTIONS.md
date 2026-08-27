# S7-W3 Backup & Freeze Instructions
## S7-W3 通用行为核心备份冻结执行指令

**对象**：`S7-W3` final checkpoint  
**角色定义**：Generic Behavioral Core v1.0  
**目的**：在进入 S8 Transit Accessibility 训练之前，将 S7-W3 固化为不可覆盖、可复现、可审计、可作为论文基线的正式模型版本。  
**原则**：S7-W3 之后不再修改；S8 必须从独立目录和独立 checkpoint 开始。

---

# 1. Freeze 目标

本次 Freeze 必须保证未来任意时刻都可以回答：

1. S7-W3 当时的精确 checkpoint 是哪个？
2. 对应的输入 schema、normalization、config 是什么？
3. 当时使用的是哪个代码版本？
4. legacy / joint / causal / seed robustness 的指标是什么？
5. 用哪些数据训练和验证？
6. S8 是否覆盖或污染了 S7-W3？
7. 是否可以独立恢复并复现实验结果？

Freeze 完成后，S7-W3 正式定义为：

> **Generic Behavioral Core v1.0**

---

# 2. 禁止事项

Freeze 后禁止：

- [ ] 覆盖 S7-W3 checkpoint；
- [ ] 修改其 config 后仍沿用同一版本名；
- [ ] 修改 feature schema 后仍声称是 S7-W3；
- [ ] 将 S8 权重保存到 S7-W3 目录；
- [ ] 删除原始 S7/S6/S5 报告；
- [ ] 用新的 test 结果回写覆盖原始冻结指标；
- [ ] 无版本号地继续训练该 checkpoint。

任何后续微调必须建立新版本：

```text
S8-...
S9-...
```

---

# 3. 推荐冻结目录

创建：

```text
releases/
  s7_w3_generic_core_v1/
    checkpoint/
    config/
    schema/
    normalization/
    metrics/
    reports/
    data_manifest/
    code_manifest/
    checksums/
    README.md
    FINAL_S7_W3_FREEZE.md
```

如果项目已有统一 release 目录，沿用现有规范即可，但必须与训练输出目录分离。

---

# 4. Step 1 — 定位唯一最终 checkpoint

Agent 首先必须确认：

- S7-W3 最终 checkpoint 路径；
- seed；
- epoch / step；
- optimizer state 是否保留；
- model architecture；
- 参数量；
- checkpoint 保存时间；
- 是否与报告中的 W3 完全一致。

生成：

```text
releases/s7_w3_generic_core_v1/checkpoint/model.pt
```

或保留原格式。

同时复制：

```text
optimizer state
scheduler state
training state
```

如果这些文件存在。

---

# 5. Step 2 — 冻结配置

复制所有与 S7-W3 有关的配置：

```text
student config
training config
loss weights
replay ratio
learning rate
batch size
seed
early stopping config
split config
feature config
```

建议输出：

```text
config/student_s7_w3.yaml
config/training_s7_w3.yaml
```

不得只依赖代码默认值。

---

# 6. Step 3 — 冻结输入 Schema

必须把 Student 当时真实使用的输入完整记录下来。

至少包含：

## Persona features
列出全部字段。

## Trip features
列出全部字段。

## Mode-level attributes
列出全部字段。

## Dynamic context
列出全部字段：

- weather
- fare
- road_congestion
- transit_delay
- parking_cost
- road_disruption
- 以及实际存在的其他字段。

## Availability representation
记录：

- choice-set encoding；
- mask；
- unavailable mode 行为。

## Output
记录：

- mode probabilities；
- departure-time shift；
- 其他输出头。

输出：

```text
schema/input_schema.json
schema/output_schema.json
schema/feature_order.txt
```

特别重要：

> `feature_order.txt` 必须与实际 tensor 输入顺序一致。

---

# 7. Step 4 — 冻结 Normalization / Encoding

保存：

- mean；
- std；
- min/max；
- categorical mapping；
- mode mapping；
- mask conventions；
- clipping range。

例如：

```text
normalization/normalization.json
normalization/mode_mapping.json
normalization/category_mapping.json
```

这是后续 S8 判断“能否扩展输入而不破坏兼容性”的关键依据。

---

# 8. Step 5 — 冻结数据版本

建立 data manifest，而不是复制所有大数据。

记录：

```text
S3 legacy dataset
S5 joint dataset
S6 causal audit dataset
S7 mechanism dataset
```

每个数据集记录：

- path；
- number of states；
- number of teacher repeats；
- split；
- persona count；
- checksum；
- creation date；
- source model；
- K 值；
- incomplete count。

输出：

```text
data_manifest/data_manifest.json
```

---

# 9. Step 6 — 冻结代码版本

记录：

```text
git commit hash
git branch
git status
python version
pytorch version
java version
MATSim version
```

若工作树非 clean：

- 必须输出 diff；
- 归档 patch。

输出：

```text
code_manifest/git_commit.txt
code_manifest/git_status.txt
code_manifest/environment.txt
code_manifest/uncommitted.patch
```

如果允许，建议先提交一个正式 commit：

```text
freeze: S7-W3 generic behavioral core v1.0
```

然后创建 Git tag：

```text
s7-w3-generic-core-v1.0
```

---

# 10. Step 7 — Freeze 前复现性 Gate

Freeze 前必须重新运行一次最小但关键的验证，不重新训练。

## 10.1 Legacy smoke
确认：

- mode accuracy；
- KL；
- probability L1。

## 10.2 S5 joint smoke
确认：

- seen joint KL；
- unseen joint KL。

## 10.3 S6/S7 causal smoke
至少确认：

- parking G_med；
- congestion Gap_shortcut。

要求：

> 与历史报告数值误差在可解释范围内。

若出现明显不一致：

**暂停 Freeze，先定位原因。**

---

# 11. Step 8 — 冻结核心指标

将以下结果写入：

```text
metrics/final_metrics.json
```

至少包括：

## Legacy
- accuracy
- KL
- probability L1
- ΔP gap
- sign agreement

## Multi-axis
- seen joint KL
- seen joint L1
- unseen joint KL
- unseen joint L1
- interaction error

## Causal/mechanism
- parking G_med
- parking Gap_shortcut
- congestion G_med
- congestion Gap_shortcut
- R_shortcut
- R_mediator

## Robustness
记录 seeds：

```text
42
7
123
2024
```

以及每个 seed 的关键结果。

---

# 12. Step 9 — 冻结四种种子结果

复制或引用：

```text
seed=42
seed=7
seed=123
seed=2024
```

明确：

- 4/4 parking G_med 改善；
- 4/4 congestion Gap_shortcut 改善；
- 4/4 legacy KL 无回归；
- seen joint 无显著回退。

输出：

```text
metrics/seed_robustness.csv
```

---

# 13. Step 10 — 复制关键报告

至少归档：

```text
EXPERIMENT_REPORT_S5_MULTI_AXIS.md
EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md
EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md
```

以及 seed robustness 报告。

复制到：

```text
reports/
```

---

# 14. Step 11 — Checksum

对以下关键文件生成 SHA256：

- final checkpoint；
- config；
- schema；
- normalization；
- metrics；
- data manifest；
- reports。

输出：

```text
checksums/SHA256SUMS.txt
```

Freeze 后任何 checksum 改变都意味着版本已被修改。

---

# 15. Step 12 — 创建 FINAL_S7_W3_FREEZE.md

内容必须包括：

## Identity
```text
Model: S7-W3
Release: Generic Behavioral Core v1.0
Status: FROZEN
```

## Scope
声明该模型已完成：

- single-axis distillation；
- multi-axis distillation；
- mechanism audit；
- mechanism-aware targeted fine-tuning；
- seed robustness。

## Claims allowed
可表述：

> axis-dependent partial mechanism preservation

> generic behavioral core

> simulation-executable traveler agent

## Claims prohibited
禁止表述：

> full causal reasoning

> universal human behavior model

> Singapore-trained behavior model

---

# 16. Step 13 — README

README 必须说明如何：

1. 加载 checkpoint；
2. 加载 schema；
3. 构造输入；
4. 执行 inference；
5. 运行最小 evaluation；
6. 恢复版本。

---

# 17. Step 14 — Git Tag

最终执行：

```bash
git tag -a s7-w3-generic-core-v1.0 -m "Freeze S7-W3 generic behavioral core v1.0"
```

如果已有 tag 管理规范，按现有规范执行。

不要 force 覆盖已有 tag。

---

# 18. Step 15 — 只读保护

如果操作系统允许：

- 将 release checkpoint 标记为 read-only；
- 或通过项目规范禁止 write。

至少代码层面：

```text
S8 scripts must reject output paths inside releases/s7_w3_generic_core_v1/
```

增加 hard assertion。

---

# 19. Freeze 完成判定

必须全部满足：

- [ ] checkpoint 唯一定位；
- [ ] config 完整；
- [ ] schema 完整；
- [ ] normalization 完整；
- [ ] metrics snapshot 完整；
- [ ] 4-seed robustness 已归档；
- [ ] data manifest 完整；
- [ ] git commit/tag 完成；
- [ ] SHA256 完成；
- [ ] smoke reproduction 通过；
- [ ] release 目录不可被 S8 覆盖；
- [ ] `FINAL_S7_W3_FREEZE.md` 已生成。

---

# 20. Freeze 后状态

正式定义：

```text
S7-W3
=
Generic Behavioral Core v1.0
=
FROZEN
```

S8 只能：

```text
load S7-W3
→ initialize new S8 experiment
→ save to separate output/release
```

不得修改 S7-W3 本体。

---

# 21. 最终 Agent 汇报格式

Agent 完成后只汇报：

1. freeze directory；
2. checkpoint checksum；
3. git commit/tag；
4. smoke reproduction 是否一致；
5. 是否存在任何未提交文件；
6. 是否完成只读/路径保护；
7. `FINAL_S7_W3_FREEZE.md` 路径。

---

# 22. 一句话执行原则

> **把 S7-W3 当成已经完成的论文级 Generic Behavioral Core，而不是“下一次训练的临时 checkpoint”；S8 只能建立在它之上，不能改写它。**
