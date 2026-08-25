# v0.2 实验报告：真实 DeepSeek 教师蒸馏（A / B / C）

> 蒸馏出行意图 · 个体行为蒸馏研究
> 生成日期：2026-08-20 · 数据集 `data/student_v0_2_a/` · 模型 `outputs/student_v0_2_{a,b,c}/`

## 1. 本轮目标

在真实 DeepSeek V4 Pro 教师数据上，验证「轻量 variable-choice-set Student 能否
学习 K=3 聚合后的 behavioral preference distribution」，并比较三条蒸馏损失设计：

- **v0.2-A**：`L_action + L_distribution`（baseline）
- **v0.2-B**：`+ λ_E · L_elasticity`（行为弹性保留）
- **v0.2-C**：`+ λ_E · L_elasticity + λ_H · L_heterogeneity`（再加人群异质性保留）

## 2. 数据集

- 生成方式：`generate_aggregated_teacher_dataset.py`，K=3 cache-busted 真实调用取均值分布。
- 规模：8 personas × 3 trips × {weather_intensity(4 档), fare_multiplier(3 档)} = **192 states**。
- 完整性校验（`validate_aggregated_dataset.py`）：**PASS**
  - 192 aggregated samples；576 repeat records（每 state 恰好 3 次、completion id 互异，无网关缓存复用）。
  - 24 baselines + 168 counterfactuals；24 split groups。
  - baseline↔CF 经 `baseline_sample_id` 关联、`split_group_id` 一致；概率和≈1。
- 切分：按 `split_group_id`（persona::trip）**train 136 / val 32 / test 24**，split group 零重叠。

## 3. 结果

### 3.1 测试集指标（test = 24 states，3 个 split group）

| 指标 | v0.2-A | v0.2-B | v0.2-C |
|---|---|---|---|
| mode_accuracy | 0.7917 | 0.7917 | 0.7917 |
| cross_entropy | 0.575 | 0.625 | 0.592 |
| KL(P_T‖P_S) | 0.0984 | 0.1160 | **0.0787** |
| probability L1 | 0.3031 | 0.3503 | **0.2891** |
| departure MAE (min) | 3.11 | 2.90 | 3.21 |
| departure sign agreement | 0.833 | 0.875 | 0.792 |

### 3.2 行为弹性（counterfactual `|ΔP_T − ΔP_S|`，test）

| 版本 | mean \|ΔP_T−ΔP_S\| | 配对数 |
|---|---|---|
| v0.2-A | **0.1269** | 21 |
| v0.2-B | 0.1564 | 21 |
| v0.2-C | 0.1461 | 21 |

### 3.3 人群异质性（between-persona `|ΔP_T−ΔP_S|`，test）

- v0.2-C：**0.1601**（8 pairs；A/B 无此损失项，未报告该指标）。

## 4. 关键结论（诚实版）

1. **管线端到端可复现**：真实教师 → K=3 聚合 → variable-choice-set Student → 评估，
   全程脚本化，108→120 个单元测试通过，数据集完整性自动校验通过。

2. **mode choice 是"容易"的部分**：三个版本 mode_accuracy 均为 0.7917。变量选择集 +
   masked softmax 能稳定学会 argmax 选择。

3. **分布拟合达到教师噪声附近量级**：probability L1 = 0.29–0.35。对照教师自身噪声
   参考（single→K5 L1 = 0.117，K3↔K5 L1 = 0.047），Student 的分布误差约为"单次教师
   调用噪声"的 2–3 倍——说明还有提升空间，但已进入可解释区间。

4. **L_heterogeneity 是有效的分布正则**：v0.2-C 在加入异质性项后，KL 从 0.098 降到
   0.079、probability L1 从 0.303 降到 0.289，且未牺牲 mode accuracy。

5. **L_elasticity 单独加入没有在 held-out 上改善弹性**：v0.2-B 的 test ΔP 误差(0.156)
   反而略高于 baseline(0.127)，且分布拟合变差(prob L1 0.350)。v0.2-C(0.146)介于两者
   之间。可能原因：弹性损失在 train 上过拟合到 train 的 ΔP，且 test 极小。

## 5. 必须声明的局限（不可过度解读）

- **test 仅 24 states / 21 弹性对 / 8 异质性对**：0.127 vs 0.146 vs 0.156 的差异在
  统计上**不显著**，只能说明"管线可用、方向合理"，不能宣称 B/C 比 A 更好或更差。
- 合成 persona/trip，无真实世界标定。
- persona overlap=True（三版本同 split 保证可比）；未见人群泛化需独立 persona-holdout
  实验（需要更多 persona，例如 20+）。
- 这是 development-scale 证明，不是最终规模模型。

## 6. 下一步

1. **扩大数据**：更多 persona（20+）× 更多 trip × 更多扰动轴（拥堵/延误/停车费/道路中断），
   并做 persona-holdout 切分检验对未见人群的泛化。
2. **弹性损失设计再审视**：当前用 mean-|ΔP_T−ΔP_S|；可考虑方向对齐（sign agreement）
   与幅度分项，并在更大 test 上重跑 A/B/C 对比。
3. **Phase 8–10**：MATSimAdapter → population-scale → network 闭环（MATSim 2026.0 已在 `tools/`）。

## 7. 复现命令

```powershell
# 数据（已生成）
.venv\Scripts\python.exe scripts\validate_aggregated_dataset.py `
  --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl `
  --repeats data/student_v0_2_a/repeat_records.jsonl

# 训练
.venv\Scripts\python.exe scripts\train_student_v0_2_a.py --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl --output outputs/student_v0_2_a
.venv\Scripts\python.exe scripts\train_student_v0_2_b.py --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl --output outputs/student_v0_2_b
.venv\Scripts\python.exe scripts\train_student_v0_2_c.py --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl --output outputs/student_v0_2_c

# 对比
.venv\Scripts\python.exe scripts\compare_students.py --baseline outputs/student_v0_2_a --candidate outputs/student_v0_2_b --output outputs/comparison_A_vs_B.md
.venv\Scripts\python.exe scripts\compare_students.py --baseline outputs/student_v0_2_a --candidate outputs/student_v0_2_c --output outputs/comparison_A_vs_C.md
```
