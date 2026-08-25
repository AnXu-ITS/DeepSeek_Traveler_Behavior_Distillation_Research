# S1 扩训实验报告：road_congestion 轴补训（决定性实验）

> 目标：回答「居民对拥堵是否敏感」——区分此前 Phase 10 观察到的
> 「student 拥堵弹性弱」是**数据缺口**（拥堵轴无训练样本）还是
> **居民(教师)对拥堵本就不敏感**。

## 数据

- 新增 road_congestion 反事实态：20 personas × 2 trips × 4 档 `[0.2, 0.4, 0.6, 0.8]`
  （baseline road_congestion=0.3，4 档均异于 baseline）= **160 态 × K=3 = 480 次真实调用**。
- 复用现有 40 条 baseline（不重烧 baseline 调用），合并后数据集 **480 态 / 1440 repeat / 40 split groups**，
  `validate_aggregated_dataset.py` PASS。
- 生成脚本：`scripts/extend_teacher_dataset.py`（从现有 baseline state 派生 CF，baseline 链接自动对齐）。

## 教师侧（ground truth，test = 3 未见 personas / 72 态）

| 扰动轴 | 教师弹性 \|ΔP_T\| | 教师采样噪声(pairwise L1, K=3) |
|---|---|---|
| weather_intensity | 0.0797 | 0.1710 |
| fare_multiplier | 0.0493 | 0.1451 |
| road_congestion | **0.0572** | 0.1325 |

关键事实：**教师对拥堵有中等强度、方向正确的响应**（\|ΔP_T\|=0.057，介于票价 0.049 与
天气 0.080 之间；car 在拥堵 0.2→0.8 单调下降约 0.07 概率质量）。但该信号
**低于教师自身的 K=3 采样噪声**（0.133），即拥堵轴的信噪比是三轴中最差的
（0.057/0.133≈0.43，天气 0.47、票价 0.34），与 v0.2 教师审计"拥堵轴噪声最大"一致。

## Student 侧：拥堵轴 before/after（同一 24 个 test 拥堵态，clean 对比）

| checkpoint | \|ΔP_T\| | \|ΔP_S\| | \|ΔP_T−ΔP_S\| | sign agreement |
|---|---|---|---|---|
| v0.3-A（无拥堵训练） | 0.0572 | 0.0017 | 0.0578 | 0.4167 |
| v0.3-C（无拥堵训练） | 0.0572 | 0.0067 | 0.0637 | 0.3611 |
| **v0.3-S1-A（补训后）** | 0.0572 | 0.0186 | 0.0465 | 0.6250 |
| **v0.3-S1-C（补训后）** | 0.0572 | 0.0215 | 0.0604 | **0.7361** |

- **方向（决定性）**：sign agreement 从 0.36（≈随机）→ **0.74**（C）。补训后 student
  学会了拥堵响应的**正确方向**——此前"拥堵弹性弱"确实是**数据缺口**，而非居民不敏感。
- **幅度**：student 仍**欠响应**（\|ΔP_S\|=0.022 vs 教师 0.057），拥堵轴的残差
  \|ΔP_T−ΔP_S\|=0.060 仍是三轴中最大（天气 0.052 / 票价 0.051）。幅度天花板由
  **教师信噪比**决定（信号 0.057 < 噪声 0.133），正则化 student 对噪声轴收缩，属预期。

## 三管线整体（test = 72 态，含拥堵；注意 test 组成从 48→72 态变化，整体指标仅参考）

| 指标(test) | v0.3-S1-A | v0.3-S1-B | v0.3-S1-C |
|---|---|---|---|
| mode_accuracy | 0.8472 | 0.8056 | **0.8750** |
| KL(P_T‖P_S) | 0.1233 | 0.1288 | **0.0659** |
| probability L1 | 0.3043 | 0.3121 | **0.2511** |
| counterfactual \|dP_T−dP_S\| | 0.0563 | 0.0547 | 0.0548 |
| sign agreement | 0.6667 | 0.6869 | **0.7424** |
| heterogeneity \|dP_T−dP_S\| | n/a | n/a | 0.0720 |

## 结论

1. **教师对拥堵敏感**（方向正确、\|ΔP_T\|=0.057），但信号弱于教师自身噪声。
2. 之前 student 拥堵弹性弱主要是**数据缺口**：补训后方向一致性 0.36→0.74。
3. 幅度欠响应是**教师信噪比上限**（0.057 信号 vs 0.133 噪声），非 student 架构缺陷；
   提高幅度拟合需增加拥堵轴重复数（K>3）以压低聚合噪声，而非更多 persona。

## 复现

```powershell
# 生成（已完成，480 calls）
.\.venv\Scripts\python.exe scripts\extend_teacher_dataset.py `
  --dataset data\student_v0_3\aggregated_teacher_dataset.jsonl `
  --axes road_congestion --output data\student_v0_3_s1 --workers 6

# 校验 + 重训 A/B/C + 对比
.\.venv\Scripts\python.exe scripts\run_v0_3_s1_experiment.py `
  --dataset data\student_v0_3_s1\aggregated_teacher_dataset.jsonl `
  --repeats data\student_v0_3_s1\repeat_records.jsonl --outputs outputs

# before/after 拥堵轴弹性
.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s1\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml   # before
.\.venv\Scripts\python.exe scripts\eval_congestion_elasticity.py `
  --dataset data\student_v0_3_s1\aggregated_teacher_dataset.jsonl `
  --checkpoint outputs\student_v0_3_s1_c\checkpoints\best.pt --config configs\student_v0_3_c.yaml # after
```
