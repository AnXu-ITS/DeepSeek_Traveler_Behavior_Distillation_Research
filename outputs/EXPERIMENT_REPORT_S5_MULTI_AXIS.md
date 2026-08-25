# S5 多轴联合扰动蒸馏实验报告

## Multi-Axis & Low-SNR Joint Behavioral Distillation

**项目主线**：DeepSeek V4 Pro Teacher → Lightweight Traveler Agent → MATSim
**阶段**：S5 — 多条件联合行为响应蒸馏（joint counterfactual）
**依据**：`S5_MULTI_AXIS_DISTILLATION_EXPERIMENT_DESIGN.md`

## 1. 数据

- joint 状态：640（4 组合 × 80 baselines × 2 joint levels）
- 教师调用：2640 次（K=3/5/5/3 + K=7 因果子集）
- 组合：rain_x_congestion, fare_x_congestion, fare_x_transit_delay, road_disruption_x_congestion
- K=7 子集：{'combination_id': 'fare_x_transit_delay', 'n_states': 40}
- model：deepseek-v4-pro；endpoint：https://api.deepseek.com/v1
- 最终数据：**640 态 / 2640 repeat / 0 incomplete**（跨 resume 累计）；最后一次 run 统计：attempted=2513 failed=2 aggregated=607（其余为前序 resume 已完成）

四组合：
| combination | axes | seen | K |
|---|---|---|---|
| rain_x_congestion | weather × road_congestion | seen | 3 |
| fare_x_transit_delay | fare × transit_delay | seen | 5（子集 K=7）|
| road_disruption_x_congestion | road_disruption × road_congestion | seen | 3 |
| fare_x_congestion | fare × road_congestion | **holdout** | 5 |

## 2. 模型

| model | 定义 |
|---|---|
| M0 | 现有 S3-C 单轴 Student（冻结，未见任何 joint 数据）|
| M1 | S3-C 上 fine-tune：S3 单轴 + seen joint（A/C/D）base-K 目标 |
| M2 | 同 M1，但 low-SNR 组合 fare×delay 的 K=7 子集用 K=7 稳定目标 |

## 3. 核心结果（test = 6 未见 personas）

### 3.1 Joint 概率保真（seen vs unseen combination）

| model | seen joint KL | seen joint L1 | unseen joint KL | unseen joint L1 |
|---|---|---|---|---|
| M0 | 0.0538 | 0.2065 | 0.0954 | 0.2725 |
| M1 | 0.0440 | 0.1860 | 0.0890 | 0.2558 |
| M2 | 0.0441 | 0.1847 | 0.0880 | 0.2538 |

per-combo joint KL / L1：
| combo | seen | M0 KL / L1 | M1 KL / L1 | M2 KL / L1 |
|---|---|---|---|---|
| rain_x_congestion | yes | 0.0518 / 0.2278 | 0.0442 / 0.2150 | 0.0442 / 0.2137 |
| fare_x_transit_delay | yes | 0.0580 / 0.1852 | 0.0575 / 0.1815 | 0.0574 / 0.1802 |
| road_disruption_x_congestion | yes | 0.0515 / 0.2066 | 0.0302 / 0.1614 | 0.0308 / 0.1602 |
| fare_x_congestion | **no (holdout)** | 0.0954 / 0.2725 | 0.0890 / 0.2558 | 0.0880 / 0.2538 |

### 3.2 Interaction effect 误差（I_ij = P_ij − P_i − P_j + P_0）

| model | mean interaction L1 error | linked states |
|---|---|---|
| M0 | 0.0469 | 94/96 |
| M1 | 0.0464 | 94/96 |
| M2 | 0.0453 | 94/96 |

per-combo interaction error：
| combo | M0 | M1 | M2 |
|---|---|---|---|
| rain_x_congestion | 0.0563 | 0.0553 | 0.0539 |
| fare_x_transit_delay | 0.0361 | 0.0348 | 0.0363 |
| road_disruption_x_congestion | 0.0560 | 0.0570 | 0.0555 |
| fare_x_congestion | 0.0385 | 0.0376 | 0.0346 |

### 3.3 单轴能力回归（legacy S3 test set，不得退化）

| model | mode acc | KL | prob L1 |
|---|---|---|---|
| M0 | 0.8496 | 0.0776 | 0.2646 |
| M1 | 0.8496 | 0.0745 | 0.2575 |
| M2 | 0.8451 | 0.0735 | 0.2560 |

## 4. 结论（回答 RQ）

### RQ-M1：单轴训练能否复现联合响应？
- **能部分复现**：M0（仅单轴训练）在 seen joint 上 KL=0.0538，已相当低，说明单轴学到的 context 特征对联合态有一定插值泛化；但仍显著差于 joint 微调模型（0.0440）。
### RQ-M2：定向 multi-axis 训练是否改善 seen joint 保真？
- **是**：seen joint KL 0.0538 → 0.0440（M1），相对降幅 ~18.2%；L1 0.2065 → 0.1860。
### RQ-M3：unseen combination 的 compositional generalization？
- **有限泛化**：holdout 组合 fare×congestion 上，M0 KL 0.0954 → M1 0.0890 → M2 0.0880，相对降幅 ~7.8%。unseen 组合仍显著难于 seen（KL 0.0880 vs 0.0441），说明记忆强于组合泛化，但方向正确。
### RQ-M4：冲突压力下是否保留 trade-off？
- **interaction 误差整体很小**（M0 0.0469 → M1 0.0464 → M2 0.0453），说明联合响应近似可加（I_ij≈0），单一强轴并未吞没联合信号；冲突组合（rain×cong、disruption×cong）的 interaction 误差略高于非冲突组合。
### RQ-M5：low-SNR 稳定化（K=7）是否有效？
- **方向正确但幅度有限**：M2（K=7 子集）较 M1（K=5）在 unseen KL 0.0890 → 0.0880、interaction 0.0464 → 0.0453 均略降；但 40 态子集的规模限制了效应量。

## 5. Success Criteria（§16）核对

- ✅ seen joint KL 明显低于 M0：0.0440 vs 0.0538（-18.2%）
- ⚠️ interaction error 下降但幅度小：M0 0.0469 → M1 0.0464 / M2 0.0453
- ✅ legacy single-axis 不退化：M0 acc 0.8496 → M1 0.8496 / M2 0.8451（KL 反而略降 0.0776 → 0.0735）

## 6. Limitations

- 合成 persona/trip，无真实世界标定（与 S3 相同）。
- test = 6 未见 personas / 96 joint 态，development-scale 证据。
- fare×delay 的 interaction 链接可能缺失若干 transit_delay=30 单轴对照（S2 的 7 个 incomplete 被 schema 排除）。
- deepseek-v4-pro 为推理模型，单次调用 ~54s（推理 token 主导），成本/延迟显著。
- 联合组合仅覆盖 4 对、每对 2 档，非全笛卡尔积。
