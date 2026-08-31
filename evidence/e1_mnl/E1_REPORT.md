# E1 MNL-B Baseline — 实验报告

> 设计书：`TRC_AIT_5_E1_MNL_BASELINE_DESIGN.md` v0.3（MNL-B vs S9，Teacher 仅决策层参照）
> 本报告所有数字由 `scripts/make_e1_report.py` 从证据 JSON 生成，无手抄数字。
> MNL 系数冻结：`outputs/e1_mnl/mnl_b_coefs.json`（dataset SHA256 `e620d386486b8262…`）

## 0. Gate 汇总

| gate | 内容 | 结果 |
|---|---|---|
| G1 | 估计门（Hessian PD、SE 有限、无 blowup、10 重启收敛 spread=2.49e-08） | PASS |
| G2 | S9 在 51 态复算与冻结 `s9_accessibility_eval` 逐位一致（28 项） | PASS |
| G3 | 管线默认路径逐位回归 + MNL × C0 Phase C gate | 见 §5.1 |
| G4 | 估计确定性（逐位复跑 Δ=0） | PASS |

规格选择：预注册 {S1, S2, S3} → 可识别规格 {S1, S3} → val(51 态) log-likelihood 选出 **S3**（k=17）；排除规格见 `mnl_b_coefs.json` `excluded_specs`。

v0.3 不可识别修正（吸收进 ASC，经典 MNL 惯例）：`weather_exposure`、`reliability_delay_min` （模式内恒定）与 `car_own_car`（≡ asc_car，因可用 car 全部有车）从效用移除；S2 的 commute×mode 项在冻结数据不可识别，被 val 选模规则排除。

## 1. T5 — MNL-B 系数表（规格 S3，n_train=234）

| 系数 | 估计值 | SE | z |
|---|---|---|---|
| asc_car | -1.3964 | 1.7104 | -0.82 |
| asc_pt | -1.6946 | 0.7984 | -2.12 |
| asc_bike | 0.3143 | 0.3584 | 0.88 |
| tt | -0.0286 | 0.0071 | -4.03 |
| cost | 0.2568 | 0.2003 | 1.28 |
| cost_low | -0.0749 | 0.0853 | -0.88 |
| cost_high | -0.0161 | 0.1114 | -0.14 |
| access | 0.1461 | 0.0578 | 2.53 |
| transfers | 0.5490 | 0.7303 | 0.75 |
| pass_pt | 0.5513 | 0.4882 | 1.13 |
| habit | 0.8619 | 0.2635 | 3.27 |
| lim_wb | -0.5112 | 0.3474 | -1.47 |
| age65_wb | -0.1378 | 0.7757 | -0.18 |
| hard_car | 0.1857 | 1.0326 | 0.18 |
| hard_pt | 0.0489 | 0.5052 | 0.10 |
| flex_car | 0.0983 | 0.4442 | 0.22 |
| flex_pt | -0.2551 | 0.3326 | -0.77 |

- train log-lik = -155.2696 · val log-lik = -30.7017 · Hessian cond = 1.85e+05 · λmin = 2.69e-01
- **诚实边界**：冻结 Teacher 标签对货币成本近乎不敏感（cost–tt 相关：car ρ≈0.72、pt ρ≈0.47），β_cost 为正（z=1.28），β_access、β_transfers 亦为正——预期 C2（票价）响应方向可能反转。这是 MNL-B 在蒸馏监督下的真实估计性质，如实呈现，不修规格。

## 2. T1 — 决策层保真度（51 态 Singapore test，unseen persona × unseen OD，vs 冻结 Teacher 标签）

| 模型 | mode acc ↑ | KL ↓ | L1 ↓ | PT-MAE ↓ | FVR ↓ | P(PT\|inf) ↓ |
|---|---|---|---|---|---|
| MNL-B | 0.8627 [0.7647, 0.9412] | 0.1648 [0.1028, 0.2410] | 0.3390 [0.2480, 0.4443] | 0.1527 [0.1064, 0.2073] | 0.0833 [0.0000, 0.2500] | 0.0957 [0.0127, 0.2201] |
| S9（冻结，G2 复算一致） | 0.9412 [0.8627, 1.0000] | 0.1704 [0.0920, 0.2939] | 0.3392 [0.2575, 0.4345] | 0.1346 [0.0921, 0.1844] | 0.0833 [0.0000, 0.2500] | 0.2130 [0.0666, 0.3908] |
| Teacher | 参照 | 参照 | 参照 | 参照 | 参照 | 参照 |

配对差分（MNL − S9，bootstrap B=2000 seed=42；正值 = MNL 更差）：

- PT-MAE Δ = +0.0181 [-0.0340, +0.0737]（CI 含 0）
- P(PT\|inf) Δ = -0.1173 [-0.1963, -0.0463]（CI 不含 0）
- FVR rate Δ = +0.0000（CI 含 0）
- pair monotonicity Δ = -0.0789（CI 含 0）

## 3. T2 — Supply-aware 可达性响应（同一 51 态）

| 模型 | P(PT\|A) | P(PT\|B) | P(PT\|C) | P(PT\|D) | P(PT\|E) | ΔP best−worst | pair agr ↑ | triplet agr ↑ |
|---|---|---|---|---|---|---|---|---|
| MNL-B | 0.2003 | 0.1406 | 0.2452 | 0.2367 | 0.0957 | 0.1045 | 0.5789 | 0.1923 |
| S9（冻结） | 0.3657 | 0.4519 | 0.3260 | 0.2153 | 0.2130 | 0.1527 | 0.6579 | 0.2692 |
| Teacher（冻结参照） | 0.4481 | 0.4290 | 0.3724 | 0.1888 | 0.0001 | 0.4480 | 0.7895 | 0.6154 |

- 解读：MNL-B 无供给特征，仅靠不可行态的 120 min 哨兵 tt 压低 P(PT)（FVR 0.0833 与 S9 相同、P(PT\|inf) 0.096 反而低于 S9 0.213）；但其 P(PT) 曲线不随可达性等级单调（C/D 类反而高于 A 类），monotonicity 与 sensitivity 均低于 S9，距 Teacher 梯度（ΔP 0.448）差距更大——supply-aware 结构化响应缺失。

## 4. T3 — 弹性与联合响应（frozen regression benchmark：legacy 226 / seen 72 / unseen 24）

| 模型 | acc ↑ | KL ↓ | L1 ↓ | ΔP gap ↓ | sign agr ↑ | seen KL ↓ | unseen KL ↓ | inter L1 err ↓ |
|---|---|---|---|---|---|---|---|---|
| MNL-B | 0.6681 [0.6062, 0.7301] | 0.3863 [0.3299, 0.4471] | 0.5083 [0.4555, 0.5608] | 0.0743 [0.0633, 0.0857] | 0.5946 [0.5479, 0.6406] | 0.4047 [0.2941, 0.5272] | 0.3673 [0.2567, 0.4783] | 0.0424 [0.0337, 0.0519] |
| S9（冻结，`s9_regression` S8 键） | 0.8894 [0.8496, 0.9292] | 0.0514 [0.0423, 0.0606] | 0.1886 [0.1641, 0.2131] | 0.0489 [0.0430, 0.0551] | 0.7076 [0.6565, 0.7559] | 0.0407 [0.0304, 0.0526] | 0.0537 [0.0218, 0.0924] | 0.0504 [0.0404, 0.0611] |

- 核心结论方向：静态保真度（T1）两者相当，但**动态/多条件**上差距悬殊——MNL-B 的 legacy KL （0.386 vs 0.051）、acc（0.668 vs 0.889）、sign agreement（0.595 vs 0.708）均大幅落后于 S9；seen/unseen joint KL 差约一个量级。这正是 E1 结论目标『MNL 可作为静态选择基线，但 S9 在动态、多条件上更有优势』的决策层证据。

## 5. T4 — Phase C 仿真镜像（MNL × C0–C5，N=10,000，seed 2026，capacity 0.3/0.3）

### 5.1 Gate（MNL 各情景）

| scenario | exit | stuck persons | pt board/alight | gate |
|---|---|---|---|---|
| C0_baseline | 0 | 265 | 7038/6178 | PASS |
| C1_heavy_rain | 0 | 272 | 7355/6443 | PASS |
| C2_fare_increase | 0 | 321 | 8572/7526 | PASS |
| C3_transit_delay | 0 | 214 | 5575/4904 | PASS |
| C4_road_disruption | 0 | 275 | 7289/6403 | PASS |
| C5_joint_rain_delay | 0 | 232 | 5975/5248 | PASS |

### 5.2 决策侧 mode share（student 口径）

| scenario | MNL car/pt/bike/walk | S9 car/pt/bike/walk（冻结） |
|---|---|---|
| C0_baseline | 36.9%/31.7%/21.8%/9.6% | 28.3%/25.3%/34.0%/12.4% |
| C1_heavy_rain | 37.0%/33.4%/21.3%/8.4% | 37.5%/45.3%/13.3%/3.8% |
| C2_fare_increase | 36.3%/39.1%/17.9%/6.7% | 29.9%/25.7%/32.5%/11.8% |
| C3_transit_delay | 37.2%/24.7%/24.8%/13.3% | 29.0%/10.2%/35.8%/25.0% |
| C4_road_disruption | 33.9%/32.9%/23.3%/9.8% | 1.8%/37.5%/41.8%/19.0% |
| C5_joint_rain_delay | 37.4%/26.8%/24.4%/11.5% | 37.6%/18.6%/24.5%/19.2% |

### 5.3 C1–C5 vs C0 paired 差分（MNL vs S9 冻结）

| scenario | MNL Δpt | S9 Δpt | MNL Δcar | S9 Δcar | MNL Δboardings | S9 Δboardings | MNL ΔVKT | S9 ΔVKT |
|---|---|---|---|---|---|---|---|---|
| C1_heavy_rain | +1.6pp | +20.0pp | +0.1pp | +9.3pp | +317 | +4148 | +124 | +8838 |
| C2_fare_increase | +7.4pp | +0.4pp | -0.5pp | +1.6pp | +1534 | +61 | -507 | +1859 |
| C3_transit_delay | -7.0pp | -15.1pp | +0.4pp | +0.8pp | -1463 | -3139 | +370 | +746 |
| C4_road_disruption | +1.2pp | +12.1pp | -2.9pp | -26.5pp | +251 | +2924 | -2649 | -30996 |
| C5_joint_rain_delay | -5.0pp | -6.7pp | +0.5pp | +9.4pp | -1063 | -1312 | +504 | +8884 |

- 方向一致性：C1/C3/C4/C5 的 Δpt/Δcar 符号与 S9 一致（幅度更弱）；**C2 方向反转已实测确认**（MNL-B 正成本系数所致，见 §1 诚实边界与 T5 系数表）。


## 6. 诚实边界

- MNL-B 拟合自 Teacher 蒸馏监督（mean-probability 软标签），**不是**真实出行调查/revealed-preference 数据；不得声称『标定到真实新加坡行为』。
- MNL-B 只做 mode choice（departure shift ≡ 0）；S9 有 departure head，shift 行不可比。
- 经典 MNL-B 效用不含供给特征（供给差距部分源于信息集差异）；E 类响应靠 120 min 哨兵 tt 压制，属『绕过』而非『理解』不可行性。
- 336 态 benchmark 的 context 全部为 baseline（C0 型）；新加坡 C1–C5 状态没有 Teacher 标签（可选 S6 probes 默认不做）。
- Phase C 差分与冻结口径一致（paired、无 bootstrap、~20% transit 截断为 10k+0.3 设置的既有特性）。
- 全部 MNL 数字来自本目录 JSON；冻结数字来自 `outputs/s9_accessibility_eval`、`outputs/s9_regression`、`outputs/singapore_phase_c_s9`。

## 7. No-Fabrication 状态

- 无任何估计值/模拟值冒充实测；全部 MNL 数字（含 T4）均由本脚本从证据 JSON 填入。
- 数字一致性建议由 `ccf-integrity-auditor` 复核。
