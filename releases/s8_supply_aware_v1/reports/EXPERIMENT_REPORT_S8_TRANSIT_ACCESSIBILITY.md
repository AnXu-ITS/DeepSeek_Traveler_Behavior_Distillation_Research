# Historical S8 accessibility experiment — deprecated

This record describes the first supply-aware accessibility adaptation, including data splits, inherited generic weights, added PT features, loss settings and validation gates. Its active-mode travel-time inputs were later found to be incorrect. Numerical results below are preserved for historical inspection only. They must not be used as final S9 or current-paper evidence. The corrected S9 release and current training guide supersede this experiment for model use.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/releases/s8_supply_aware_v1/reports/EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md).

[Current research](<../../../docs/RESEARCH_DESIGN.md>) · [Training](<../../../docs/TRAINING.md>) · [Results](<../../../docs/RESULTS.md>) · [Data and access](<../../../docs/DATA_SOURCES.md>) · [Model use](<../../../docs/MODEL_USE.md>)

## Teacher transit probability by accessibility class

| class | n | Teacher P(PT) |
|---|---|---|
| A_excellent | 86 | 0.236 [0.194, 0.284] |
| B_good | 36 | 0.264 [0.191, 0.345] |
| C_moderate | 60 | 0.194 [0.155, 0.238] |
| D_poor | 78 | 0.123 [0.100, 0.152] |
| E_infeasible | 78 | 0.000 [0.000, 0.000] |

## Student probability fidelity on the accessibility test set

| model | KL | prob L1 | PT prob MAE |
|---|---|---|---|
| B0_S7W3 | 0.1984 [0.1659, 0.2295] | 0.5016 [0.4530, 0.5496] | 0.1336 [0.1036, 0.1631] |
| B1_S8 | 0.1980 [0.1568, 0.2414] | 0.4845 [0.4191, 0.5519] | 0.1184 [0.0863, 0.1527] |
| Teacher | — | — | — |

## Best-to-worst accessibility response

| model | ΔP_PT | n_groups |
|---|---|---|
| B0_S7W3 | 0.1631 [0.1376, 0.1910] | 12 |
| B1_S8 | 0.1561 [0.1188, 0.2001] | 12 |
| Teacher | 0.2117 [0.1547, 0.2715] | 12 |

## Pair and triplet accessibility monotonicity

| model | pair agreement | triplet agreement | n_pairs / n_triplets |
|---|---|---|---|
| B0_S7W3 | 0.6571 [0.4857, 0.8000] | 0.2609 [0.0870, 0.4348] | 35 / 23 |
| B1_S8 | 0.6857 [0.5143, 0.8286] | 0.3043 [0.1304, 0.4783] | 35 / 23 |
| Teacher | 0.8000 [0.6571, 0.9143] | 0.5217 [0.3043, 0.7391] | 35 / 23 |

## Transit feasibility violations and infeasible-state probability

| model | FVR | mean P(PT\|infeasible) | n_infeasible |
|---|---|---|---|
| B0_S7W3 | 0.0000 [0.0000, 0.0000] | 0.0978 [0.0693, 0.1342] | 12 |
| B1_S8 | 0.0000 [0.0000, 0.0000] | 0.0599 [0.0359, 0.0904] | 12 |

## Transit probability error by accessibility class

| model | Excellent | Good | Moderate | Poor | Infeasible |
|---|---|---|---|---|---|
| B0_S7W3 | 0.1137 [0.0706, 0.1663] | 0.1570 [0.1018, 0.2219] | 0.1556 [0.0546, 0.2798] | 0.1736 [0.0841, 0.2804] | 0.0975 [0.0691, 0.1338] |
| B1_S8 | 0.1108 [0.0559, 0.1762] | 0.1286 [0.0643, 0.2036] | 0.1769 [0.0742, 0.2941] | 0.1556 [0.0506, 0.2764] | 0.0596 [0.0356, 0.0902] |
| Teacher | — | — | — | — | — |

## Accessibility-loss ablation comparison

| metric(test) | B0 S7-W3 | R1 λ=0 | R2 λ=1.0 | Teacher |
|---|---|---|---|---|
| PT prob MAE | 0.1336 [0.1036, 0.1631] | 0.1147 [0.0856, 0.1457] | 0.1184 [0.0863, 0.1527] | — |
| mean P(PT\|infeasible) | 0.0978 [0.0693, 0.1342] | 0.0643 [0.0398, 0.0958] | 0.0599 [0.0359, 0.0904] | — |
| sensitivity ΔP_PT | 0.1631 [0.1376, 0.1910] | 0.1331 [0.1018, 0.1696] | 0.1561 [0.1188, 0.2001] | 0.2117 [0.1547, 0.2715] |
| monotonicity pair | 0.6571 [0.4857, 0.8000] | 0.6000 [0.4286, 0.7714] | 0.6857 [0.5143, 0.8286] | 0.8000 [0.6571, 0.9143] |
| monotonicity triplet | 0.2609 [0.0870, 0.4348] | 0.2174 [0.0435, 0.3913] | 0.3043 [0.1304, 0.4783] | 0.5217 [0.3043, 0.7391] |

## Generic-capability regression checks

|validation check|value|interpretation|
|---|---|---|
| legacy accuracy drop | -3.54 pp | ✅ ≤ 1 pp |
| legacy KL change | -35.83% | ✅ ≤ +10% |
| seen joint KL change | -15.44% | ✅ ≤ +10% |
| unseen joint KL change | -44.62% | ✅ ≤ +10% |
