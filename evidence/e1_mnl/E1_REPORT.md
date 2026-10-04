# Historical MNL baseline comparison

The E1 comparison evaluates the earlier MNL-B baseline against the frozen neural model under its stated supervision and deployment conditions. MNL-B and the current same-feature MNL-S are different baselines. Inherited generic/replay supervision prevents interpreting the older comparison as an architecture-only effect. Current matched training settings and MNL-S coefficients are documented in the training guide.

Original source: [archived document](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/blob/de906ac8a3efae692797bd08e340375107b8e52a/evidence/e1_mnl/E1_REPORT.md).

[Current research](<../../docs/RESEARCH_DESIGN.md>) · [Training](<../../docs/TRAINING.md>) · [Results](<../../docs/RESULTS.md>) · [Data and access](<../../docs/DATA_SOURCES.md>) · [Model use](<../../docs/MODEL_USE.md>)

## MNL-B coefficient estimates

|coefficient|estimate| SE | z |
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

## MNL-B and frozen Student prediction fidelity

|model| mode acc ↑ | KL ↓ | L1 ↓ | PT-MAE ↓ | FVR ↓ | P(PT\|inf) ↓ |
|---|---|---|---|---|---|
| MNL-B | 0.8627 [0.7647, 0.9412] | 0.1648 [0.1028, 0.2410] | 0.3390 [0.2480, 0.4443] | 0.1527 [0.1064, 0.2073] | 0.0833 [0.0000, 0.2500] | 0.0957 [0.0127, 0.2201] |
| S9(frozen, G2 recomputed values match) | 0.9412 [0.8627, 1.0000] | 0.1704 [0.0920, 0.2939] | 0.3392 [0.2575, 0.4345] | 0.1346 [0.0921, 0.1844] | 0.0833 [0.0000, 0.2500] | 0.2130 [0.0666, 0.3908] |
| Teacher |reference|reference|reference|reference|reference|reference|

## Transit-accessibility response by supply class

|model| P(PT\|A) | P(PT\|B) | P(PT\|C) | P(PT\|D) | P(PT\|E) | ΔP best−worst | pair agr ↑ | triplet agr ↑ |
|---|---|---|---|---|---|---|---|---|
| MNL-B | 0.2003 | 0.1406 | 0.2452 | 0.2367 | 0.0957 | 0.1045 | 0.5789 | 0.1923 |
| S9(frozen) | 0.3657 | 0.4519 | 0.3260 | 0.2153 | 0.2130 | 0.1527 | 0.6579 | 0.2692 |
| Teacher(frozen reference) | 0.4481 | 0.4290 | 0.3724 | 0.1888 | 0.0001 | 0.4480 | 0.7895 | 0.6154 |

## Legacy and joint-context performance comparison

|model| acc ↑ | KL ↓ | L1 ↓ | ΔP gap ↓ | sign agr ↑ | seen KL ↓ | unseen KL ↓ | inter L1 err ↓ |
|---|---|---|---|---|---|---|---|---|
| MNL-B | 0.6681 [0.6062, 0.7301] | 0.3863 [0.3299, 0.4471] | 0.5083 [0.4555, 0.5608] | 0.0743 [0.0633, 0.0857] | 0.5946 [0.5479, 0.6406] | 0.4047 [0.2941, 0.5272] | 0.3673 [0.2567, 0.4783] | 0.0424 [0.0337, 0.0519] |
| S9(frozen, `s9_regression` S8 key) | 0.8894 [0.8496, 0.9292] | 0.0514 [0.0423, 0.0606] | 0.1886 [0.1641, 0.2131] | 0.0489 [0.0430, 0.0551] | 0.7076 [0.6565, 0.7559] | 0.0407 [0.0304, 0.0526] | 0.0537 [0.0218, 0.0924] | 0.0504 [0.0404, 0.0611] |

## MNL scenario execution checks

| scenario | exit | stuck persons | pt board/alight | gate |
|---|---|---|---|---|
| C0_baseline | 0 | 265 | 7038/6178 | PASS |
| C1_heavy_rain | 0 | 272 | 7355/6443 | PASS |
| C2_fare_increase | 0 | 321 | 8572/7526 | PASS |
| C3_transit_delay | 0 | 214 | 5575/4904 | PASS |
| C4_road_disruption | 0 | 275 | 7289/6403 | PASS |
| C5_joint_rain_delay | 0 | 232 | 5975/5248 | PASS |

## MNL and frozen Student mode shares by scenario

| scenario | MNL car/pt/bike/walk | S9 car/pt/bike/walk(frozen) |
|---|---|---|
| C0_baseline | 36.9%/31.7%/21.8%/9.6% | 28.3%/25.3%/34.0%/12.4% |
| C1_heavy_rain | 37.0%/33.4%/21.3%/8.4% | 37.5%/45.3%/13.3%/3.8% |
| C2_fare_increase | 36.3%/39.1%/17.9%/6.7% | 29.9%/25.7%/32.5%/11.8% |
| C3_transit_delay | 37.2%/24.7%/24.8%/13.3% | 29.0%/10.2%/35.8%/25.0% |
| C4_road_disruption | 33.9%/32.9%/23.3%/9.8% | 1.8%/37.5%/41.8%/19.0% |
| C5_joint_rain_delay | 37.4%/26.8%/24.4%/11.5% | 37.6%/18.6%/24.5%/19.2% |

## MNL and frozen Student changes from baseline

| scenario | MNL Δpt | S9 Δpt | MNL Δcar | S9 Δcar | MNL Δboardings | S9 Δboardings | MNL ΔVKT | S9 ΔVKT |
|---|---|---|---|---|---|---|---|---|
| C1_heavy_rain | +1.6pp | +20.0pp | +0.1pp | +9.3pp | +317 | +4148 | +124 | +8838 |
| C2_fare_increase | +7.4pp | +0.4pp | -0.5pp | +1.6pp | +1534 | +61 | -507 | +1859 |
| C3_transit_delay | -7.0pp | -15.1pp | +0.4pp | +0.8pp | -1463 | -3139 | +370 | +746 |
| C4_road_disruption | +1.2pp | +12.1pp | -2.9pp | -26.5pp | +251 | +2924 | -2649 | -30996 |
| C5_joint_rain_delay | -5.0pp | -6.7pp | +0.5pp | +9.4pp | -1063 | -1312 | +504 | +8884 |
