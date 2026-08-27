# S9 Evidence Index

Index of the key evidence files backing the S9 retrain and freeze (S9 =
Supply-Aware Traveler Agent v2.0, retrained on the corrected walk/bike speeds).
S8 (deprecated) audit trail: `docs/S8_DEPRECATION.md`.

## Data-error discovery & fix

| Item | Evidence | Location |
|---|---|---|
| S8 data error (walk/bike at road free-flow speed) | deprecation record | `docs/S8_DEPRECATION.md` |
| Measured S8 effective speeds (walk/bike ~29 km/h) | analysis over 114-OD pool + 338-state records | (session analysis, quoted in `docs/S8_DEPRECATION.md` and `provenance/data_fix.json`) |
| S9 fix (walk 1.34 / bike 4.17 m/s) | code | `src/traveler_distillation/accessibility/gtfs_accessibility.py` (`SupplyIndex`), `src/traveler_distillation/matsim/adapter.py` (leg-routing weights) |
| Fix verification (walk 3.28 / bike 10.14 / car 31.09 km/h) | measured | `releases/s9_supply_aware_v2/accessibility/feature_definitions.md` |
| MATSim-side walk/bike limitation (network modes at freespeed) | documented | `releases/s9_supply_aware_v2/FINAL_S9_FREEZE.md`, `README.md` |
| S8 legacy dataset archive | files | `data/singapore_accessibility_s8_legacy/` |

## S9 dataset + teacher

| Item | Evidence | Location |
|---|---|---|
| S9 dataset (336 states, corrected) | records | `data/singapore_accessibility/records.jsonl` |
| Splits (234/51/51; OD 79/14/20; persona 28/6/6) | split manifest | `data/singapore_accessibility/split_manifest.json` |
| Teacher labeling (1,502 valid calls, K=3×89/K=5×247, 0 incomplete) | generation manifest | `data/singapore_accessibility/generation_manifest.json` |
| Teacher labels | aggregated targets | `data/singapore_accessibility/states_with_teacher.jsonl` |
| Teacher gradient (P(PT): A .448/B .429/C .372/D .189/E .000) | derived | `reports/EXPERIMENT_REPORT_S9_TRANSIT_ACCESSIBILITY_V2.md` |

## S9 training & evaluation

| Item | Evidence | Location |
|---|---|---|
| Training config | `configs/student_s9.yaml` (+ training_s9.json in release) | repo / `releases/s9_supply_aware_v2/config/` |
| Training run | history + val metrics | `outputs/student_s9/` |
| Accessibility eval (test n=51, double holdout) | metrics | `outputs/s9_accessibility_eval/eval_metrics.json` |
| Regression gates | metrics | `outputs/s9_regression/eval_metrics.json` |
| Unseen-OD audit | audit | `outputs/s9_unseen_od/unseen_od_audit.json` |
| Experiment report | report | `reports/EXPERIMENT_REPORT_S9_TRANSIT_ACCESSIBILITY_V2.md` (frozen copy in release) |

## Freeze artifacts

| Item | Location |
|---|---|
| Freeze executor | `scripts/freeze_s9_release.py` |
| Final freeze record | `releases/s9_supply_aware_v2/FINAL_S9_FREEZE.md` |
| Release README | `releases/s9_supply_aware_v2/README.md` |
| Reproduction gate | `releases/s9_supply_aware_v2/reports/reproduction_gate/` |
| Checksums | `releases/s9_supply_aware_v2/checksums/SHA256SUMS.txt` |
| Guard | `src/traveler_distillation/student/release_guard.py` (+ `tests/test_release_guard.py`) |
| Git tag | `s9-supply-aware-v2.0` |

## S8 (deprecated) — kept intact

- Release: `releases/s8_supply_aware_v1/` (byte-identical to its freeze, tag `s8-supply-aware-v1.0`).
- S8 Phase C results (`outputs/singapore_phase_c/`) are void for the paper;
  the rerun uses `outputs/singapore_phase_c_s9/`.
