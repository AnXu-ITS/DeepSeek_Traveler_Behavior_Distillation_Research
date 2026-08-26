# S8 Evidence Index

Index of the key evidence files backing the S8 freeze and the Singapore
Phase A / B / B.5 supply validation. All paths are relative to the repository
root. Frozen copies marked `(frozen)` live inside release directories; the
canonical live copies are listed alongside.

## Model line

| Stage | Evidence | Location |
|---|---|---|
| S5 multi-axis distillation | experiment report | `outputs/EXPERIMENT_REPORT_S5_MULTI_AXIS.md` (frozen: `releases/s7_w3_generic_core_v1/reports/`) |
| S5 joint eval | metrics | `outputs/s5_joint_eval/eval_metrics.json` |
| S6 causal-mechanism audit | experiment report | `outputs/EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md` (frozen: `releases/s7_w3_generic_core_v1/reports/`) |
| S6 audit data | states + teacher labels | `data/causal_audit/` |
| S7 mechanism-aware fine-tuning | experiment report | `outputs/EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md` (frozen: `releases/s7_w3_generic_core_v1/reports/`) |
| S7-W3 freeze | final freeze record | `releases/s7_w3_generic_core_v1/FINAL_S7_W3_FREEZE.md` (frozen) |
| S7-W3 seed robustness | report + metrics | `releases/s7_w3_generic_core_v1/reports/SEED_ROBUSTNESS_S7_W3.md`, `metrics/seeds/` (frozen) |
| S8 schema audit (Case B) | audit report | `reports/S8_SCHEMA_AUDIT.md` (frozen: `releases/s8_supply_aware_v1/reports/`) |
| S8 training | config + history | `configs/student_s8.yaml`, `outputs/student_s8/` (R2 final), `outputs/student_s8_r1_lam0/` (R1 ablation) |
| S8 accessibility eval | metrics | `outputs/s8_accessibility_eval/eval_metrics.json` (R2), `outputs/s8_accessibility_eval_r1/eval_metrics.json` (R1) |
| S8 regression gate | metrics | `outputs/s8_regression/eval_metrics.json` (R2), `outputs/s8_regression_r1/eval_metrics.json` (R1) |
| S8 unseen-OD audit | audit | `outputs/s8_unseen_od/unseen_od_audit.json` |
| S8 experiment report | report | `reports/EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md` (frozen: `releases/s8_supply_aware_v1/reports/`) |

## S8 dataset + teacher

| Item | Evidence | Location |
|---|---|---|
| S8 accessibility dataset | data notes | `data/singapore_accessibility/DATA_README.md` |
| S8 states (338) | records | `data/singapore_accessibility/records.jsonl` |
| S8 splits (triple holdout) | split manifest | `data/singapore_accessibility/split_manifest.json` |
| Teacher labeling (1,468 calls) | generation manifest | `data/singapore_accessibility/generation_manifest.json` |
| Teacher labels | aggregated targets | `data/singapore_accessibility/states_with_teacher.jsonl` |
| Teacher repeats | raw repeats | `data/singapore_accessibility/repeat_records.jsonl` |
| K=5 boundary samples | boundary pairs | `data/singapore_accessibility/boundary_pairs.json` |
| Teacher provenance (frozen) | provenance JSON | `releases/s8_supply_aware_v1/provenance/teacher_s8.json` |

## Singapore supply (OSM + GTFS)

| Item | Evidence | Location |
|---|---|---|
| Supply data notes | README | `data/singapore/DATA_README.md` |
| OSM snapshot | query + stats | `data/singapore/osm/query.txt`, `data/singapore/osm/network_stats.json` |
| GTFS snapshot | metadata + checksum | `data/singapore/gtfs/README_snapshot.md`, `data/singapore/gtfs/raw/source_metadata.json`, `data/singapore/gtfs/raw/checksum.sha256` |
| Transit prep | prep meta + reports | `data/singapore/transit/prep_meta.json`, `data/singapore/transit/stop_snapping_report.json`, `data/singapore/transit/route_routing_report.json` |
| Supply provenance (frozen) | provenance JSONs | `releases/s8_supply_aware_v1/provenance/singapore_supply.json`, `osm_manifest.json`, `gtfs_manifest.json` |

## Singapore pipeline phases

| Phase | Evidence | Location |
|---|---|---|
| Phase A gate (100 agents, exit=0, 4 modes) | gate report + result | `outputs/singapore_phase_a/PHASE_A_GATE.md`, `outputs/singapore_phase_a/smoke_100/phase_a_result.json` |
| Phase B scaling (100/500/1000) | metrics + report | `outputs/singapore_phase_b/scale_metrics.json`, `outputs/singapore_phase_b/scale_report.md` |
| Phase B.5 PT validity | validity evidence | `outputs/singapore_phase_b5/pt_validity_v3/`, `outputs/singapore_phase_b5/cache_smoke/pt_validity.json` |
| Phase B.5C capacity calibration | calibration report | `outputs/singapore_phase_b5/calibration/calibration_report.md` (per PROGRESS.md §Phase B.5C) |
| Loading diagnostic (N* decision) | diagnostic | `outputs/singapore_phase_b5/loading_diagnostic.md` |
| S8 Singapore smoke (freeze gate) | smoke result | `outputs/s8_singapore_smoke/s8_smoke_result.json` (frozen copy: `releases/s8_supply_aware_v1/reports/reproduction_gate/singapore_smoke_result.json`) |

## Freeze artifacts (this release)

| Item | Location |
|---|---|
| Freeze instructions | `S8_BACKUP_FREEZE_INSTRUCTIONS.md` |
| Freeze executor | `scripts/freeze_s8_release.py` |
| Final freeze record | `releases/s8_supply_aware_v1/FINAL_S8_FREEZE.md` |
| Release README | `releases/s8_supply_aware_v1/README.md` |
| Reproduction gate | `releases/s8_supply_aware_v1/reports/reproduction_gate/` |
| Checksums | `releases/s8_supply_aware_v1/checksums/SHA256SUMS.txt` |
| Phase C guard | `src/traveler_distillation/student/release_guard.py`, `src/traveler_distillation/matsim/s8_adapter.py`, `scripts/singapore/run_s8_smoke.py` |
