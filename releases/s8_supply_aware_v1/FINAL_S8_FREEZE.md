# FINAL_S8_FREEZE.md

## Identity

```text
Model: S8
Release: Supply-Aware Traveler Agent v1.0
Base: S7-W3 Generic Behavioral Core v1.0
Parameters: 24,562
Schema evolution: Case B
Status: FROZEN
```

- **Checkpoint**: `releases/s8_supply_aware_v1/checkpoint/model.pt`
  - SHA256 `f8232dde4442d882982ffb3cfaecb150b27c3e3ddb9fcc3738166b76ce823885`
  - Source: `outputs/student_s8/checkpoints/best.pt` (byte-identical copy, saved 2026-08-26T03:58:45Z)
  - Seed 42 · best epoch 26 (early stop 41) · 24,562 parameters · arch `student_s8_v1`
  - Init checkpoint: `releases/s7_w3_generic_core_v1/checkpoint/model.pt` (FROZEN S7-W3 release)
  - λ_accessibility = 1.0 (R2 selected) · λ_mechanism = 1.0 · λ_broken = 1.0
  - Optimizer / scheduler state: **not present** (S8 `best.pt` stores model weights only)
- **Config**: `config/student_s8.yaml`, `config/accessibility_features.yaml`,
  `config/training_s8.yaml/json` (R1 λ_acc=0 ablation recorded; R2 selected)
- **Schema**: `schema/input_schema_s8.json`, `schema/output_schema_s8.json`,
  `schema/feature_order_s8.txt`, `schema/schema_diff_vs_s7.json` (Case B)
- **Normalization**: `normalization/` (S7 fields verbatim + 6 new S8 fields fit on S8 train)
- **Accessibility**: `accessibility/feature_definitions.md`, `accessibility/feature_schema.json`
- **Provenance**: `provenance/singapore_supply.json`, `osm_manifest.json`,
  `gtfs_manifest.json`, `teacher_s8.json`
- **Data**: `data_manifest/s8_accessibility_dataset.json`, `s8_split_manifest.json`,
  `s8_generation_manifest.json`
- **Metrics**: `metrics/final_metrics.json`, `metrics/comparison_s7_vs_s8.csv`
- **Reports**: `reports/S8_SCHEMA_AUDIT.md`, `reports/EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md`,
  `reports/S8_EVIDENCE_INDEX.md`, `reports/reproduction_gate/`
- **Code**: `code_manifest/` (git commits, worktree status, environment, uncommitted inventory)

## Scope

S8 completes the training line of the research project and is frozen as the
supply-aware extension of the generic behavioral core:

- **Base**: S7-W3 Generic Behavioral Core v1.0 (single/multi-axis distillation,
  causal audit, mechanism-aware fine-tuning, seed-stable) — permanently retained
  as the paper's generic baseline.
- **S8**: real-supply transit accessibility adaptation on Singapore OSM + GTFS
  supply; Case B schema evolution (+6 city-independent alternative features,
  alt_encoder 14→20, +192 parameters); trained with replay 2:1:1:1 from the
  frozen S7-W3 weights; Stop Rule met (§33).
- **Key results (test = unseen persona × unseen OD, 95% paired bootstrap)**:
  PT-prob MAE 0.1336 → 0.1184 (Δ −0.0152*); mean P(PT|infeasible) 0.0978 →
  0.0599 (Δ −0.0379*); pair monotonicity 0.657 → 0.686; legacy KL −35.8%;
  seen joint −15.4%; unseen joint −44.6%; mechanism no regression; regression
  gate all pass; identity leakage 0.
- **Phase C** (Singapore real-network scenario experiments) may only LOAD this
  frozen checkpoint.

## Claims allowed

- supply-aware traveler agent
- city-independent transit-accessibility representation
- real-supply-conditioned behavioral adaptation
- generic behavioral core augmented with transferable supply attributes

## Claims prohibited

- trained on real Singapore traveler behavior
- universally validated across cities
- full causal reasoning
- exact reproduction of Singapore demand

## Freeze record (steps 1–15)

| Step | Item | Status |
|---|---|---|
| 1 | Unique final checkpoint located & copied (`checkpoint/model.pt`, best_epoch 26, 24,562 params) | ✅ |
| 2 | Config frozen (`student_s8.yaml`, `accessibility_features.yaml`, `training_s8.yaml/json` incl. R1/R2) | ✅ |
| 3 | Case B schema diff frozen (`schema_diff_vs_s7.json` + input/output schema + feature order) | ✅ |
| 4 | Transit accessibility definitions frozen (`accessibility/feature_definitions.md` + `feature_schema.json`) | ✅ |
| 5 | Singapore supply provenance frozen (OSM/GTFS manifests + `singapore_supply.json`) | ✅ |
| 6 | S8 dataset manifest frozen (338 states, triple holdout, leakage=0) | ✅ |
| 7 | Teacher provenance frozen (1,468 calls, prompt `teacher_s8_accessibility_v0.1`, 0 incomplete) | ✅ |
| 8 | Pre-freeze reproduction gate — see "Reproduction gate" below | ✅ PASS |
| 9 | Final metrics frozen (`metrics/final_metrics.json`, `comparison_s7_vs_s8.csv`) | ✅ |
| 10 | Key reports copied + `S8_EVIDENCE_INDEX.md` (S5/S6/S7/S8 + Phase A/B/B.5) | ✅ |
| 11 | Code manifest (S8 code commit `65e505f`, worktree status, environment) | ✅ |
| 12 | Git commit + annotated tag `s8-supply-aware-v1.0` | ✅ |
| 13 | SHA256 checksums (`checksums/SHA256SUMS.txt`) | ✅ |
| 14 | This file | ✅ |
| 15 | Write guard: read-only release + `release_guard.py` hard assertion (S8 release added; tests 158/158) | ✅ |

## Reproduction gate (Step 8 — frozen checkpoint re-evaluation, no retraining)

Same test-only pipelines as the original S8 evaluation, run against the frozen
`checkpoint/model.pt`; results at `reports/reproduction_gate/`.

| Gate metric | Historical | Re-run | Δ |
|---|---|---|---|
| PT prob MAE (S8) | 0.1184 | 0.1184 | 0.0000 |
| mean P(PT\|infeasible) (S8) | 0.0599 | 0.0599 | 0.0000 |
| pair monotonicity (S8) | 0.6857 | 0.6857 | 0.0000 |
| triplet monotonicity (S8) | 0.3043 | 0.3043 | 0.0000 |
| sensitivity ΔP_PT (S8) | 0.1561 | 0.1561 | 0.0000 |
| accessibility KL (S7-W3 B0) | 0.1984 | 0.1984 | 0.0000 |
| PT prob MAE (S7-W3 B0) | 0.1336 | 0.1336 | 0.0000 |
| unseen OD n_test states / all_unseen | 53 / True | 53 / True | 0.0000 |
| legacy accuracy (S8) | 0.8805 | 0.8805 | 0.0000 |
| legacy KL (S8) | 0.0446 | 0.0446 | 0.0000 |
| seen joint KL (S8) | 0.0356 | 0.0356 | 0.0000 |
| unseen joint KL (S8) | 0.0468 | 0.0468 | 0.0000 |
| legacy accuracy (S7-W3) | 0.8451 | 0.8451 | 0.0000 |
| mechanism parking G_med (S8) | 0.2554 | 0.2554 | 0.0000 |
| mechanism congestion Gap_shortcut (S8) | 0.2847 | 0.2847 | 0.0000 |

**Verdict: 16/16 gate metrics reproduce the historical values exactly
(Δ = 0.0000).** (`scripts/freeze_s8_release.py verify-gate` re-runs this
comparison.)

### Singapore smoke (frozen Phase C settings: capacity factors 0.3/0.3, N=100 of N*=10,000)

- S8 inference → adapter: schema round-trip asserted (12 alt fields, encoder
  input 20, `schema_match=True`);
- PT accessibility pipeline: 98 feasible / 2 infeasible real-supply pt vectors
  over 100 agents;
- MATSim minimal execution: **exit = 0**; legs executed across all four modes
  (bike/car/pt/walk); student modes walk 43 / bike 33 / car 20 / pt 4; executed
  outbound walk 46 / bike 33 / car 17 / pt 4; 9 PT boardings; 20,966 transit
  vehicle departures; no stuck agents.
- Evidence: `outputs/s8_singapore_smoke/s8_smoke_result.json` (frozen copy in
  `reports/reproduction_gate/singapore_smoke_result.json`).

## Code version

- S8 training/evaluation code commit: `65e505f` ("S8: transit accessibility
  adaptation results (Case B, R2 lambda_accessibility=1.0, regression gate
  all-pass, stop rule met)"), branch `main`.
- Freeze commit: `freeze: S8 supply-aware traveler agent v1.0` (this release
  directory, `scripts/freeze_s8_release.py`, the S8 adapter
  `src/traveler_distillation/matsim/s8_adapter.py`, the smoke runner
  `scripts/singapore/run_s8_smoke.py`, the extended release guard, and
  `S8_BACKUP_FREEZE_INSTRUCTIONS.md`).
- Tag: `s8-supply-aware-v1.0` (annotated).
- Environment: Python 3.12.13 · torch 2.13.0+cpu · OpenJDK 25.0.4 · MATSim 2026.0.
- Pre-freeze verification: test suite 155 passed / 0 error (as stated in the
  freeze instructions); after the guard extension 158 passed / 0 error.

## Data identity (summary; full record in `data_manifest/`)

| Dataset | States | Splits | Teacher | Incomplete / leakage |
|---|---|---|---|---|
| S8 Singapore accessibility | 338 (A 86 / B 36 / C 60 / D 78 / E 78) | train 234 / val 51 / test 53; persona 28/6/6; OD 79/16/19 disjoint; accessibility holdout = walking burden ≥ 15 min (test-only) | 1,468 calls (K=3 ×111, K=5 ×227), prompt `teacher_s8_accessibility_v0.1`, mean-probability aggregation | 0 incomplete / 0 identity-leak hits |

## Prohibited after freeze

- Overwriting the S8 checkpoint, or saving Phase C weights into this directory;
- continuing to train this checkpoint (no optimizer resume, no weight updates);
- modifying config / schema / normalization / accessibility definitions while
  keeping the S8 name;
- re-writing the frozen metrics with new results;
- writing any Phase C output into `releases/` (Phase C output directories are
  guarded by `assert_not_frozen_output()`).

## Phase C fixed settings

```text
Student = frozen S8 (this release)
N* = 10,000
flowCapacityFactor = 0.3
storageCapacityFactor = 0.3
Singapore supply = frozen Phase B.5 version
PT planner = frozen validated version
Scenario order = C0 baseline -> C1 heavy rain -> C2 PT fare increase
                 -> C3 transit delay -> C4 road disruption -> C5 joint scenario
```

All scenarios use the same frozen S8 + the same frozen Singapore supply + the
same N* + the same capacity configuration. The student must not be retrained
based on Phase C results.

## Post-freeze status

```text
S7-W3 = Generic Behavioral Core v1.0 = FROZEN (paper generic baseline)
S8    = Supply-Aware Traveler Agent v1.0 = FROZEN (paper supply-aware extension)
```

Phase C may only: `load frozen S8 → run scenario experiments → save results to
separate output directories`.
