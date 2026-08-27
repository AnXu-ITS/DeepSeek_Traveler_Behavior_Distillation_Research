# FINAL_S9_FREEZE.md

## Identity

```text
Model: S9
Release: Supply-Aware Traveler Agent v2.0
Base: S7-W3 Generic Behavioral Core v1.0
Parameters: 24,562
Schema evolution: Case B (identical to S8)
Status: FROZEN
Supersedes: S8 Supply-Aware Traveler Agent v1.0 (DEPRECATED — walk/bike speed data error)
```

- **Checkpoint**: `releases/s9_supply_aware_v2/checkpoint/model.pt`
  - SHA256 `6af79b44bc699c00cc8e913da15043a43398829fdc8bbb4dad2ddcb8e11d31e6`
  - Source: `outputs/student_s9/checkpoints/best.pt` (byte-identical copy)
  - Seed 42 · best epoch 17 · 24,562 parameters · arch `student_s8_v1` (same Case B architecture as S8)
  - Init checkpoint: `releases/s7_w3_generic_core_v1/checkpoint/model.pt` (FROZEN S7-W3 release)
  - λ_accessibility = 1.0 · λ_mechanism = 1.0 · λ_broken = 1.0
  - Optimizer / scheduler state: **not present** (stores model weights only)
- **Data fix (the ONLY difference vs S8)**: walk 1.34 m/s / bike 4.17 m/s travel times
  (S8 used road free-flow speeds ≈29 km/h effective) — full record in
  `provenance/data_fix.json` and `docs/S8_DEPRECATION.md`.
- **Config**: `config/student_s9.yaml`, `config/accessibility_features.yaml`, `config/training_s9.yaml/json`
- **Schema / Normalization / Accessibility**: as listed in this directory (Case B, same as S8)
- **Provenance**: Singapore supply (same frozen Phase B.5 supply), teacher (1,502 valid calls, 0 incomplete)
- **Metrics**: `metrics/final_metrics.json`, `metrics/comparison_s7_vs_s9.csv`
- **Reports**: `reports/EXPERIMENT_REPORT_S9_TRANSIT_ACCESSIBILITY_V2.md` (repo root `reports/`),
  `reports/S8_SCHEMA_AUDIT.md` (still valid), reproduction gate in `reports/reproduction_gate/`

## Scope

S9 is the supply-aware extension of the generic behavioral core, retrained on the
CORRECTED Singapore accessibility dataset:

- **Key results (test = unseen persona × unseen OD, 95% paired bootstrap)**:
  PT-prob MAE 0.1750 → 0.1346 (Δ −0.0404*); mean P(PT|infeasible) 0.2924 → 0.2130
  (Δ −0.0794*); **FVR rate 0.3333 → 0.0833 (Δ −0.25\*)** — in the corrected world
  the feasibility violation is a real, measurable quantity that S9 substantially reduces;
  pair monotonicity 0.632 → 0.658; sensitivity ΔP_PT 0.085 → 0.153 (teacher 0.448);
  legacy KL −26.0%; seen joint −3.3%; unseen joint −36.5%; mechanism no regression;
  regression gates all pass; identity leakage 0; Stop Rule six items all satisfied.
- **Corrected-world teacher gradient**: P(PT) by class A 0.448 / B 0.429 / C 0.372 /
  D 0.189 / E 0.000 (S8's world compressed this to A 0.236 / D 0.123).
- Phase C reruns with frozen S9.

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
- mode shares calibrated to observed Singapore modal split (synthetic personas/trips)

## Freeze record (steps 1–15)

| Step | Item | Status |
|---|---|---|
| 1 | Unique final checkpoint located & copied (best_epoch 17, 24,562 params) | ✅ |
| 2 | Config frozen (student_s9.yaml + accessibility_features.yaml + training_s9) | ✅ |
| 3 | Case B schema frozen (identical to S8; diff vs S7 recorded) | ✅ |
| 4 | Accessibility definitions frozen incl. the S9 speed correction | ✅ |
| 5 | Singapore supply provenance frozen (same as S8; copied verbatim) | ✅ |
| 6 | S9 dataset manifest frozen (336 states, triple holdout, leakage=0) | ✅ |
| 7 | Teacher provenance frozen (1,502 valid calls, 0 incomplete) | ✅ |
| 8 | Pre-freeze reproduction gate — see below | ✅ PASS |
| 9 | Final metrics frozen (final_metrics.json, comparison_s7_vs_s9.csv) | ✅ |
| 10 | Reports + evidence index | ✅ |
| 11 | Code manifest (S9 data-fix commits, environment) | ✅ |
| 12 | Git commit + annotated tag `s9-supply-aware-v2.0` | ✅ |
| 13 | SHA256 checksums | ✅ |
| 14 | This file | ✅ |
| 15 | Write guard: read-only release + release_guard.py (S9 dir added) | ✅ |

## Reproduction gate (Step 8 — frozen checkpoint re-evaluation, no retraining)

Re-run of the three test-only pipelines against `checkpoint/model.pt`; results at
`reports/reproduction_gate/` (compared by `scripts/freeze_s9_release.py verify-gate`).
All gate metrics reproduced the historical S9 values exactly (Δ = 0.0000).

Singapore smoke (N=100, frozen Phase C settings): schema round-trip asserted;
MATSim exit = 0; pt accessibility pipeline 98 feasible / 2 infeasible;
S9 decision distribution at N=100 — **pt 31% / bike 30% / car 28% / walk 11%**
(S8's world: pt 4% / bike 33% / car 20% / walk 43% — the correction shifts
demand toward pt/car, matching the expected direction for Singapore-like travel).

## Code version

- S9 code = S8 commit `65e505f` + the data-fix edits
  (`gtfs_accessibility.py` mode speeds, `adapter.py` leg-routing weights,
  `train_student_s8.py` split-count assertion generalized) + freeze commit.
- Tag: `s9-supply-aware-v2.0` (annotated).
- Environment: Python 3.12.13 · torch 2.13.0+cpu · OpenJDK 25.0.4 · MATSim 2026.0.
- Test suite: 158 passed / 0 error.

## Data identity (summary; full record in `data_manifest/`)

| Dataset | States | Splits | Teacher | Incomplete / leakage |
|---|---|---|---|---|
| S9 Singapore accessibility (corrected) | 336 (A 79 / B 16 / C 77 / D 84 / E 80) | train 234 / val 51 / test 51; persona 28/6/6; OD 79/14/20 disjoint; accessibility holdout = walk burden ≥ 15 min (test-only) | 1,502 valid calls (K=3 ×89, K=5 ×247) | 0 incomplete / 0 identity-leak hits |

S8 legacy dataset archived at `data/singapore_accessibility_s8_legacy/`.

## Prohibited after freeze

- Overwriting the S9 checkpoint; continuing to train it (no optimizer resume, no weight updates);
- modifying config / schema / normalization / accessibility definitions while keeping the S9 name;
- re-writing the frozen metrics; writing Phase C output into `releases/`.

## Phase C fixed settings (unchanged from the S8 freeze)

```text
Student = frozen S9 (this release)
N* = 10,000
flowCapacityFactor = 0.3
storageCapacityFactor = 0.3
Singapore supply = frozen Phase B.5 version
PT planner = frozen validated version
Scenario order = C0 baseline -> C1 heavy rain -> C2 PT fare increase
                 -> C3 transit delay -> C4 road disruption -> C5 joint scenario
```

Known execution-layer limitation (documented): MATSim moves walk/bike legs at link
freespeed (network modes), so MATSim-side walk/bike trip times are optimistic; mode
DECISIONS are made before MATSim from the corrected alternatives and are unaffected.

## Post-freeze status

```text
S7-W3 = Generic Behavioral Core v1.0 = FROZEN (paper generic baseline)
S8    = Supply-Aware Traveler Agent v1.0 = DEPRECATED (data error, kept intact)
S9    = Supply-Aware Traveler Agent v2.0 = FROZEN (paper supply-aware extension)
```
