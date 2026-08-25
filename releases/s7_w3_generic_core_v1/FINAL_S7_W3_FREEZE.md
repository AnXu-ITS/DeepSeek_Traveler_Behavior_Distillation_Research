# FINAL_S7_W3_FREEZE.md

## Identity

```text
Model:   S7-W3
Release: Generic Behavioral Core v1.0
Status:  FROZEN
```

- **Checkpoint**: `releases/s7_w3_generic_core_v1/checkpoint/model.pt`
  - SHA256 `8263faec7d9a1e76d7aaca67b44b36616f36e379c2fc1bc6e156167e5c53b2b6`
  - Source: `outputs/student_s7_w3/checkpoints/best.pt` (byte-identical copy, saved 2026-08-25T05:32:25Z)
  - Seed 42 · best epoch 1 · 24,370 parameters · init checkpoint `outputs/student_s5_joint_m2/checkpoints/best.pt` (C1)
  - Optimizer / scheduler state: **not present** (S7 `best.pt` stores model weights only)
- **Config**: `config/student_s7_w3.yaml` (λ_mechanism = 1.0, λ_broken = 1.0, LR 1.25e-4, replay 2:1:1)
- **Schema**: `schema/input_schema.json`, `schema/output_schema.json`, `schema/feature_order.txt`
- **Normalization**: `normalization/normalization.json`, `mode_mapping.json`, `category_mapping.json`
- **Metrics**: `metrics/final_metrics.json`, `metrics/seed_robustness.csv`
- **Data**: `data_manifest/data_manifest.json` (S3 legacy / S5 joint / S6 causal / S7 mechanism)
- **Code**: `code_manifest/` (git commits, worktree status, environment, uncommitted inventory)

## Scope

S7-W3 completes the model-training line and is frozen as the paper baseline:

- single-axis distillation (S1–S4, six perturbation axes, 40 personas);
- multi-axis joint distillation (S5, seen/unseen combinations);
- causal-mechanism audit (S6, Grade B — partial mechanism preservation);
- mechanism-aware targeted fine-tuning (S7, Grade B, seed-stable);
- seed robustness (4 training seeds: 42 / 7 / 123 / 2024).

The 4-seed checkpoints and seed evaluation artifacts are archived under
`checkpoint/seeds/` and `metrics/seeds/`.

## Claims allowed

- *axis-dependent partial mechanism preservation under targeted mechanism-aware supervision*
- *generic behavioral core*
- *simulation-executable traveler agent*

## Claims prohibited

- *full causal reasoning*
- *universal human behavior model*
- *Singapore-trained behavior model*

## Freeze record (steps 1–15)

| Step | Item | Status |
|---|---|---|
| 1 | Unique final checkpoint located & copied (`checkpoint/model.pt` + seed checkpoints) | ✅ |
| 2 | Config frozen (`config/student_s7_w3.yaml` + checkpoint-embedded config snapshot) | ✅ |
| 3 | Input/output schema + `feature_order.txt` (matches actual tensor order) | ✅ |
| 4 | Normalization frozen from checkpoint `extractor_state` (fit on S7 train split only) | ✅ |
| 5 | Data manifest with per-dataset counts, K distribution, checksums | ✅ |
| 6 | Code manifest (git commits, status, environment, uncommitted inventory) | ✅ |
| 7 | Pre-freeze reproducibility gate — see "Smoke reproduction" below | ✅ PASS |
| 8 | Core metrics frozen (`metrics/final_metrics.json`, verbatim from historical eval artifacts) | ✅ |
| 9 | 4-seed results archived (`metrics/seeds/`, `metrics/seed_robustness.csv`, `reports/SEED_ROBUSTNESS_S7_W3.md`) | ✅ |
| 10 | Key reports copied (`reports/`: S5, S6, S7 + seed robustness) | ✅ |
| 11 | SHA256 checksums (`checksums/SHA256SUMS.txt`) | ✅ |
| 12 | This file | ✅ |
| 13 | `README.md` (load / infer / evaluate / restore) | ✅ |
| 14 | Git commit + annotated tag `s7-w3-generic-core-v1.0` | ✅ |
| 15 | Read-only protection: Windows read-only attributes + `src/traveler_distillation/student/release_guard.py` hard assertion (tested: `tests/test_release_guard.py`) | ✅ |

## Smoke reproduction (Step 7 gate — run against the frozen `checkpoint/model.pt`, no retraining)

Same test-only pipelines as the original S7 evaluation
(`scripts/eval_s7_regression.py`, `scripts/eval_s7_causal_repair.py`), results at
`smoke/regression_smoke.json` and `smoke/causal_smoke.json`.

| Gate metric | Historical (S7 W3) | Frozen release | Δ |
|---|---|---|---|
| legacy mode accuracy | 0.8451 | 0.8451 | 0.0000 |
| legacy KL | 0.0695 | 0.0695 | 0.0000 |
| legacy probability L1 | 0.2420 | 0.2420 | 0.0000 |
| seen joint KL | 0.0421 | 0.0421 | 0.0000 |
| unseen joint KL | 0.0845 | 0.0845 | 0.0000 |
| parking G_med | 0.2580 | 0.2580 | 0.0000 |
| congestion Gap_shortcut | 0.3455 | 0.3455 | 0.0000 |

**Verdict: 7/7 gates reproduce the historical values exactly (Δ = 0.0000) —
smoke reproduction PASS.**
(`scripts/freeze_s7_w3_release.py verify-smoke` re-runs this comparison.)

## Code version

- S7-W3 training/evaluation code commit: `222e17e` ("S6/S7: causal mechanism audit + mechanism-aware fine-tuning").
- Freeze performed at `HEAD = 02f64b3a87921c7651a27b2f1755fee7069a00c4`.
- All S7-relevant paths (`src/traveler_distillation/student/`, `schemas/`, S7 scripts,
  `configs/student_s7_w3.yaml`) are identical between the two commits; the only
  differences are S5-era changes that were already present in the worktree when S7
  ran and were committed in `02f64b3` (`joint_axes` field in `schemas/dataset.py`,
  joint generator). The exact smoke reproduction above empirically confirms the
  evaluation pipeline matches the S7-time code.
- Freeze commit: `freeze: S7-W3 generic behavioral core v1.0` (contains this release
  directory, `scripts/freeze_s7_w3_release.py`, the release guard, and the two
  instruction files `S7_W3_BACKUP_FREEZE_INSTRUCTIONS.md` / `S8_TRANSIT_ACCESSIBILITY_TRAINING_INSTRUCTIONS.md`).
- Tag: `s7-w3-generic-core-v1.0` (annotated).

## Data identity (summary; full record in `data_manifest/data_manifest.json`)

| Dataset | States | Personas | Teacher K | Incomplete |
|---|---|---|---|---|
| S3 legacy single-axis (`data/student_v0_3_s3/`) | 1513 (4539 repeats) | 40 | 3 | 0 |
| S5 joint (`data/student_s5_joint/`) | 640 | 40 | 3:320 / 5:280 / 7:40 | 0 |
| S6 causal audit (`data/causal_audit/`) | 960 | 40 | 3:473 / 5:487 | 0 |
| S7 mechanism quadruplets (`data/student_s7_mechanism/`) | 72 (train 26 / val 2 / test 8 per axis) | 18 (car-available) | A/B=3, C/D=5 | 0 |

## Prohibited after freeze

- Overwriting the S7-W3 checkpoint, or saving S8 weights into this directory;
- modifying config/schema/normalization while keeping the S7-W3 name;
- re-writing original frozen metrics with new test results;
- deleting the original S5/S6/S7 reports;
- continuing to train this checkpoint without a new version name (`S8-…`, `S9-…`).

Any later fine-tuning must start a new version and save to a separate output/release
directory. `assert_not_frozen_output()` in
`src/traveler_distillation/student/release_guard.py` hard-rejects output paths inside
this release; every S8+ script must call it.

## Post-freeze status

```text
S7-W3 = Generic Behavioral Core v1.0 = FROZEN
```

S8 may only: `load S7-W3 → initialize new S8 experiment → save to separate output/release`.
