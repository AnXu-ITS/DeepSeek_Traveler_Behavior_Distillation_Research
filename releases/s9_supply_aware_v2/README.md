# S9 — Supply-Aware Traveler Agent v2.0 (FROZEN)

Frozen release of the S9 student checkpoint, retrained on the corrected
Singapore accessibility dataset (walk 1.34 m/s / bike 4.17 m/s). S8 used road
free-flow speeds; see the [deprecation record](../../docs/S8_DEPRECATION.md).
The checkpoint, configuration, normalization and numerical artifacts remain
frozen: never overwrite them or save training/Phase C outputs here. See
`FINAL_S9_FREEZE.md` for the historical freeze record, claims boundaries and guard.

Markdown documentation on the current branch has editorial updates. The
full-package checksums apply to the [original tagged package](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/tree/s9-supply-aware-v2.0/releases/s9_supply_aware_v2).
Model and numerical artifact bytes are unchanged; see the
[documentation boundary](../../docs/DOCUMENTATION_HISTORY.md#maintained-documentation-and-source-boundaries).

## Contents

```text
checkpoint/       model.pt + training history + val metrics (best_epoch=17)
config/           student_s9.yaml, accessibility_features.yaml, training_s9.yaml/json
schema/           input/output schema, feature_order_s9.txt, schema_diff_vs_s7.json (Case B)
normalization/    z-score stats (S7 fields verbatim + 6 accessibility fields fit on S9 train)
accessibility/    feature definitions (incl. the S9 speed correction) + feature schema
provenance/       Singapore supply + teacher provenance + data_fix.json
data_manifest/    S9 dataset manifest (336 states, splits, leakage=0) + split manifest
metrics/          final_metrics.json + comparison_s7_vs_s9.csv
reports/          S9 experiment report + S8 schema audit + reproduction gate
code_manifest/    git commits, worktree status, environment
checksums/        SHA256SUMS.txt over every frozen file
```

## 1. Load the checkpoint

```python
import torch, json
ck = torch.load("releases/s9_supply_aware_v2/checkpoint/model.pt",
                map_location="cpu", weights_only=False)

from traveler_distillation.accessibility.accessibility_features import (
    S8FeatureExtractor, TravelerStudentS8)
extractor = S8FeatureExtractor().from_state_dict(ck["extractor_state"])
model = TravelerStudentS8(ck["config"], ck["feature_spec"])
model.load_state_dict(ck["model_state"])
model.eval()
assert sum(p.numel() for p in model.parameters()) == 24562
```

Phase C must use THIS frozen checkpoint only — no optimizer resume, no weight
updates, no normalization or schema changes.

## 2. Construct an input

Inputs are `UniversalTravelerState` objects. The pt alternative must carry the
6 accessibility attributes computed from real supply with the validated PT
planner; **walk/bike travel times must be computed at 1.34 m/s / 4.17 m/s**
(`SupplyIndex` in `gtfs_accessibility.py` does this). Encode with the frozen S9
extractor and collate exactly as in the S8 release README.

## 3. Reproduction gate

```bash
python scripts/eval_s8_accessibility.py \
    --s8-checkpoint releases/s9_supply_aware_v2/checkpoint/model.pt --output <gate_dir>/accessibility_eval
python scripts/eval_s8_unseen_od.py \
    --accessibility-eval <gate_dir>/accessibility_eval/eval_metrics.json --output <gate_dir>/unseen_od
python scripts/eval_s8_regression.py \
    --s8-checkpoint releases/s9_supply_aware_v2/checkpoint/model.pt --output <gate_dir>/regression_eval
python scripts/singapore/run_s8_smoke.py --checkpoint releases/s9_supply_aware_v2/checkpoint/model.pt
python scripts/freeze_s9_release.py verify-gate
```

Expected frozen values are in `metrics/final_metrics.json`; the freeze-time gate
reproduced all 12 gate metrics exactly (Δ = 0.0000).

## 4. Restore / rebuild

```bash
python scripts/freeze_s9_release.py build
python scripts/freeze_s9_release.py checksums
```

## Guard

`src/traveler_distillation/student/release_guard.py::assert_not_frozen_output()`
hard-rejects any output path resolving inside this release (and the S7/S8
releases). All Phase C scripts must call it on their output directories.

## Phase C fixed settings (frozen)

```text
Student           = frozen S9 (this release)
N*                = 10,000
flowCapacityFactor   = 0.3
storageCapacityFactor = 0.3
Singapore supply  = frozen Phase B.5 version
PT planner        = frozen validated version
Scenario order    = C0 baseline -> C1 heavy rain -> C2 PT fare increase
                    -> C3 transit delay -> C4 road disruption -> C5 joint
```

Known limitation: MATSim executes walk/bike legs as network modes at link
freespeed (optimistic trip times); mode decisions are made before MATSim from
the corrected alternatives and are unaffected.
