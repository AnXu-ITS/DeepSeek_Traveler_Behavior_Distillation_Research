# S8 — Supply-Aware Traveler Agent v1.0 (FROZEN)

Frozen release of the S8 student checkpoint (real-supply transit accessibility
adaptation, Case B schema evolution, stop rule met). This directory is
**read-only by policy**: never modify, overwrite, or save training/Phase C
outputs into it. See `FINAL_S8_FREEZE.md` for the freeze record, claims
boundaries, and the Phase C guard.

## Contents

```text
checkpoint/       model.pt + training history + val metrics (best_epoch=26)
config/           student_s8.yaml, accessibility_features.yaml, training_s8.yaml/json
schema/           input/output schema, feature_order_s8.txt, schema_diff_vs_s7.json (Case B)
normalization/    z-score stats (S7 fields verbatim + 6 new S8 fields fit on S8 train)
accessibility/    feature definitions + feature schema (city-independent)
provenance/       Singapore supply (OSM/GTFS manifests) + teacher provenance
data_manifest/    S8 dataset manifest (338 states, splits, leakage=0) + split manifest
metrics/          final_metrics.json + comparison_s7_vs_s8.csv
reports/          S8 schema audit + S8 experiment report + evidence index + reproduction gate
code_manifest/    git commits, worktree status, environment, uncommitted inventory
checksums/        SHA256SUMS.txt over every frozen file
```

## 1. Load the checkpoint

```python
import torch, json
ck = torch.load("releases/s8_supply_aware_v1/checkpoint/model.pt",
                map_location="cpu", weights_only=False)
# keys: model_state, feature_spec, extractor_state, config, arch_version,
#       init_checkpoint, lambda_accessibility, lambda_mechanism, lambda_broken

from traveler_distillation.accessibility.accessibility_features import (
    S8FeatureExtractor, TravelerStudentS8)
extractor = S8FeatureExtractor().from_state_dict(ck["extractor_state"])
model = TravelerStudentS8(ck["config"], ck["feature_spec"])
model.load_state_dict(ck["model_state"])
model.eval()
# 24,562 parameters — verify:
assert sum(p.numel() for p in model.parameters()) == 24562
```

Phase C must use THIS frozen checkpoint only — no optimizer resume, no weight
updates, no normalization or schema changes.

## 2. Load the schema / normalization

They live both as JSON files and inside the checkpoint:

```python
schema = json.load(open("releases/s8_supply_aware_v1/schema/input_schema_s8.json", encoding="utf-8"))
norm  = json.load(open("releases/s8_supply_aware_v1/normalization/normalization.json", encoding="utf-8"))
```

`feature_order_s8.txt` documents the exact tensor column order
(12 alternative numeric fields); `schema_diff_vs_s7.json` records the Case B
evolution (6 added fields, alt_encoder 14→20, zero-init migration, S7 release
unmodified).

## 3. Construct an input

Inputs are `UniversalTravelerState` objects. The pt alternative must carry the
6 S8 accessibility attributes (`pt_feasible`, `egress_time_min`,
`wait_time_min`, `in_vehicle_time_min`, `transfer_time_min`, `coverage_ratio`)
computed from real supply with the validated PT planner — see
`accessibility/feature_definitions.md` and
`src/traveler_distillation/accessibility/gtfs_accessibility.py::plan_accessibility`.
Encode with the frozen S8 extractor and collate:

```python
from traveler_distillation.student import collate_batch
import torch
feats = extractor.encode(state)          # state: UniversalTravelerState
batch = collate_batch([{
    "global_cat":   torch.tensor(feats["global_cat"], dtype=torch.long),
    "global_num":   torch.tensor(feats["global_num"], dtype=torch.float32),
    "alt_mode_idx": torch.tensor(feats["alt_mode_idx"], dtype=torch.long),
    "alt_num":      torch.tensor(feats["alt_num"], dtype=torch.float32),
    "alt_mask":     torch.tensor(feats["alt_available"], dtype=torch.float32),
    "target_probs": torch.zeros(len(feats["alt_available"])),
    "target_mode_idx": torch.zeros((), dtype=torch.long),
    "target_departure": torch.zeros(()),
}])
```

## 4. Run inference

```python
with torch.no_grad():
    out = model(batch)
# out["mode_probabilities"]      : masked softmax over available modes
# out["departure_time_shift_min"]: 60 * tanh(head), i.e. minutes in (-60, +60)
# out["utilities"]               : raw scorer logits per alternative
```

## 5. Reproduction gate (same commands used for the freeze)

```bash
python scripts/eval_s8_accessibility.py \
    --s8-checkpoint releases/s8_supply_aware_v1/checkpoint/model.pt \
    --output <gate_dir>/accessibility_eval
python scripts/eval_s8_unseen_od.py \
    --accessibility-eval <gate_dir>/accessibility_eval/eval_metrics.json \
    --output <gate_dir>/unseen_od
python scripts/eval_s8_regression.py \
    --s8-checkpoint releases/s8_supply_aware_v1/checkpoint/model.pt \
    --output <gate_dir>/regression_eval
python scripts/singapore/run_s8_smoke.py          # S8 inference -> adapter -> MATSim
python scripts/freeze_s8_release.py verify-gate   # compares gate outputs vs historical values
```

Expected frozen values are in `metrics/final_metrics.json`; the freeze-time gate
must reproduce them (see `reports/reproduction_gate/`).

## 6. Restore / rebuild this release

```bash
python scripts/freeze_s8_release.py build      # rebuild directory contents
python scripts/freeze_s8_release.py checksums  # regenerate checksums/SHA256SUMS.txt
```

Re-running `build` overwrites release files — only do this from a clean checkout
at the freeze commits and only when reconstructing the release, never to "update"
the frozen version. Any checksum change after freeze means the version has been
modified.

## Guard

`src/traveler_distillation/student/release_guard.py::assert_not_frozen_output()`
hard-rejects any output path resolving inside `releases/s8_supply_aware_v1/`
(and `releases/s7_w3_generic_core_v1/`). All Phase C scripts must call it on
their output directories before writing; Phase C output must never land in this
release.

## Phase C fixed settings (frozen)

```text
Student           = frozen S8 (this release)
N*                = 10,000
flowCapacityFactor   = 0.3
storageCapacityFactor = 0.3
Singapore supply  = frozen Phase B.5 version
PT planner        = frozen validated version
Scenario order    = C0 baseline -> C1 heavy rain -> C2 PT fare increase
                    -> C3 transit delay -> C4 road disruption -> C5 joint
```

The student must not be retrained based on Phase C results.
