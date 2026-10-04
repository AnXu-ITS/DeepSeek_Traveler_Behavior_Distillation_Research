# S7-W3 — Generic Behavioral Core v1.0 (FROZEN)

Frozen release of the S7-W3 student checkpoint (mechanism-aware fine-tuning,
Grade B, seed-stable). The checkpoint, configuration, normalization and numerical
artifacts are frozen: never overwrite them or save new training outputs here.
See `FINAL_S7_W3_FREEZE.md` for the historical freeze record and claims boundaries.

Markdown documentation on the current branch has editorial updates. The
full-package checksums apply to the [original tagged package](https://github.com/AnXu-ITS/DeepSeek_Traveler_Behavior_Distillation_Research/tree/s7-w3-generic-core-v1.0/releases/s7_w3_generic_core_v1).
Model and numerical artifact bytes are unchanged; see the
[documentation boundary](../../docs/DOCUMENTATION_HISTORY.md#maintained-documentation-and-source-boundaries).

## Contents

```text
checkpoint/       model.pt + training/val artifacts + 4 seed checkpoints
config/           student_s7_w3.yaml + checkpoint-embedded config snapshot
schema/           input_schema.json, output_schema.json, feature_order.txt
normalization/    z-score stats + mode/category vocabularies (from the checkpoint)
metrics/          final_metrics.json, seed_robustness.csv, seed eval artifacts
reports/          S5 / S6 / S7 experiment reports + seed robustness report
data_manifest/    identities + SHA256 of the four training datasets
code_manifest/    git commits, worktree status, environment, uncommitted inventory
smoke/            reproduction-gate outputs (frozen checkpoint re-evaluation)
checksums/        SHA256SUMS.txt over every frozen file
```

## 1. Load the checkpoint

```python
import torch, json
ck = torch.load("releases/s7_w3_generic_core_v1/checkpoint/model.pt",
                map_location="cpu", weights_only=False)
# keys: model_state, feature_spec, extractor_state, config,
#       init_checkpoint, lambda_mechanism, lambda_broken

from traveler_distillation.student import FeatureExtractor, TravelerStudent
extractor = FeatureExtractor().from_state_dict(ck["extractor_state"])
model = TravelerStudent(ck["config"], ck["feature_spec"])
model.load_state_dict(ck["model_state"])
model.eval()
# 24,370 parameters — verify:
assert sum(p.numel() for p in model.parameters()) == 24370
```

## 2. Load the schema / normalization

They live both as JSON files and inside the checkpoint:

```python
schema = json.load(open("releases/s7_w3_generic_core_v1/schema/input_schema.json", encoding="utf-8"))
norm  = json.load(open("releases/s7_w3_generic_core_v1/normalization/normalization.json", encoding="utf-8"))
```

`feature_order.txt` documents the exact tensor column order consumed by
`TravelerStudent.forward`; `extractor_state` in the checkpoint is the authoritative
encoding (vocabularies + mean/std fit on the S7 train split only).

## 3. Construct an input

Inputs are `UniversalTravelerState` objects (persona, trip, context with weather /
fare / road_congestion / transit_delay / parking_cost / road_disruption etc.,
and mode-level alternatives with travel_time, monetary_cost, access_time,
transfers, reliability_delay, weather_exposure + availability flags). Encode with
the frozen extractor and collate:

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

Unavailable alternatives are masked out of the softmax and carry zero
probability mass (see `schema/output_schema.json`).

## 5. Run the minimal evaluation (reproduction)

The reproduction gate itself (same commands used for the freeze):

```bash
python scripts/eval_s7_regression.py \
    --variant W3=releases/s7_w3_generic_core_v1/checkpoint/model.pt \
    --output /tmp/smoke_regression.json --device cpu
python scripts/eval_s7_causal_repair.py \
    --variant W3=releases/s7_w3_generic_core_v1/checkpoint/model.pt \
    --output /tmp/smoke_causal.json --device cpu
python scripts/freeze_s7_w3_release.py verify-smoke   # expects outputs in smoke/
```

Expected frozen values are in `metrics/final_metrics.json`; the gate must match
them (historical S7 numbers) within tolerance — at freeze time all 7 gate metrics
reproduced exactly (Δ = 0.0000).

## 6. Restore / rebuild this release

```bash
python scripts/freeze_s7_w3_release.py build      # rebuild directory contents
python scripts/freeze_s7_w3_release.py checksums  # regenerate checksums/SHA256SUMS.txt
```

Re-running `build` overwrites release files — only do this from a clean checkout
at the freeze commits and only when reconstructing the release, never to "update"
the frozen version. Any checksum change after freeze means the version has been
modified.

## Guard

`src/traveler_distillation/student/release_guard.py::assert_not_frozen_output()`
hard-rejects any output path resolving inside `releases/s7_w3_generic_core_v1/`.
All S8+ training/eval scripts must call it on their output directories before
writing.
