#!/usr/bin/env python
"""S8 schema audit — programmatic assertions backing reports/S8_SCHEMA_AUDIT.md.

Verifies the Case B decision mechanically:
  1. the frozen S7-W3 checkpoint has n_alt_num=6 and alt_encoder input 14;
  2. Student-S8 (n_alt_num=12) has alt_encoder input 20 and 24,562 params;
  3. S7 weight reuse: first 14 columns copied verbatim, 6 new columns zero;
  4. init invariance: S8-at-init reproduces S7-W3 outputs exactly on a legacy
     state (the new-feature columns contribute nothing until training).

Usage:
    python scripts/audit_s8_schema.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch

from traveler_distillation.accessibility import S8FeatureExtractor, TravelerStudentS8
from traveler_distillation.accessibility.accessibility_features import S8_ALT_NUM, S8_NEW_ALT_NUM
from traveler_distillation.config import load_yaml
from traveler_distillation.generators import (
    AlternativeGenerator,
    BaselineStateGenerator,
    PersonaGenerator,
    TripGenerator,
)
from traveler_distillation.student import FeatureExtractor, TravelerStudent, collate_batch

RELEASE_CKPT = _ROOT / "releases" / "s7_w3_generic_core_v1" / "checkpoint" / "model.pt"


def _encode(state, extractor):
    feats = extractor.encode(state)
    return collate_batch([{
        "global_cat": torch.tensor(feats["global_cat"], dtype=torch.long),
        "global_num": torch.tensor(feats["global_num"], dtype=torch.float32),
        "alt_mode_idx": torch.tensor(feats["alt_mode_idx"], dtype=torch.long),
        "alt_num": torch.tensor(feats["alt_num"], dtype=torch.float32),
        "alt_mask": torch.tensor(feats["alt_available"], dtype=torch.float32),
        "target_probs": torch.zeros(len(feats["alt_available"])),
        "target_mode_idx": torch.zeros((), dtype=torch.long),
        "target_departure": torch.zeros(()),
    }])


def main() -> int:
    ck = torch.load(RELEASE_CKPT, map_location="cpu", weights_only=False)
    s7_spec = ck["feature_spec"]
    assert s7_spec["n_alt_num"] == 6, s7_spec["n_alt_num"]
    assert ck["model_state"]["alt_encoder.0.weight"].shape == (32, 14)
    n_params_s7 = sum(v.numel() for v in ck["model_state"].values())
    assert n_params_s7 == 24370, n_params_s7
    print("[1] S7-W3: n_alt_num=6, alt_encoder.0=(32,14), 24,370 params — OK")

    # S8 feature spec + extractor (fit new fields on a tiny synthetic train set)
    cfg = load_yaml(str(_ROOT / "configs" / "generation_v0_1.yaml"))
    personas = PersonaGenerator(seed=42, config=cfg).generate(4)
    trips = TripGenerator(seed=42, config=cfg).generate(4)
    alt_gen = AlternativeGenerator(cfg)
    train_states = [
        BaselineStateGenerator(cfg).generate(p, t) for p, t in zip(personas, trips)
    ]
    for s in train_states:
        s.alternatives = alt_gen.generate(s.persona, s.trip, s.context)
    s8_ext = S8FeatureExtractor.from_s7_and_train(ck["extractor_state"], train_states)
    s8_spec = s8_ext.spec
    assert s8_spec["n_alt_num"] == 12
    assert len(S8_ALT_NUM) == 12 and len(S8_NEW_ALT_NUM) == 6

    model_s8 = TravelerStudentS8.from_s7_weights(RELEASE_CKPT, s8_spec)
    assert model_s8.alt_encoder[0].weight.shape == (32, 20)
    n_params_s8 = model_s8.count_parameters()
    assert n_params_s8 == 24562, n_params_s8
    print(f"[2] Student-S8: n_alt_num=12, alt_encoder.0=(32,20), {n_params_s8} params — OK")

    # weight reuse: first 14 columns verbatim, new 6 columns zero
    w_old = ck["model_state"]["alt_encoder.0.weight"]
    w_new = model_s8.alt_encoder[0].weight.detach()
    assert torch.equal(w_new[:, :14], w_old)
    assert torch.count_nonzero(w_new[:, 14:]) == 0
    print("[3] Weight reuse: cols 0:14 verbatim, cols 14:20 zero-initialized — OK")

    # init invariance: S8-at-init == S7-W3 exactly on a legacy state
    ext_s7 = FeatureExtractor().from_state_dict(ck["extractor_state"])
    model_s7 = TravelerStudent(ck["config"], s7_spec)
    model_s7.load_state_dict(ck["model_state"])
    model_s7.eval()
    probe = train_states[0]
    with torch.no_grad():
        out_s7 = model_s7(_encode(probe, ext_s7))
        out_s8 = model_s8(_encode(probe, s8_ext))
    assert torch.allclose(out_s7["mode_probabilities"], out_s8["mode_probabilities"], atol=1e-6)
    assert torch.allclose(out_s7["departure_time_shift_min"], out_s8["departure_time_shift_min"], atol=1e-6)
    print("[4] Init invariance: S8-at-init output == S7-W3 output (atol 1e-6) — OK")

    print("\nS8_SCHEMA_AUDIT: all programmatic assertions PASS (Case B confirmed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
