"""Tests for Student-S8 features (Case B weight reuse + init invariance)."""
from __future__ import annotations

from pathlib import Path

import torch

from traveler_distillation.accessibility.accessibility_features import (
    S8FeatureExtractor,
    TravelerStudentS8,
)
from traveler_distillation.config import load_yaml
from traveler_distillation.generators import BaselineStateGenerator, PersonaGenerator, TripGenerator
from traveler_distillation.student import FeatureExtractor, TravelerStudent, collate_batch

_ROOT = Path(__file__).resolve().parents[1]
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


def _setup():
    cfg = load_yaml(str(_ROOT / "configs" / "generation_v0_1.yaml"))
    personas = PersonaGenerator(seed=42, config=cfg).generate(3)
    trips = TripGenerator(seed=42, config=cfg).generate(3)
    states = [BaselineStateGenerator(cfg).generate(p, t) for p, t in zip(personas, trips)]
    ck = torch.load(RELEASE_CKPT, map_location="cpu", weights_only=False)
    ext_s8 = S8FeatureExtractor.from_s7_and_train(ck["extractor_state"], states)
    model_s8 = TravelerStudentS8.from_s7_weights(RELEASE_CKPT, ext_s8.spec)
    ext_s7 = FeatureExtractor().from_state_dict(ck["extractor_state"])
    model_s7 = TravelerStudent(ck["config"], ck["feature_spec"])
    model_s7.load_state_dict(ck["model_state"])
    model_s7.eval()
    return states, ck, ext_s8, model_s8, ext_s7, model_s7


def test_weight_reuse_and_parameter_count():
    _states, ck, _e, model_s8, _e7, _m7 = _setup()
    w_old = ck["model_state"]["alt_encoder.0.weight"]
    w_new = model_s8.alt_encoder[0].weight.detach()
    assert torch.equal(w_new[:, :14], w_old)
    assert torch.count_nonzero(w_new[:, 14:]) == 0
    assert model_s8.count_parameters() == 24562


def test_frozen_s7_stats_kept_verbatim():
    _states, ck, ext_s8, _m, _e7, _m7 = _setup()
    for key in ("household_size", "travel_time_min", "monetary_cost"):
        assert ext_s8.num_mean[key] == ck["extractor_state"]["num_mean"][key]
        assert ext_s8.num_std[key] == ck["extractor_state"]["num_std"][key]
    assert ext_s8.spec["n_alt_num"] == 12


def test_init_invariance_s8_matches_s7w3():
    states, _ck, ext_s8, model_s8, ext_s7, model_s7 = _setup()
    probe = states[0]
    with torch.no_grad():
        out_s7 = model_s7(_encode(probe, ext_s7))
        out_s8 = model_s8(_encode(probe, ext_s8))
    assert torch.allclose(out_s7["mode_probabilities"], out_s8["mode_probabilities"], atol=1e-6)
    assert torch.allclose(out_s7["departure_time_shift_min"], out_s8["departure_time_shift_min"], atol=1e-6)


def test_legacy_states_encode_with_zero_new_features():
    states, _ck, ext_s8, _m, _e7, _m7 = _setup()
    feats = ext_s8.encode(states[0])
    assert len(feats["alt_num"][0]) == 12
    # legacy alternatives carry no S8 fields -> z-scored new features are the
    # zero-value z-scores (constant across alternatives)
    alt_rows = feats["alt_num"]
    for i in range(6, 12):
        assert all(abs(row[i] - alt_rows[0][i]) < 1e-9 for row in alt_rows)
