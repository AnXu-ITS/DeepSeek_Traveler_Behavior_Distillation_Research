"""Student forward pass tests."""
from __future__ import annotations

import torch

from traveler_distillation.student import FeatureExtractor, TravelerStudent, collate_batch


def _item(state, extractor):
    f = extractor.encode(state)
    return {
        "global_cat": torch.tensor(f["global_cat"], dtype=torch.long),
        "global_num": torch.tensor(f["global_num"], dtype=torch.float32),
        "alt_mode_idx": torch.tensor(f["alt_mode_idx"], dtype=torch.long),
        "alt_num": torch.tensor(f["alt_num"], dtype=torch.float32),
        "alt_mask": torch.tensor(f["alt_available"], dtype=torch.float32),
    }


def test_forward_shapes_and_constraints(baseline_state):
    extractor = FeatureExtractor().fit([baseline_state])
    model = TravelerStudent(
        {"global_hidden_dim": 16, "alternative_hidden_dim": 8, "scorer_hidden_dim": 16,
         "cat_embedding_dim": 4, "mode_embedding_dim": 4, "dropout": 0.0},
        extractor.spec,
    )
    model.eval()
    batch = collate_batch([_item(baseline_state, extractor), _item(baseline_state, extractor)])
    out = model(batch)

    n_alt = batch["alt_mode_idx"].shape[1]
    assert out["mode_probabilities"].shape == (2, n_alt)
    assert out["departure_time_shift_min"].shape == (2,)
    assert out["utilities"].shape == (2, n_alt)
    assert torch.allclose(out["mode_probabilities"].sum(dim=1), torch.ones(2), atol=1e-5)


def test_forward_departure_bounded(baseline_state):
    extractor = FeatureExtractor().fit([baseline_state])
    model = TravelerStudent(
        {"global_hidden_dim": 16, "alternative_hidden_dim": 8, "scorer_hidden_dim": 16,
         "cat_embedding_dim": 4, "mode_embedding_dim": 4, "dropout": 0.0},
        extractor.spec,
    )
    batch = collate_batch([_item(baseline_state, extractor)])
    out = model(batch)
    assert (out["departure_time_shift_min"].abs() <= 60.0 + 1e-4).all()


def test_parameter_count_positive(baseline_state):
    extractor = FeatureExtractor().fit([baseline_state])
    model = TravelerStudent(
        {"global_hidden_dim": 16, "alternative_hidden_dim": 8, "scorer_hidden_dim": 16},
        extractor.spec,
    )
    assert model.count_parameters() > 0
