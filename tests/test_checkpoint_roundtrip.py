"""Checkpoint save/reload round-trip tests."""
from __future__ import annotations

import torch

from traveler_distillation.student import FeatureExtractor, TravelerStudent


def _batch(state, extractor):
    f = extractor.encode(state)
    from traveler_distillation.student import collate_batch
    return collate_batch([{
        "global_cat": torch.tensor(f["global_cat"], dtype=torch.long),
        "global_num": torch.tensor(f["global_num"], dtype=torch.float32),
        "alt_mode_idx": torch.tensor(f["alt_mode_idx"], dtype=torch.long),
        "alt_num": torch.tensor(f["alt_num"], dtype=torch.float32),
        "alt_mask": torch.tensor(f["alt_available"], dtype=torch.float32),
    }])


def test_checkpoint_roundtrip(baseline_state, tmp_path):
    cfg = {"global_hidden_dim": 16, "alternative_hidden_dim": 8, "scorer_hidden_dim": 16,
           "cat_embedding_dim": 4, "mode_embedding_dim": 4, "dropout": 0.0}
    extractor = FeatureExtractor().fit([baseline_state])
    model = TravelerStudent(cfg, extractor.spec)
    model.eval()
    batch = _batch(baseline_state, extractor)

    with torch.no_grad():
        before = model(batch)["mode_probabilities"].clone()

    ckpt = tmp_path / "best.pt"
    torch.save(
        {"model_state": model.state_dict(), "feature_spec": extractor.spec, "config": cfg},
        ckpt,
    )

    # fresh model + load
    model2 = TravelerStudent(cfg, extractor.spec)
    model2.load_state_dict(torch.load(ckpt)["model_state"])
    model2.eval()
    with torch.no_grad():
        after = model2(batch)["mode_probabilities"]

    assert torch.allclose(before, after, atol=1e-6)
