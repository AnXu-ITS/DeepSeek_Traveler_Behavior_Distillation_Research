"""Training step smoke test: forward + backward + optimizer.step."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from traveler_distillation.schemas.action import UniversalTravelerAction
from traveler_distillation.schemas.dataset import Perturbation
from traveler_distillation.dataset.aggregation import (
    AggregatedTeacherTarget,
    AggregationMetadata,
    aggregate_repeats,
)
from traveler_distillation.student import (
    FeatureExtractor,
    TravelerStudent,
    student_loss,
    AggregatedTeacherDataset,
    collate_batch,
)


def _targets(states):
    targets = []
    for i, s in enumerate(states):
        modes = s.available_modes
        probs = {m: 1.0 / len(modes) for m in modes}
        action = UniversalTravelerAction(
            selected_mode=modes[0], mode_probabilities=probs, departure_time_shift_min=0, confidence=0.8
        )
        agg = aggregate_repeats([action, action, action])
        targets.append(
            AggregatedTeacherTarget(
                sample_id=f"S{i}",
                counterfactual_group_id=f"CF{i}",
                perturbation=Perturbation(axis="baseline", level=0.0),
                state=s,
                teacher_aggregate=agg,
                aggregation_metadata=AggregationMetadata(k=3, prompt_version="v", model="m"),
            )
        )
    return targets


def test_training_step_backward(baseline_state):
    states = [baseline_state, baseline_state.model_copy(deep=True)]
    targets = _targets(states)
    extractor = FeatureExtractor().fit(states)
    ds = AggregatedTeacherDataset(targets, extractor)
    loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_batch)

    model = TravelerStudent(
        {"global_hidden_dim": 16, "alternative_hidden_dim": 8, "scorer_hidden_dim": 16,
         "cat_embedding_dim": 4, "mode_embedding_dim": 4, "dropout": 0.0},
        extractor.spec,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    batch = next(iter(loader))
    out = model(batch)
    losses = student_loss(
        out,
        batch["target_probs"],
        batch["target_mode_idx"],
        batch["target_departure"],
    )
    loss = losses["total"].mean()
    assert torch.isfinite(loss)

    loss_before = loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # after a step, parameters should have changed (loss should move)
    out2 = model(batch)
    loss2 = student_loss(out2, batch["target_probs"], batch["target_mode_idx"], batch["target_departure"])["total"].mean()
    assert torch.isfinite(loss2)
    assert loss2.item() != loss_before
