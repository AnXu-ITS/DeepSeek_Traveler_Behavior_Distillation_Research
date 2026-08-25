"""S7 mechanism quadruplet dataset unit tests.

Covers: JSONL loading, quadruplet completeness enforcement, split/axis
filtering, mode-alignment of the four members, and collation shapes.
"""
from __future__ import annotations

import json

import pytest
import torch

from traveler_distillation.student import (
    FeatureExtractor,
    load_mechanism_quadruplets,
    MechanismQuadrupletDataset,
    collate_quadruplets,
)
from traveler_distillation.student.mechanism_dataset import ROLES


def _mutate(state, *, context_updates=None, alt_updates=None):
    """Deep-copy a state and apply targeted updates (context / car alternative)."""
    s = state.model_copy(deep=True)
    if context_updates:
        for k, v in context_updates.items():
            setattr(s.context, k, v)
    if alt_updates:
        for a in s.alternatives:
            if a.mode == "car":
                for k, v in alt_updates.items():
                    setattr(a, k, v)
    return s


def _member(state, probs, k=3):
    return {
        "state": state.model_dump(),
        "teacher_probs": probs,
        "teacher_departure": 0.0,
        "teacher_k": k,
    }


def _quad_row(gid, axis, persona_id, trip_id, split, base, natural, broken, mediator):
    return {
        "audit_group_id": gid,
        "axis_id": axis,
        "persona_id": persona_id,
        "trip_id": trip_id,
        "split": split,
        "members": {
            "baseline": base,
            "natural": natural,
            "broken": broken,
            "mediator": mediator,
        },
    }


def _make_quads_jsonl(tmp_path, persona, trip, baseline_state):
    """Two synthetic quadruplets: one parking_cost (train), one congestion (test)."""
    modes = baseline_state.available_modes
    n = len(modes)
    probs = {m: 1.0 / n for m in modes}

    pA = baseline_state
    pB = _mutate(pA, context_updates={"parking_cost_multiplier": 3.0},
                 alt_updates={"monetary_cost": 25.0})
    pC = _mutate(pA, context_updates={"parking_cost_multiplier": 3.0})
    pD = _mutate(pA, alt_updates={"monetary_cost": 25.0})

    cA = _mutate(pA)
    cB = _mutate(pA, context_updates={"road_congestion": 0.8},
                 alt_updates={"travel_time_min": 60.0, "reliability_delay_min": 12.0})
    cC = _mutate(pA, context_updates={"road_congestion": 0.8})
    cD = _mutate(pA, alt_updates={"travel_time_min": 60.0, "reliability_delay_min": 12.0})

    rows = [
        _quad_row("P::T::parking_cost", "parking_cost", persona.persona_id, trip.trip_id,
                  "train", _member(pA, probs), _member(pB, probs, k=5),
                  _member(pC, probs, k=5), _member(pD, probs, k=5)),
        _quad_row("P::T::congestion", "congestion", persona.persona_id, trip.trip_id,
                  "test", _member(cA, probs), _member(cB, probs, k=5),
                  _member(cC, probs, k=5), _member(cD, probs, k=5)),
    ]
    path = tmp_path / "quadruplets.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path, pA


def test_load_and_split_filter(tmp_path, persona, trip, baseline_state):
    path, _ = _make_quads_jsonl(tmp_path, persona, trip, baseline_state)

    quads = load_mechanism_quadruplets(path)
    assert len(quads) == 2
    assert {q.axis_id for q in quads} == {"parking_cost", "congestion"}
    assert all(set(q.members) == set(ROLES) for q in quads)

    train = load_mechanism_quadruplets(path, split="train")
    assert [q.axis_id for q in train] == ["parking_cost"]
    assert train[0].A.teacher_k == 3 and train[0].B.teacher_k == 5

    cong = load_mechanism_quadruplets(path, axes=["congestion"])
    assert [q.axis_id for q in cong] == ["congestion"]


def test_incomplete_quadruplet_raises(tmp_path, persona, trip, baseline_state):
    path, _ = _make_quads_jsonl(tmp_path, persona, trip, baseline_state)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    del rows[0]["members"]["mediator"]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing member"):
        load_mechanism_quadruplets(path)


def test_members_mode_aligned_and_collate(tmp_path, persona, trip, baseline_state):
    path, _ = _make_quads_jsonl(tmp_path, persona, trip, baseline_state)
    quads = load_mechanism_quadruplets(path)

    states = [m.state for q in quads for m in q.members.values()]
    extractor = FeatureExtractor().fit(states)

    ds = MechanismQuadrupletDataset(quads, extractor)
    assert len(ds) == 2
    batch = collate_quadruplets([ds[0], ds[1]])

    for key in ("A", "B", "C", "D"):
        sub = batch[key]
        assert sub["global_cat"].shape[0] == 2
        assert sub["alt_num"].ndim == 3
        assert sub["target_probs"].shape == sub["alt_mask"].shape
    # members of a quadruplet share persona+trip -> identical alternative order
    for key in ("B", "C", "D"):
        assert torch.equal(batch["A"]["alt_mode_idx"], batch[key]["alt_mode_idx"])
        assert torch.equal(batch["A"]["alt_mask"], batch[key]["alt_mask"])


def test_target_vector_available_only(tmp_path, persona, trip, baseline_state):
    path, base = _make_quads_jsonl(tmp_path, persona, trip, baseline_state)
    quads = load_mechanism_quadruplets(path)
    extractor = FeatureExtractor().fit([base])
    from traveler_distillation.student.mechanism_dataset import encode_quad_member

    member = quads[0].A
    enc = encode_quad_member(member, extractor)
    for i, alt in enumerate(member.state.alternatives):
        if alt.available:
            assert enc["target_probs"][i].item() == pytest.approx(
                member.teacher_probs[alt.mode]
            )
        else:
            assert enc["target_probs"][i].item() == 0.0
            assert enc["alt_mask"][i].item() == 0.0
    assert enc["target_mode_idx"].item() == int(torch.argmax(enc["target_probs"]))
