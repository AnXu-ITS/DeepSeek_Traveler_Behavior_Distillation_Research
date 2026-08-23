"""Group-aware split tests."""
from __future__ import annotations

from traveler_distillation.student import group_aware_split, group_overlap


class _S:
    def __init__(self, sid, gid):
        self.sample_id = sid
        self.counterfactual_group_id = gid


def _make(n_per_group=3, n_groups=6):
    samples = []
    for g in range(n_groups):
        for i in range(n_per_group):
            samples.append(_S(f"S{g}_{i}", f"CF{g:03d}"))
    return samples


def test_group_split_no_overlap():
    samples = _make()
    train, val, test = group_aware_split(samples, seed=42)
    assert not group_overlap(train, val, test)
    # every sample assigned exactly once
    assert len(train) + len(val) + len(test) == len(samples)


def test_group_split_keeps_group_whole():
    samples = _make()
    train, val, test = group_aware_split(samples, seed=42)
    groups_train = {s.counterfactual_group_id for s in train}
    groups_test = {s.counterfactual_group_id for s in test}
    assert groups_train & groups_test == set()


def test_group_split_singleton_groups():
    samples = [_S("S1", None), _S("S2", None), _S("S3", "CF001"), _S("S4", "CF001")]
    train, val, test = group_aware_split(samples, seed=0)
    assert not group_overlap(train, val, test)
    assert len(train) + len(val) + len(test) == 4


def test_group_split_ratios_sum_to_one_enforced():
    samples = _make()
    try:
        group_aware_split(samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.2)
        assert False, "expected ValueError"
    except ValueError:
        pass
