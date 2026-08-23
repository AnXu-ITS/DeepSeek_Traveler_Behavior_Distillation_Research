"""split_group_id grouping: baseline + counterfactuals stay in one split."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from traveler_distillation.student import group_aware_split, group_overlap


class _S:
    def __init__(self, sid, split_group_id, cf_group_id=None):
        self.sample_id = sid
        self.split_group_id = split_group_id
        self.counterfactual_group_id = cf_group_id


def _make_pairs(n_pairs=6):
    """Each pair = baseline + 2 counterfactuals sharing a split_group_id."""
    samples = []
    for p in range(n_pairs):
        sg = f"SG{p:03d}"
        samples.append(_S(f"B{p}", sg, None))
        samples.append(_S(f"C{p}_1", sg, f"CF{p}_1"))
        samples.append(_S(f"C{p}_2", sg, f"CF{p}_2"))
    return samples


def _sg(s):
    return s.split_group_id


def test_split_group_keeps_baseline_with_counterfactuals():
    samples = _make_pairs()
    train, val, test = group_aware_split(samples, group_field="split_group_id", seed=42)
    assert not group_overlap(train, val, test, group_field="split_group_id")
    assert len(train) + len(val) + len(test) == len(samples)

    # every split_group lives in exactly one split
    for split in (train, val, test):
        for s in split:
            assert not any(s2 not in split and _sg(s2) == _sg(s) for s2 in samples)


def test_split_group_no_cross_split_counterfactual_leak():
    samples = _make_pairs()
    train, val, test = group_aware_split(samples, group_field="split_group_id", seed=1)
    assert not group_overlap(train, val, test, group_field="split_group_id")
    assert not group_overlap(train, val, test, group_field="counterfactual_group_id")


def _load_enumerate_states():
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_aggregated_teacher_dataset.py"
    spec = importlib.util.spec_from_file_location("gen_agg_teacher", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.enumerate_states


def test_enumerate_states_sets_split_and_persona_groups(config):
    enumerate_states = _load_enumerate_states()
    states = enumerate_states(config, num_personas=2, num_trips=1, seed=42)

    assert len(states) > 0
    # baseline + its counterfactuals share the same split_group_id and persona_group_id
    by_sample = {st["sample_id"]: st for st in states}
    baselines = [st for st in states if st["perturbation"].axis == "baseline"]
    assert len(baselines) == 2

    for b in baselines:
        cf_states = [st for st in states if st["baseline_sample_id"] == b["sample_id"]]
        assert len(cf_states) > 0
        for cf in cf_states:
            assert cf["split_group_id"] == b["split_group_id"]
            assert cf["persona_group_id"] == b["persona_group_id"]
            assert cf["persona_group_id"] is not None

    # baselines have no counterfactual group, CF states do
    assert all(st["cf_group_id"] is None for st in baselines)
    assert all(st["cf_group_id"] is not None for st in states if st["perturbation"].axis != "baseline")
