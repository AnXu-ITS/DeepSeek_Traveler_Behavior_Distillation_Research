"""Counterfactual metrics tests."""
from __future__ import annotations

import pytest

from traveler_distillation.audit import (
    spearman_rank,
    direction_reversals,
    analyze_counterfactual,
    elasticity_preview,
)
from traveler_distillation.generators import CounterfactualContextGenerator
from traveler_distillation.schemas.dataset import (
    Perturbation,
    TeacherMetadata,
    TeacherDatasetSample,
)
from traveler_distillation.schemas.action import UniversalTravelerAction


def test_spearman_monotonic_increasing():
    x = [0.0, 0.25, 0.5, 0.75, 1.0]
    y = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert spearman_rank(x, y) == pytest.approx(1.0)


def test_spearman_monotonic_decreasing():
    x = [0.0, 0.25, 0.5, 0.75, 1.0]
    y = [0.5, 0.4, 0.3, 0.2, 0.1]
    assert spearman_rank(x, y) == pytest.approx(-1.0)


def test_direction_reversals():
    # +0.07, -0.05, -0.04, -0.03 -> one reversal
    assert direction_reversals([0.75, 0.82, 0.77, 0.73, 0.70]) == 1
    assert direction_reversals([0.1, 0.2, 0.3, 0.4]) == 0
    # up, down, up -> two reversals
    assert direction_reversals([0.1, 0.3, 0.2, 0.4]) == 2


def _action(mode, probs):
    return UniversalTravelerAction(
        selected_mode=mode, mode_probabilities=probs,
        departure_time_shift_min=0, confidence=0.8,
    )


def _meta():
    return TeacherMetadata(model="mock", prompt_version="teacher_v0.1", dataset_version="t")


def test_analyze_counterfactual_fare_direction(baseline_state, config):
    cf_gen = CounterfactualContextGenerator(config)
    cf_samples = cf_gen.generate(baseline_state, "fare_multiplier", [1.5, 2.0])
    samples = []
    baseline_id = "S000001"
    samples.append(
        TeacherDatasetSample(
            sample_id=baseline_id,
            perturbation=Perturbation(axis="baseline", level=0.0),
            state=baseline_state,
            teacher=_action("pt", {"pt": 0.6, "bike": 0.4}),
            teacher_metadata=_meta(),
        )
    )
    for i, cs in enumerate(cf_samples, 1):
        # pt probability decreases as fare rises
        p_pt = 0.6 - i * 0.15
        samples.append(
            TeacherDatasetSample(
                sample_id=f"S00000{i + 1}",
                counterfactual_group_id="CF_fare_multiplier_000001",
                baseline_sample_id=baseline_id,
                perturbation=Perturbation(axis="fare_multiplier", level=cs.level),
                state=cs.state,
                teacher=_action("pt", {"pt": p_pt, "bike": 1 - p_pt}),
                teacher_metadata=_meta(),
            )
        )
    res = analyze_counterfactual(samples)
    assert "fare_multiplier" in res["axes"]
    rho = res["axes"]["fare_multiplier"]["mode_stats"]["pt"]["spearman_rho"]
    assert rho < 0


def test_elasticity_preview(baseline_state, config):
    cf_gen = CounterfactualContextGenerator(config)
    cf_samples = cf_gen.generate(baseline_state, "fare_multiplier", [2.0])
    samples = [
        TeacherDatasetSample(
            sample_id="S000001",
            perturbation=Perturbation(axis="baseline", level=0.0),
            state=baseline_state,
            teacher=_action("pt", {"pt": 0.6, "bike": 0.4}),
            teacher_metadata=_meta(),
        ),
        TeacherDatasetSample(
            sample_id="S000002",
            counterfactual_group_id="CF_fare_multiplier_000001",
            baseline_sample_id="S000001",
            perturbation=Perturbation(axis="fare_multiplier", level=2.0),
            state=cf_samples[0].state,
            teacher=_action("pt", {"pt": 0.4, "bike": 0.6}),
            teacher_metadata=_meta(),
        ),
    ]
    e = elasticity_preview(samples)
    assert "fare_multiplier" in e
    entry = e["fare_multiplier"][0]["entries"][0]
    # baseline level 1.0, counterfactual 2.0, pt: (0.4-0.6)/(2.0-1.0) = -0.2
    assert entry["elasticity"]["pt"] == pytest.approx(-0.2)
