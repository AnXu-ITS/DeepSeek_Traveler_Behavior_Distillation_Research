"""Aggregation metric tests: mean distribution and single-to-aggregate L1."""
from __future__ import annotations

import pytest

from traveler_distillation.audit import (
    aggregate_distribution,
    mean_single_to_aggregate_l1,
    mean_pairwise_l1,
    max_pairwise_l1,
    l1,
)


def test_aggregate_distribution_mean():
    dists = [{"bike": 0.7, "pt": 0.3}, {"bike": 0.5, "pt": 0.5}]
    agg = aggregate_distribution(dists)
    assert agg["bike"] == pytest.approx(0.6)
    assert agg["pt"] == pytest.approx(0.4)


def test_aggregate_distribution_single():
    assert aggregate_distribution([{"bike": 0.9, "pt": 0.1}]) == pytest.approx(
        {"bike": 0.9, "pt": 0.1}
    )


def test_mean_single_to_aggregate_l1():
    dists = [{"bike": 1.0, "pt": 0.0}, {"bike": 0.0, "pt": 1.0}]
    # aggregate = {bike: 0.5, pt: 0.5}; each single L1 to aggregate = 1.0
    assert mean_single_to_aggregate_l1(dists) == pytest.approx(1.0)


def test_pairwise_l1_mean_and_max():
    dists = [
        {"bike": 1.0},
        {"bike": 0.8, "pt": 0.2},
        {"bike": 0.5, "pt": 0.5},
    ]
    # pairs: (0,1)=0.4, (0,2)=1.0, (1,2)=0.6 -> mean=0.6667, max=1.0
    assert mean_pairwise_l1(dists) == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert max_pairwise_l1(dists) == pytest.approx(1.0)


def test_l1_between_aggregates_is_zero_when_identical():
    dists = [{"bike": 0.8, "pt": 0.2}] * 3
    assert l1(aggregate_distribution(dists), aggregate_distribution(dists)) == 0.0
