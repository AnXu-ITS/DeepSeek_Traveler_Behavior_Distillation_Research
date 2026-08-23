"""K=3 vs K=5 aggregation comparison tests."""
from __future__ import annotations

import pytest

from traveler_distillation.audit import aggregate_distribution, l1


def _k3_vs_k5_l1(dists):
    return l1(aggregate_distribution(dists[:3]), aggregate_distribution(dists))


def test_k3_vs_k5_zero_when_distributions_identical():
    dists = [{"bike": 0.8, "pt": 0.2}] * 5
    assert _k3_vs_k5_l1(dists) == pytest.approx(0.0)


def test_k3_vs_k5_small_when_distributions_close():
    dists = [
        {"bike": 0.80, "pt": 0.20},
        {"bike": 0.82, "pt": 0.18},
        {"bike": 0.79, "pt": 0.21},
        {"bike": 0.81, "pt": 0.19},
        {"bike": 0.78, "pt": 0.22},
    ]
    assert _k3_vs_k5_l1(dists) < 0.05


def test_k3_and_k5_are_valid_distributions():
    dists = [
        {"bike": 0.8, "pt": 0.2},
        {"bike": 0.7, "pt": 0.3},
        {"bike": 0.9, "pt": 0.1},
        {"bike": 0.6, "pt": 0.4},
        {"bike": 1.0, "pt": 0.0},
    ]
    k3 = aggregate_distribution(dists[:3])
    k5 = aggregate_distribution(dists)
    for d in (k3, k5):
        assert abs(sum(d.values()) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in d.values())
