"""Tests for the frozen-release output guard (S7-W3 + S8 freeze, step 15)."""
from __future__ import annotations

from pathlib import Path

import pytest

from traveler_distillation.student.release_guard import (
    FROZEN_RELEASE_DIRS,
    assert_not_frozen_output,
)


def test_frozen_release_dir_is_known():
    assert "s7_w3_generic_core_v1" in FROZEN_RELEASE_DIRS
    assert "s8_supply_aware_v1" in FROZEN_RELEASE_DIRS
    assert "s9_supply_aware_v2" in FROZEN_RELEASE_DIRS


@pytest.mark.parametrize(
    "path",
    [
        "outputs/s7_w3_generic_core_v1/checkpoint/model.pt",
        "outputs/s7_w3_generic_core_v1",
        "releases/s7_w3_generic_core_v1/checkpoints/best.pt",
        "outputs/s8_supply_aware_v1/checkpoint/model.pt",
        "releases/s8_supply_aware_v1",
        "outputs/phase_c/s8_supply_aware_v1/population.xml",
        "outputs/s9_supply_aware_v2/checkpoint/model.pt",
        "releases/s9_supply_aware_v2/checkpoint/model.pt",
    ],
)
def test_paths_inside_frozen_release_are_rejected(path):
    with pytest.raises(RuntimeError, match="frozen release"):
        assert_not_frozen_output(path)


def test_ordinary_paths_are_allowed():
    p = assert_not_frozen_output("outputs/student_s8_w1")
    assert p == Path("outputs/student_s8_w1").resolve()


def test_guard_is_for_output_paths_only():
    # The guard rejects ANY resolved path inside the frozen release because it is
    # only ever called on output directories. Loading the frozen checkpoint for
    # S8 is a normal read and must not go through this function.
    with pytest.raises(RuntimeError, match="frozen release"):
        assert_not_frozen_output("releases/s7_w3_generic_core_v1/checkpoint/model.pt")
