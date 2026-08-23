"""Masked softmax tests."""
from __future__ import annotations

import pytest
import torch

from traveler_distillation.student import masked_softmax


def test_masked_softmax_sums_to_one_over_available():
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    probs = masked_softmax(logits, mask)
    assert probs.shape == (1, 4)
    assert probs[0, 2] == pytest.approx(0.0)
    assert probs[0, 3] == pytest.approx(0.0)
    assert probs[0, :2].sum() == pytest.approx(1.0)


def test_masked_softmax_padding_probability_is_zero():
    logits = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    probs = masked_softmax(logits, mask)
    assert probs[0, 0] == pytest.approx(1.0)
    assert probs[0, 1:].sum() == pytest.approx(0.0)


def test_masked_softmax_batch():
    logits = torch.tensor([[1.0, 2.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    probs = masked_softmax(logits, mask)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    assert probs[0, 2:] .sum() == pytest.approx(0.0)
    assert probs[1, 3] == pytest.approx(0.0)
