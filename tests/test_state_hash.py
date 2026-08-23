"""State hash tests."""
from __future__ import annotations

from traveler_distillation.audit import state_hash
from traveler_distillation.generators import CounterfactualContextGenerator


def test_state_hash_deterministic(baseline_state):
    assert state_hash(baseline_state) == state_hash(baseline_state)


def test_state_hash_differs_on_context_change(baseline_state, config):
    cf = CounterfactualContextGenerator(config)
    cs = cf.generate(baseline_state, "weather_intensity", [0.5])[0]
    assert state_hash(baseline_state) != state_hash(cs.state)


def test_state_hash_differs_on_persona_change(baseline_state):
    p = baseline_state.persona.model_copy(deep=True)
    p.income_group = "high"
    other = baseline_state.model_copy(deep=True)
    other.persona = p
    assert state_hash(baseline_state) != state_hash(other)


def test_state_hash_is_hex_sha256(baseline_state):
    h = state_hash(baseline_state)
    assert len(h) == 64
    int(h, 16)  # must be valid hex
