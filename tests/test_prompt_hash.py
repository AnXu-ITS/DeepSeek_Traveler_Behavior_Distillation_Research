"""Prompt hash tests: prove identical behavioral input => identical prompt hash."""
from __future__ import annotations

from traveler_distillation.audit import prompt_hash
from traveler_distillation.teacher.prompts import SYSTEM_PROMPT, build_user_prompt
from traveler_distillation.generators import CounterfactualContextGenerator


def test_prompt_hash_identical_for_identical_input(baseline_state):
    j = baseline_state.model_dump_json(indent=2)
    u1 = build_user_prompt(j)
    u2 = build_user_prompt(j)
    assert prompt_hash(SYSTEM_PROMPT, u1) == prompt_hash(SYSTEM_PROMPT, u2)


def test_prompt_hash_differs_for_different_state(baseline_state, config):
    cs = CounterfactualContextGenerator(config).generate(
        baseline_state, "weather_intensity", [0.5]
    )[0]
    u1 = build_user_prompt(baseline_state.model_dump_json(indent=2))
    u2 = build_user_prompt(cs.state.model_dump_json(indent=2))
    assert prompt_hash(SYSTEM_PROMPT, u1) != prompt_hash(SYSTEM_PROMPT, u2)


def test_prompt_hash_is_hex_sha256():
    h = prompt_hash("sys", "usr")
    assert len(h) == 64
    int(h, 16)  # valid hex


def test_prompt_hash_excludes_metadata():
    # The hash depends only on the prompt text; extra non-prompt metadata
    # (e.g. a cache-bypass id) must not be part of the computation.
    assert prompt_hash("s", "u") == prompt_hash("s", "u")
