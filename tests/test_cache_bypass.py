"""Cache-bypass tests: bypass metadata must never change behavioral input."""
from __future__ import annotations

from traveler_distillation.audit import state_hash, prompt_hash
from traveler_distillation.teacher import DeepSeekTeacherClient
from traveler_distillation.teacher.prompts import SYSTEM_PROMPT, build_user_prompt


def test_cache_bypass_id_does_not_change_state_or_prompt_hash(baseline_state):
    h_state = state_hash(baseline_state)
    system = SYSTEM_PROMPT
    user = build_user_prompt(baseline_state.model_dump_json(indent=2))
    h_prompt = prompt_hash(system, user)

    # A cache-bypass id lives only in gateway request metadata; it must never
    # affect the state hash or the behavioral prompt hash.
    for bypass_id in ("bp-aaa", "bp-bbb", "bp-ccc"):
        assert state_hash(baseline_state) == h_state
        assert prompt_hash(system, user) == h_prompt


def test_client_build_prompt_identical_regardless_of_cache_bypass(baseline_state):
    client = DeepSeekTeacherClient(base_url="http://127.0.0.1:9/v1", api_key="test-key")
    s1, u1 = client.build_prompt(baseline_state)
    s2, u2 = client.build_prompt(baseline_state)
    assert s1 == s2 == SYSTEM_PROMPT
    assert u1 == u2
    assert prompt_hash(s1, u1) == prompt_hash(s2, u2)
    # cache_bypass is a request-body control field, not part of build_prompt.
    assert "no-cache" not in s1 and "no-cache" not in u1
