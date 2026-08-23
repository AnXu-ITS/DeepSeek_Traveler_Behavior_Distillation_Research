#!/usr/bin/env python
"""Offline token/cost estimation from the generated dataset (no API calls).

Estimates per-call prompt/output tokens from the actual stored JSONL:
- prompt_tokens ~ len(state_json) / chars_per_token + system_prompt_tokens
- completion_tokens ~ len(action_json) / chars_per_token
Prints totals and cost under a configurable price table.

Usage:
    python scripts/estimate_dataset_cost.py \
        --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl \
        --repeats data/student_v0_2_a/repeat_records.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.teacher.prompts import SYSTEM_PROMPT

CHARS_PER_TOKEN = 3.6  # JSON/ASCII-dense text; conservative for English+JSON

# Public reference prices (USD per 1M tokens). deepseek-v4-pro via the corp
# gateway may differ; these are scenarios, NOT a quote.
PRICE_SCENARIOS = {
    "v3-class": {"input": 0.28, "output": 0.42},
    "reasoner-class": {"input": 0.55, "output": 2.19},
    "premium": {"input": 2.00, "output": 8.00},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repeats", required=True)
    args = ap.parse_args()

    states = [json.loads(l) for l in Path(args.dataset).read_text(encoding="utf-8").splitlines() if l.strip()]
    repeats = [json.loads(l) for l in Path(args.repeats).read_text(encoding="utf-8").splitlines() if l.strip()]

    state_lens = [len(json.dumps(s["state"], ensure_ascii=False)) for s in states]
    action_lens = [len(json.dumps(r["action"], ensure_ascii=False)) for r in repeats]

    sys_tokens = len(SYSTEM_PROMPT) / CHARS_PER_TOKEN
    avg_state_tokens = sum(state_lens) / len(state_lens) / CHARS_PER_TOKEN
    avg_action_tokens = sum(action_lens) / len(action_lens) / CHARS_PER_TOKEN
    per_call_in = sys_tokens + avg_state_tokens
    per_call_out = avg_action_tokens

    n_states = len(states)
    n_calls = len(repeats)

    print(f"states={n_states}  repeat_records={n_calls}")
    print(f"avg state json chars = {sum(state_lens)/len(state_lens):.0f}  "
          f"(min {min(state_lens)}, max {max(state_lens)})")
    print(f"avg action json chars = {sum(action_lens)/len(action_lens):.0f}  "
          f"(min {min(action_lens)}, max {max(action_lens)})")
    print(f"estimated per-call: input ~{per_call_in:.0f} tok, output ~{per_call_out:.0f} tok")
    print(f"total calls = {n_calls} -> input ~{n_calls*per_call_in/1e6:.2f}M tok, "
          f"output ~{n_calls*per_call_out/1e6:.3f}M tok")
    print()
    print("cost scenarios (USD):")
    for name, p in PRICE_SCENARIOS.items():
        cost_per_call = per_call_in * p["input"] / 1e6 + per_call_out * p["output"] / 1e6
        total = n_calls * cost_per_call
        print(f"  {name:16s} ${cost_per_call:.5f}/call -> ${total:.2f} total "
              f"({n_calls} calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
