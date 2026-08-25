#!/usr/bin/env python
"""Deterministically rebuild the M1 (base-K) view of the S5 joint dataset.

The generator writes the FULL-K aggregated view (``aggregated_teacher_dataset.jsonl``,
where the K=7 causal subset states carry k=7) and, in parallel, a base-K view
(``aggregated_teacher_dataset_k5.jsonl``) where the same subset states are
re-aggregated from their first base-K repeats (M1 target). This script rebuilds
the base-K view from scratch so any states whose streaming write was interrupted
(e.g. by a crash) are backfilled.

For every state in the full view:
  - non-subset states (aggregation_metadata.k == combo base K): copied verbatim.
  - subset states (k > base K): re-aggregated from the first base-K repeats.

Usage:
    python scripts/rebuild_k5_view.py \
        --full data/student_s5_joint/aggregated_teacher_dataset.jsonl \
        --repeats data/student_s5_joint/repeat_records.jsonl \
        --joint-config configs/joint_sampling.yaml \
        --output data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.config import load_yaml
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget, AggregationMetadata, aggregate_repeats
from traveler_distillation.generators import resolve_combination
from traveler_distillation.schemas.action import UniversalTravelerAction


def _load(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="data/student_s5_joint/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--repeats", default="data/student_s5_joint/repeat_records.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--output", default="data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl")
    args = ap.parse_args()

    combos = load_yaml(args.joint_config).get("joint_combinations", [])
    full = _load(Path(args.full))
    repeats_by_sample = defaultdict(list)
    for r in _load(Path(args.repeats)):
        repeats_by_sample[r["sample_id"]].append(r)

    out_lines = []
    n_subset_rebuilt = 0
    n_copied = 0
    for s in full:
        combo = resolve_combination(s.get("perturbation", {}).get("joint_axes", []), combos)
        base_k = int(combo["teacher_repeats"]) if combo else None
        k = (s.get("aggregation_metadata") or {}).get("k")
        if k is None or base_k is None or k <= base_k:
            out_lines.append(s)
            n_copied += 1
            continue
        # subset state: rebuild from first base_k repeats
        records = sorted(repeats_by_sample.get(s["sample_id"], []), key=lambda r: r.get("repeat_index", 0))
        if len(records) < base_k:
            print(f"[WARN] {s['sample_id']}: only {len(records)} repeats available (< {base_k}), skipping base-K rebuild")
            continue
        actions = [UniversalTravelerAction.model_validate(r["action"]) for r in records[:base_k]]
        agg = aggregate_repeats(actions)
        target = AggregatedTeacherTarget(
            sample_id=s["sample_id"],
            counterfactual_group_id=s.get("counterfactual_group_id"),
            persona_group_id=s.get("persona_group_id"),
            baseline_sample_id=s.get("baseline_sample_id"),
            split_group_id=s.get("split_group_id"),
            perturbation=s["perturbation"],
            state=s["state"],
            teacher_aggregate=agg,
            aggregation_metadata=AggregationMetadata(
                k=base_k,
                aggregation_method="mean_probability",
                prompt_version=(s.get("aggregation_metadata") or {}).get("prompt_version", "teacher_v0.1"),
                model=(s.get("aggregation_metadata") or {}).get("model", "unknown"),
                source_completion_ids=[r.get("completion_id") for r in records[:base_k] if r.get("completion_id")],
            ),
            status="complete",
        )
        out_lines.append(json.loads(target.model_dump_json()))
        n_subset_rebuilt += 1

    out = Path(args.output)
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    print(f"copied={n_copied} subset_rebuilt={n_subset_rebuilt} total={len(out_lines)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
