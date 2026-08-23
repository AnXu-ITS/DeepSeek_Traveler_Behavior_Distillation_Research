#!/usr/bin/env python
"""Validate a generated K=3 aggregated teacher dataset for integrity.

Checks (exit code 0 = all pass):
  1. JSONL parse + unique sample ids.
  2. Each aggregated sample has exactly K repeat records with DISTINCT
     completion ids (cache-bypass must have held; no gateway-cache reuse).
  3. Aggregate probabilities sum to ~1.
  4. Every counterfactual sample links to a baseline sample via
     ``baseline_sample_id``, and the two share the same ``split_group_id``
     (baseline + its counterfactual curves are kept together for elasticity).
  5. Every baseline has at least one counterfactual pointing at it.
  6. No cross-file orphan: every repeat record maps to an aggregated sample.

Usage:
    python scripts/validate_aggregated_dataset.py \
        --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl \
        --repeats data/student_v0_2_a/repeat_records.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"[FAIL] {path.name}:{i} unparseable: {exc}")
            raise
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repeats", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    dataset = _load_jsonl(Path(args.dataset))
    repeats = _load_jsonl(Path(args.repeats))

    failures: list[str] = []
    info: list[str] = []

    # 1. unique sample ids
    ids = [s["sample_id"] for s in dataset]
    dup = [sid for sid, n in Counter(ids).items() if n > 1]
    if dup:
        failures.append(f"duplicate sample ids: {dup}")
    info.append(f"aggregated samples = {len(dataset)}")

    # 2. repeat coverage + distinct completion ids per sample
    reps_by_sample = defaultdict(list)
    for r in repeats:
        reps_by_sample[r["sample_id"]].append(r)

    bad_k = [sid for sid, rs in reps_by_sample.items() if len(rs) != args.k]
    if bad_k:
        failures.append(f"{len(bad_k)} samples with repeat count != {args.k}: {bad_k[:10]}")

    dup_cid = []
    for sid, rs in reps_by_sample.items():
        cids = [r.get("completion_id") for r in rs if r.get("completion_id")]
        if len(set(cids)) != len(cids):
            dup_cid.append(sid)
    if dup_cid:
        failures.append(f"duplicate completion ids within a state (cache reuse!): {dup_cid}")
    info.append(f"repeat records = {len(repeats)} across {len(reps_by_sample)} states")

    # orphans in repeat records
    rep_orphans = [sid for sid in reps_by_sample if sid not in set(ids)]
    if rep_orphans:
        failures.append(f"repeat records with no aggregated sample: {rep_orphans[:10]}")

    # 3. probability sums
    bad_sum = []
    for s in dataset:
        probs = s.get("teacher_aggregate", {}).get("mode_probabilities", {})
        total = sum(float(p) for p in probs.values())
        if abs(total - 1.0) > 1e-4:
            bad_sum.append((s["sample_id"], round(total, 6)))
    if bad_sum:
        failures.append(f"aggregate probabilities not summing to 1: {bad_sum[:10]}")

    # 4/5. baseline linkage
    by_id = {s["sample_id"]: s for s in dataset}
    baselines = [s for s in dataset if s.get("perturbation", {}).get("axis") == "baseline"]
    cfs = [s for s in dataset if s.get("perturbation", {}).get("axis") != "baseline"]
    info.append(f"baselines = {len(baselines)}, counterfactuals = {len(cfs)}")

    bad_link = []
    for s in cfs:
        bid = s.get("baseline_sample_id")
        base = by_id.get(bid)
        if base is None:
            bad_link.append((s["sample_id"], "missing baseline", bid))
            continue
        if s.get("split_group_id") != base.get("split_group_id"):
            bad_link.append((s["sample_id"], "split_group mismatch", s["split_group_id"]))
    if bad_link:
        failures.append(f"broken baseline linkage: {bad_link[:10]}")

    # every baseline has >=1 CF pointing at it
    cf_ptrs = Counter(s.get("baseline_sample_id") for s in cfs)
    uncovered = [b["sample_id"] for b in baselines if cf_ptrs.get(b["sample_id"], 0) == 0]
    if uncovered:
        failures.append(f"baselines with no counterfactual: {uncovered}")

    # split group coverage
    n_groups = len({s.get("split_group_id") for s in dataset})
    info.append(f"split groups = {n_groups}")

    # summary
    print("\n".join(f"[info] {x}" for x in info))
    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        print(f"\n{len(failures)} problem(s) found")
        return 1
    print("\n[OK] dataset valid")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FAIL] validation crashed: {exc}")
        sys.exit(1)
