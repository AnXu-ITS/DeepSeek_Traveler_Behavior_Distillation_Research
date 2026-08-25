#!/usr/bin/env python
"""Validate the S5 joint (multi-axis) teacher dataset for integrity.

Checks (exit 0 = pass):
  1. JSONL parse + unique sample ids.
  2. Each sample's repeat count == its own ``aggregation_metadata.k`` (mixed
     K=3/5/7 is expected here, unlike the fixed-K single-axis validator).
  3. Distinct completion ids per sample (no cache reuse).
  4. Aggregate probabilities sum to ~1.
  5. ``baseline_sample_id`` points to a valid S3 baseline with a matching
     ``split_group_id`` (baselines live in the base dataset, not here).
  6. ``perturbation.joint_axes`` is non-empty and resolves to a known
     combination in the joint config.

Usage:
    python scripts/validate_joint_dataset.py \
        --dataset data/student_s5_joint/aggregated_teacher_dataset.jsonl \
        --repeats data/student_s5_joint/repeat_records.jsonl \
        --base-dataset data/student_v0_3_s3/aggregated_teacher_dataset.jsonl \
        --joint-config configs/joint_sampling.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import resolve_combination  # noqa: E402


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
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    args = ap.parse_args()

    dataset = _load_jsonl(Path(args.dataset))
    repeats = _load_jsonl(Path(args.repeats))
    base = _load_jsonl(Path(args.base_dataset))
    combos = load_yaml(args.joint_config).get("joint_combinations", [])

    failures: list[str] = []
    info: list[str] = []

    # 1. unique ids
    ids = [s["sample_id"] for s in dataset]
    dup = [sid for sid, n in Counter(ids).items() if n > 1]
    if dup:
        failures.append(f"duplicate sample ids: {dup}")
    info.append(f"joint samples = {len(dataset)}")

    # 2/3. per-sample repeat count == own k; distinct completion ids
    reps_by_sample = defaultdict(list)
    for r in repeats:
        reps_by_sample[r["sample_id"]].append(r)
    bad_k = []
    dup_cid = []
    k_counts = Counter()
    for s in dataset:
        sid = s["sample_id"]
        k = (s.get("aggregation_metadata") or {}).get("k")
        k_counts[k] += 1
        rs = reps_by_sample.get(sid, [])
        # A sample needs AT LEAST k repeats; the base-K (k5) view legitimately
        # has MORE repeats than its k for the K=7 subset states (7 sampled,
        # re-aggregated from the first k=5).
        if k is not None and len(rs) < k:
            bad_k.append((sid, len(rs), k))
        cids = [r.get("completion_id") for r in rs if r.get("completion_id")]
        if len(set(cids)) != len(cids):
            dup_cid.append(sid)
    if bad_k:
        failures.append(f"{len(bad_k)} samples with repeat count != own k: {bad_k[:10]}")
    if dup_cid:
        failures.append(f"duplicate completion ids within a state: {dup_cid}")
    info.append(f"repeat records = {len(repeats)}; K distribution = {dict(k_counts)}")

    # orphans
    rep_orphans = [sid for sid in reps_by_sample if sid not in set(ids)]
    if rep_orphans:
        failures.append(f"repeat records with no aggregated sample: {rep_orphans[:10]}")

    # 4. probability sums
    bad_sum = []
    for s in dataset:
        probs = s.get("teacher_aggregate", {}).get("mode_probabilities", {})
        total = sum(float(p) for p in probs.values())
        if abs(total - 1.0) > 1e-4:
            bad_sum.append((s["sample_id"], round(total, 6)))
    if bad_sum:
        failures.append(f"aggregate probabilities not summing to 1: {bad_sum[:10]}")

    # 5. baseline linkage to S3
    base_by_id = {s["sample_id"]: s for s in base if (s.get("perturbation") or {}).get("axis") == "baseline"}
    bad_link = []
    for s in dataset:
        bid = s.get("baseline_sample_id")
        b = base_by_id.get(bid)
        if b is None:
            bad_link.append((s["sample_id"], "missing S3 baseline", bid))
        elif s.get("split_group_id") != b.get("split_group_id"):
            bad_link.append((s["sample_id"], "split_group mismatch", s["split_group_id"]))
    if bad_link:
        failures.append(f"broken baseline linkage: {bad_link[:10]}")

    # 6. joint_axes present + resolves to known combination
    bad_combo = []
    for s in dataset:
        ja = (s.get("perturbation") or {}).get("joint_axes") or []
        if not ja:
            bad_combo.append((s["sample_id"], "empty joint_axes"))
            continue
        if resolve_combination(ja, combos) is None:
            bad_combo.append((s["sample_id"], "unknown combination", [j["axis"] for j in ja]))
    if bad_combo:
        failures.append(f"joint_axes issues: {bad_combo[:10]}")

    # seen/unseen counts
    seen = unseen = 0
    for s in dataset:
        ja = (s.get("perturbation") or {}).get("joint_axes") or []
        c = resolve_combination(ja, combos)
        if c is None:
            continue
        if c.get("seen_in_training", True):
            seen += 1
        else:
            unseen += 1
    info.append(f"seen joint states = {seen}, unseen (holdout) = {unseen}")

    print("\n".join(f"[info] {x}" for x in info))
    if failures:
        print("\n".join(f"[FAIL] {x}" for x in failures))
        print(f"\n{len(failures)} problem(s) found")
        return 1
    print("\n[OK] joint dataset valid")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FAIL] validation crashed: {exc}")
        sys.exit(1)
