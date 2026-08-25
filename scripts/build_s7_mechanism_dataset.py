#!/usr/bin/env python
"""Build the S7 mechanism quadruplet dataset from S6 audit data (reuse, no API).

Reads ``data/causal_audit/states_with_teacher.jsonl`` (S6 teacher targets are
reused VERBATIM — S7 constraint: zero new API variables in the first round),
groups the A/B/C/D quadruplets, applies the S7 filters, and writes one row per
quadruplet to ``data/student_s7_mechanism/quadruplets.jsonl``.

Filters (S7 instructions §9 + constraint 1):
  - axes: congestion + parking_cost only (transit_delay is out of S7 scope);
  - car availability: the car alternative must be AVAILABLE in all four states
    (otherwise E_natural ≈ 0 and the ratios lose meaning);
  - split: S3-C persona holdout applied VERBATIM via the split manifest, so the
    final causal test set (test personas) never enters training or selection.
    The train/val/test persona sets are asserted disjoint and the assignment is
    frozen in ``split_manifest.json`` next to the quadruplets.

Usage:
    python scripts/build_s7_mechanism_dataset.py \
        --states data/causal_audit/states_with_teacher.jsonl \
        --split-manifest outputs/student_v0_3_s3_c/split_manifest.json \
        --output data/student_s7_mechanism/quadruplets.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ("baseline", "natural", "broken", "mediator")
S7_AXES = ("congestion", "parking_cost")


def _car_available(state: dict) -> bool:
    for alt in state["alternatives"]:
        if alt["mode"] == "car" and alt["available"]:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="data/causal_audit/states_with_teacher.jsonl")
    ap.add_argument("--split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--output", default="data/student_s7_mechanism/quadruplets.jsonl")
    args = ap.parse_args()

    records = [
        json.loads(line)
        for line in Path(args.states).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    personas = manifest["personas"]
    split_of = {}
    for sname in ("train", "val", "test"):
        for pid in personas[sname]:
            split_of[pid] = sname
    # hard integrity assertions (S7 constraint 1)
    train_s, val_s, test_s = map(set, (personas["train"], personas["val"], personas["test"]))
    assert not (train_s & val_s) and not (train_s & test_s) and not (val_s & test_s), \
        "split manifest personas overlap — cannot isolate the causal test set"
    assert split_of, "split manifest empty"

    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        groups[r["audit_group_id"]][r["state_type"]] = r

    n_incomplete = 0
    n_axis_skipped = 0
    n_car_skipped = 0
    n_missing_teacher = 0
    rows = []
    counts = Counter()
    for gid, members in sorted(groups.items()):
        if not all(role in members for role in ROLES):
            n_incomplete += 1
            continue
        axis = members["baseline"]["axis_id"]
        if axis not in S7_AXES:
            n_axis_skipped += 1
            continue
        states = {role: members[role]["state"] for role in ROLES}
        car_flags = {role: _car_available(states[role]) for role in ROLES}
        if not all(car_flags.values()):
            n_car_skipped += 1
            continue
        # availability must be identical across the quadruplet (same persona+trip)
        assert len(set(car_flags.values())) == 1
        targets = {role: members[role].get("teacher_aggregate") for role in ROLES}
        if any(t is None for t in targets.values()):
            n_missing_teacher += 1
            continue

        persona_id = members["baseline"]["persona_group_id"]
        split = split_of.get(persona_id)
        if split is None:
            print(f"[WARN] persona {persona_id} not in split manifest — skipped", file=sys.stderr)
            continue

        rows.append(
            {
                "audit_group_id": gid,
                "axis_id": axis,
                "persona_id": persona_id,
                "trip_id": gid.split("::")[1],
                "split": split,
                "members": {
                    role: {
                        "state": states[role],
                        "teacher_probs": targets[role]["mode_probabilities"],
                        "teacher_departure": float(targets[role]["departure_time_shift_min"]),
                        "teacher_k": int(members[role].get("teacher_k")),
                    }
                    for role in ROLES
                },
            }
        )
        counts[(axis, split)] += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_axis_split = defaultdict(dict)
    for (axis, split), n in counts.items():
        per_axis_split[axis][split] = n
    manifest_out = {
        "source": "data/causal_audit/states_with_teacher.jsonl (S6 verbatim, no new API)",
        "axes": list(S7_AXES),
        "car_available_filter": True,
        "persona_split": "s3c_persona_holdout (verbatim)",
        "counts_per_axis_split": dict(per_axis_split),
        "personas": personas,
        "persona_overlap": bool(train_s & val_s | train_s & test_s | val_s & test_s),
    }
    (out.parent / "split_manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"rows={len(rows)} incomplete={n_incomplete} axis_skipped={n_axis_skipped} "
          f"car_skipped={n_car_skipped} missing_teacher={n_missing_teacher}")
    print(json.dumps(per_axis_split, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
