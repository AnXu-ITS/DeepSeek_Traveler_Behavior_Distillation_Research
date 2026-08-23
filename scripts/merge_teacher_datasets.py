#!/usr/bin/env python
"""Merge an existing aggregated teacher dataset with a newly generated part.

Used for S3 (persona 20->40): the new personas' states (fresh sample ids,
persona ids, trip ids) are appended to the existing dataset so the combined
file is self-contained. Assumes the two parts use disjoint id spaces (the
caller guarantees this via --persona-offset/--trip-offset/--id-prefix).

Usage:
    python scripts/merge_teacher_datasets.py \
        --base data/student_v0_3_s4 \
        --new data/student_v0_3_s3_new \
        --output data/student_v0_3_s3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _append(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base = Path(args.base)
    new = Path(args.new)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    base_ds = _load_jsonl(base / "aggregated_teacher_dataset.jsonl")
    base_rep = _load_jsonl(base / "repeat_records.jsonl")
    new_ds = _load_jsonl(new / "aggregated_teacher_dataset.jsonl")
    new_rep = _load_jsonl(new / "repeat_records.jsonl")

    base_ids = {r["sample_id"] for r in base_ds}
    new_ids = {r["sample_id"] for r in new_ds}
    overlap = base_ids & new_ids
    if overlap:
        print(f"[FAIL] {len(overlap)} overlapping sample ids: {sorted(overlap)[:10]}")
        return 1

    # seed output with the base, then append the new part
    (out / "aggregated_teacher_dataset.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in base_ds),
        encoding="utf-8",
    )
    (out / "repeat_records.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in base_rep),
        encoding="utf-8",
    )
    _append(out / "aggregated_teacher_dataset.jsonl", new_ds)
    _append(out / "repeat_records.jsonl", new_rep)

    # incomplete samples from the new part (informational only)
    new_inc = _load_jsonl(new / "incomplete_samples.jsonl")
    if new_inc:
        (out / "incomplete_samples.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_inc),
            encoding="utf-8",
        )

    print(f"base: {len(base_ds)} states / {len(base_rep)} repeats")
    print(f"new:  {len(new_ds)} states / {len(new_rep)} repeats")
    print(f"combined: {len(base_ds) + len(new_ds)} states / "
          f"{len(base_rep) + len(new_rep)} repeats -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
