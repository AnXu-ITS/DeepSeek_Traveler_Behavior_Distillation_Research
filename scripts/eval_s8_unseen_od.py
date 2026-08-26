#!/usr/bin/env python
"""S8 unseen-OD test (S8 instructions §15, RQ-S8-4).

The S8 test split is a DOUBLE holdout: unseen personas AND unseen ODs. This
script audits the OD-holdout guarantee (train ODs never appear in test, and
every test state uses only test-ODs) and emits the focused unseen-OD summary
drawn from outputs/s8_accessibility_eval/eval_metrics.json.

Usage:
    python scripts/eval_s8_unseen_od.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/singapore_accessibility/records.jsonl")
    ap.add_argument("--split-manifest", default="data/singapore_accessibility/split_manifest.json")
    ap.add_argument("--accessibility-eval", default="outputs/s8_accessibility_eval/eval_metrics.json")
    ap.add_argument("--output", default="outputs/s8_unseen_od")
    args = ap.parse_args()

    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    od_split = manifest["od_split"]
    train_ods = set(od_split["train"])
    test_ods = set(od_split["test"])
    persona_test = set(manifest["persona_split"]["test"])

    records = [json.loads(l) for l in Path(args.records).read_text(encoding="utf-8").splitlines() if l.strip()]
    test_records = [r for r in records if r["split"] == "test"]
    train_records = [r for r in records if r["split"] == "train"]

    assert not (train_ods & test_ods), "OD holdout violated"
    for r in test_records:
        assert r["od_index"] in test_ods, f"test state {r['sample_id']} uses a non-test OD"
        assert r["persona_id"] in persona_test, f"test state {r['sample_id']} uses a non-test persona"
    for r in train_records:
        assert r["od_index"] not in test_ods, f"train state {r['sample_id']} uses a test OD"

    audit = {
        "od_holdout": {
            "n_train_ods": len(train_ods),
            "n_test_ods": len(test_ods),
            "overlap": sorted(train_ods & test_ods),
            "verified": True,
        },
        "test_records": {
            "n": len(test_records),
            "od_indices_used": sorted({r["od_index"] for r in test_records}),
            "class_distribution": dict(Counter(r["accessibility_class"] for r in test_records)),
            "personas": sorted({r["persona_id"] for r in test_records}),
            "all_unseen": all(r["od_index"] in test_ods and r["persona_id"] in persona_test
                              for r in test_records),
        },
        "note": "student sees only the city-independent accessibility vector; "
                "OD identity is never an input feature (S8 §15)",
    }

    eval_path = Path(args.accessibility_eval)
    summary = {}
    if eval_path.exists():
        em = json.loads(eval_path.read_text(encoding="utf-8"))
        summary = {
            "models": {name: {"pt_probability_mae": m.get("fidelity", {}).get("all", {}).get("pt_probability_mae"),
                              "fvr": m.get("fvr"),
                              "sensitivity": m.get("sensitivity")}
                       for name, m in em.get("models", {}).items()},
            "deltas_s8_vs_s7w3": em.get("deltas_s8_vs_s7w3", {}),
        }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "unseen_od_audit.json").write_text(
        json.dumps({"unseen_od_audit": audit, "summary_from_accessibility_eval": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"unseen_od_audit": audit}, ensure_ascii=False, indent=2))
    print(f"\nwrote {out / 'unseen_od_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
