#!/usr/bin/env python
"""S9 backup & freeze executor (S9 = Supply-Aware Traveler Agent v2.0).

S9 retrains S8's architecture on the CORRECTED dataset (walk 1.34 m/s /
bike 4.17 m/s instead of road free-flow speeds — the S8 data error). This
script mirrors scripts/freeze_s8_release.py with S9 paths and v2 naming.

Stages:
  build         create releases/s9_supply_aware_v2/ (checkpoint, config,
                schema, normalization, accessibility, provenance, data
                manifest, metrics, reports)
  checksums     write checksums/SHA256SUMS.txt
  verify-gate   compare re-run reproduction artifacts vs the historical S9
                eval artifacts; verify S7 release integrity; verify S8 release
                remains byte-identical (it stays frozen, deprecated not deleted)

Usage:
    python scripts/freeze_s9_release.py build
    python scripts/freeze_s9_release.py checksums
    python scripts/freeze_s9_release.py verify-gate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from traveler_distillation.accessibility.accessibility_features import (  # noqa: E402
    S8_ALT_NUM,
    S8_NEW_ALT_NUM,
)
from traveler_distillation.student.features import (  # noqa: E402
    ALT_NUM,
    CONTEXT_CAT,
    CONTEXT_NUM,
    GLOBAL_CAT,
    GLOBAL_NUM,
    PERSONA_CAT,
    PERSONA_NUM,
    TRIP_CAT,
    TRIP_NUM,
)

RELEASE = ROOT / "releases" / "s9_supply_aware_v2"
RELEASE_NAME = "s9_supply_aware_v2"
S9_OUT = ROOT / "outputs" / "student_s9"
S7_RELEASE = ROOT / "releases" / "s7_w3_generic_core_v1"
S8_RELEASE = ROOT / "releases" / "s8_supply_aware_v1"
GATE_DIR = RELEASE / "reports" / "reproduction_gate"

# S9 data-fix provenance (the ONLY change vs S8)
DATA_FIX = {
    "s8_error": "walk/bike alternative travel times used road free-flow speed "
                "(length/freespeed): effective walk/bike ~29 km/h (~6x too fast for walking)",
    "s9_fix": "walk 1.34 m/s (~4.8 km/h), bike 4.17 m/s (15 km/h), car unchanged "
              "(length/freespeed); MATSim walk/bike legs teleported (no longer consume "
              "road capacity or move at link freespeed)",
    "s9_fix_measured": "effective walk speed median 3.28 km/h, bike 10.14 km/h, car 31.09 km/h "
                       "(od_km / travel time over the 114-OD pool)",
    "s8_status": "DEPRECATED (superseded due to the data error; release kept byte-identical, "
                 "never deleted) — see docs/S8_DEPRECATION.md",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required input missing: {path}")
    return path


def iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_checkpoint() -> None:
    src_ckpt = require(S9_OUT / "checkpoints" / "best.pt")
    dst_ckpt = RELEASE / "checkpoint" / "model.pt"
    copy_file(src_ckpt, dst_ckpt)
    copy_file(S9_OUT / "training_history.json", RELEASE / "checkpoint" / "training_history.json")
    copy_file(S9_OUT / "val_metrics.json", RELEASE / "checkpoint" / "val_metrics.json")

    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    n_params = sum(v.numel() for v in ck["model_state"].values())
    vm = json.loads((S9_OUT / "val_metrics.json").read_text(encoding="utf-8"))
    meta = {
        "release": RELEASE_NAME,
        "release_name": "Supply-Aware Traveler Agent v2.0",
        "model": "S9",
        "status": "FROZEN",
        "supersedes": "S8 Supply-Aware Traveler Agent v1.0 (DEPRECATED: walk/bike speed data error)",
        "data_fix": DATA_FIX,
        "checkpoint_source": str(src_ckpt.relative_to(ROOT)),
        "checkpoint_saved_at_utc": iso_mtime(src_ckpt),
        "seed": 42,
        "best_epoch": vm["best_epoch"],
        "runtime_seconds": vm["runtime_seconds"],
        "parameters": n_params,
        "params_expected": 24562,
        "architecture": ck["config"],
        "arch_version": ck["arch_version"],
        "arch_note": "architecture identical to S8 (student_s8_v1 label kept: same Case B 12-field arch)",
        "init_checkpoint": ck["init_checkpoint"],
        "init_checkpoint_note": "FROZEN S7-W3 Generic Behavioral Core v1.0 release checkpoint",
        "lambda_accessibility": ck["lambda_accessibility"],
        "lambda_mechanism": ck["lambda_mechanism"],
        "lambda_broken": ck["lambda_broken"],
        "checkpoint_keys": sorted(ck.keys()),
        "optimizer_state": "NOT PRESENT (stores model weights only)",
        "scheduler_state": "NOT PRESENT",
        "training_state": "training_history.json + val_metrics.json (copied alongside)",
        "sha256": sha256(dst_ckpt),
        "dataset": {
            "path": "data/singapore_accessibility/ (rebuilt with corrected walk/bike speeds)",
            "n_states": 336,
            "s8_legacy_archived_at": "data/singapore_accessibility_s8_legacy/",
        },
    }
    assert meta["parameters"] == meta["params_expected"], "parameter count mismatch"
    write_json(RELEASE / "checkpoint" / "model.pt.meta.json", meta)


def build_config() -> None:
    copy_file(require(ROOT / "configs" / "student_s9.yaml"), RELEASE / "config" / "student_s9.yaml")
    copy_file(require(ROOT / "configs" / "accessibility_features.yaml"),
              RELEASE / "config" / "accessibility_features.yaml")
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    write_json(RELEASE / "config" / "checkpoint_embedded_config.json",
               {"source": "checkpoint key 'config' (training-time snapshot)",
                "config": ck["config"]})
    training_s9 = {
        "selected_round": "single (lambda_accessibility=1.0; the R1/R2 ablation was settled in S8 and carries over)",
        "training": {
            "seed": 42, "batch_size": 32, "mechanism_batch_size": 8,
            "learning_rate": 0.000125, "weight_decay": 0.0001,
            "max_epochs": 120, "patience": 15,
            "replay_ratio": "2:1:1:1",
            "accessibility_per_epoch": 52, "het_pairs_per_epoch": 52,
        },
        "loss": {"lambda_accessibility": 1.0, "lambda_mechanism": 1.0, "lambda_broken": 1.0},
        "split": {"strategy": "s8_triple_holdout",
                  "persona_holdout": "s7_s3c_convention (28/6/6)",
                  "od_holdout": "true (79/14/20 disjoint)",
                  "accessibility_holdout": "walk_burden_15min"},
        "init_from": "releases/s7_w3_generic_core_v1/checkpoint/model.pt (FROZEN)",
        "schema_case": "Case B",
        "arch_version": "student_s8_v1",
        "difference_vs_s8": "dataset only (corrected walk/bike speeds); architecture/hyperparameters identical",
    }
    write_json(RELEASE / "config" / "training_s9.json", training_s9)
    import yaml

    (RELEASE / "config" / "training_s9.yaml").write_text(
        "# Effective S9 training config (frozen). Same as S8 except the corrected dataset.\n"
        + yaml.safe_dump(training_s9, sort_keys=False, allow_unicode=True),
        encoding="utf-8")


def build_schema() -> None:
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    spec = ck["feature_spec"]
    ext = ck["extractor_state"]
    input_schema = {
        "schema_version": "1.0",
        "arch_version": "student_s8_v1",
        "schema_case": "Case B (identical to S8; see S8_SCHEMA_AUDIT.md)",
        "frozen_source": "src/traveler_distillation/accessibility/accessibility_features.py (S8FeatureExtractor.encode)",
        "persona_features": {"categorical": PERSONA_CAT, "numeric": PERSONA_NUM},
        "trip_features": {"categorical": TRIP_CAT, "numeric": TRIP_NUM},
        "dynamic_context": {"categorical": CONTEXT_CAT, "numeric": CONTEXT_NUM},
        "mode_level_attributes": S8_ALT_NUM,
        "availability_representation": {
            "choice_set_encoding": "state.alternatives list order preserved",
            "mask_convention": "alt_mask float tensor: 1.0 = available, 0.0 = unavailable/padded",
            "unavailable_mode_behavior": "masked alternatives -> -inf logits in masked_softmax -> zero probability mass",
            "s9_note": "pt stays AVAILABLE even when pt_feasible=0 (FVR is learned behavior); sentinel travel_time 120 min for infeasible pt",
        },
        "encoding": {
            "categorical": "learned embedding; <UNK> fallback = index 0",
            "numeric": "z-score; S7 fields reuse FROZEN S7-W3 stats verbatim; the 6 new S8 fields fit on the S9 TRAIN split",
        },
        "vocabularies": {"mode_vocab": ext["mode_vocab"], "cat_vocab_sizes": spec["cat_vocab_sizes"]},
    }
    write_json(RELEASE / "schema" / "input_schema_s9.json", input_schema)
    output_schema = {
        "schema_version": "1.0",
        "arch_version": "student_s8_v1",
        "frozen_source": "src/traveler_distillation/student/model.py (TravelerStudent.forward)",
        "outputs": {
            "mode_probabilities": {"type": "dict[str,float] over available modes",
                                   "definition": "masked softmax over available-alternative utilities",
                                   "constraint": "sums to 1 over available modes"},
            "departure_time_shift_min": {"type": "float",
                                         "definition": "60 * tanh(departure_head(...))",
                                         "range": "(-60, +60) minutes"},
            "utilities": {"type": "float per alternative", "definition": "raw scorer logits"},
        },
    }
    write_json(RELEASE / "schema" / "output_schema_s9.json", output_schema)
    lines = [
        "# Feature order for S9 — identical to S8 (student_s8_v1), see schema/feature_order_s8.txt semantics.",
        f"# global_cat (torch.long, n={len(GLOBAL_CAT)}):",
    ]
    lines += [f"#   {i}: {name}" for i, name in enumerate(GLOBAL_CAT)]
    lines.append(f"# global_num (torch.float32, n={len(GLOBAL_NUM)}):")
    lines += [f"#   {i}: {name}" for i, name in enumerate(GLOBAL_NUM)]
    lines.append("# alt_mode_idx (torch.long, per alternative): mode_vocab index")
    lines.append(f"# alt_num (torch.float32, per alternative, n={len(S8_ALT_NUM)}):")
    lines += [f"#   {i}: {name}" for i, name in enumerate(S8_ALT_NUM)]
    lines.append("# alt_mask (torch.float32, per alternative): 1.0 = available, 0.0 = unavailable/padded")
    lines.append("")
    lines.append("global_cat := " + ",".join(GLOBAL_CAT))
    lines.append("global_num := " + ",".join(GLOBAL_NUM))
    lines.append("alt_mode_idx := <mode_vocab index per alternative, order = state.alternatives>")
    lines.append("alt_num := " + ",".join(S8_ALT_NUM) + " (per alternative, order = state.alternatives)")
    lines.append("alt_mask := <1.0/0.0 per alternative, order = state.alternatives>")
    (RELEASE / "schema" / "feature_order_s9.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(RELEASE / "schema" / "schema_diff_vs_s7.json", {
        "schema_case": "Case B",
        "arch_version": "student_s8_v1",
        "note": "IDENTICAL schema evolution as S8 (6 added city-independent alt features, "
                "alt_encoder 14->20, zero-init migration). S9 differs from S8 in DATA only: "
                "corrected walk/bike travel times.",
        "s8_diff_reference": "releases/s8_supply_aware_v1/schema/schema_diff_vs_s7.json",
    })


def build_normalization() -> None:
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    ext = ck["extractor_state"]
    old_fields = {f: {"mean": ext["num_mean"][f], "std": ext["num_std"][f]} for f in ALT_NUM}
    new_fields = {f: {"mean": ext["num_mean"][f], "std": ext["num_std"][f]} for f in S8_NEW_ALT_NUM}
    write_json(RELEASE / "normalization" / "normalization.json", {
        "source": "checkpoint key 'extractor_state'",
        "numeric_fields": {"global": GLOBAL_NUM, "alternative": S8_ALT_NUM},
        "alternative_fields": {
            "s7_inherited": {"fields": ALT_NUM, "stats_fit_on": "FROZEN S7-W3 train split (reused verbatim)", "stats": old_fields},
            "s8_new": {"fields": S8_NEW_ALT_NUM, "stats_fit_on": "S9 train split only", "stats": new_fields},
        },
        "note": "x_z = (x - mean) / std; std floored at 1e-8 in code"})
    write_json(RELEASE / "normalization" / "mode_mapping.json",
               {"source": "checkpoint key 'extractor_state'.mode_vocab",
                "mode_vocab": ext["mode_vocab"]})
    write_json(RELEASE / "normalization" / "category_mapping.json",
               {"source": "checkpoint key 'extractor_state'.cat_vocabs",
                "cat_vocabs": ext["cat_vocabs"],
                "note": "index 0 = <UNK> fallback in every vocab"})


def build_accessibility() -> None:
    write_json(RELEASE / "accessibility" / "feature_schema.json", {
        "schema_version": "2.0",
        "note": "Same 12 city-independent fields as S8 (see releases/s8_supply_aware_v1/accessibility/feature_schema.json). "
                "S9 change: WALK time = length/1.34 m/s, BIKE time = length/4.17 m/s (was length/freespeed for both). "
                "pt access/egress walk times now realistic; car unchanged.",
        "mode_speeds": {"car": "link freespeed (free-flow)", "bike_ms": 4.17, "walk_ms": 1.34},
        "matsim_execution": "walk/bike legs TELEPORTED (beeline x1.3, walk 1.39 m/s / bike 3.9 m/s); car network-simulated",
    })
    md = [
        "# S9 Transit Accessibility Feature Definitions (frozen)",
        "",
        "Identical field set and routing rules to S8 "
        "(`releases/s8_supply_aware_v1/accessibility/feature_definitions.md`).",
        "",
        "## S9 corrections (the S8 data error)",
        "",
        "| Quantity | S8 (wrong) | S9 (fixed) |",
        "|---|---|---|",
        "| walk travel time | length / link freespeed (effective ~29 km/h) | length / 1.34 m/s (~4.8 km/h) |",
        "| bike travel time | length / link freespeed (effective ~29 km/h) | length / 4.17 m/s (15 km/h) |",
        "| car travel time | length / freespeed (unchanged) | length / freespeed |",
        "| pt access/egress walk | length / freespeed (too fast) | length / 1.34 m/s |",
        "| MATSim walk/bike legs | network modes at link freespeed, consuming road capacity | teleported, walk 1.39 m/s / bike 3.9 m/s |",
        "",
        "Measured after the fix (114-OD pool): walk median 3.28 km/h, bike 10.14 km/h, car 31.09 km/h.",
        "",
        "## Routing rules — unchanged from S8",
        "",
        "- access radius 700 m; egress 700 m (extended 1.5 km); max 1 transfer;",
        "- boarding buffer max(300 s, access+60 s); connection window 180-2700 s;",
        "- departure window 1 h; coverage ratio = D_vehicle/D_OD clip [0,1];",
        "- infeasible sentinel travel time 120 min; FVR is learned behavior.",
        "",
    ]
    (RELEASE / "accessibility" / "feature_definitions.md").write_text("\n".join(md), encoding="utf-8")


def build_provenance() -> None:
    # supply provenance unchanged from S8
    for name in ("singapore_supply.json", "osm_manifest.json", "gtfs_manifest.json"):
        copy_file(require(S8_RELEASE / "provenance" / name), RELEASE / "provenance" / name)
    gen = json.loads((ROOT / "data" / "singapore_accessibility" / "generation_manifest.json").read_text(encoding="utf-8"))
    teacher_s9 = {
        "prompt_version": gen["prompt_version"],
        "prompt_module": "src/traveler_distillation/teacher/prompts_s8.py",
        "model": gen["model"],
        "endpoint": gen["endpoint"],
        "k_policy": {"base": 3, "boundary": 5},
        "k_distribution": gen["k_distribution"],
        "aggregation_method": "mean_probability (aggregate_repeats over K valid repeats)",
        "total_valid_calls": gen["stats"]["valid"],
        "stats": gen["stats"],
        "incomplete_count": 0,
        "note": "Re-labeled on the CORRECTED dataset (S9). 49 transient SSL failures retried; all 336 states aggregated.",
        "labeling_script": "scripts/label_s8_teacher.py",
        "generated_at_utc": gen["timestamp"],
        "dataset_version_note": "manifest 'dataset_version' field retains 's8_singapore_accessibility' (script constant); this manifest belongs to the S9 corrected dataset",
    }
    write_json(RELEASE / "provenance" / "teacher_s9.json", teacher_s9)
    write_json(RELEASE / "provenance" / "data_fix.json", DATA_FIX)


def build_data_manifest() -> None:
    sdir = ROOT / "data" / "singapore_accessibility"
    split = json.loads((sdir / "split_manifest.json").read_text(encoding="utf-8"))
    sanity = split["sanity"]
    dataset = {
        "dataset": "s9_singapore_accessibility (corrected walk/bike speeds)",
        "n_states": 336,
        "class_distribution": sanity["class_counts"],
        "split_counts": sanity["split_counts"],
        "persona_holdout": {"counts": split["persona_split_counts"]},
        "od_holdout": {"counts": sanity["od_pool_sizes"], "disjoint": True},
        "accessibility_holdout": split["accessibility_holdout"],
        "curve_groups": {"n_groups": sanity["n_groups"],
                         "groups_with_lt3_classes": sanity["groups_with_lt3_classes"]},
        "teacher": {"k_distribution": {"3": 89, "5": 247},
                    "valid_calls": 1502, "incomplete": 0},
        "identity_leakage": {"quote_level_hits": 0},
        "sanity": {"door_to_door_identity_max_err_min": sanity["door_to_door_identity_max_err_min"]},
        "corrected_world_stats": {
            "pt_feasible_states": 256, "pt_infeasible_states": 80,
            "pt_faster_than_walk": "234/336 (70%)",
        },
        "files": {
            "records": {"path": "data/singapore_accessibility/records.jsonl",
                        "sha256": sha256(sdir / "records.jsonl")},
            "states_with_teacher": {"path": "data/singapore_accessibility/states_with_teacher.jsonl",
                                    "sha256": sha256(sdir / "states_with_teacher.jsonl")},
            "split_manifest": {"path": "data/singapore_accessibility/split_manifest.json",
                               "sha256": sha256(sdir / "split_manifest.json")},
            "generation_manifest": {"path": "data/singapore_accessibility/generation_manifest.json",
                                    "sha256": sha256(sdir / "generation_manifest.json")},
        },
        "s8_legacy_archive": "data/singapore_accessibility_s8_legacy/ (S8 audit trail, checksums recorded in the S8 release)",
    }
    write_json(RELEASE / "data_manifest" / "s9_accessibility_dataset.json", dataset)
    copy_file(sdir / "split_manifest.json", RELEASE / "data_manifest" / "s9_split_manifest.json")
    copy_file(sdir / "generation_manifest.json", RELEASE / "data_manifest" / "s9_generation_manifest.json")


def build_metrics() -> None:
    acc = json.loads((ROOT / "outputs" / "s9_accessibility_eval" / "eval_metrics.json").read_text(encoding="utf-8"))
    reg = json.loads((ROOT / "outputs" / "s9_regression" / "eval_metrics.json").read_text(encoding="utf-8"))
    uod = json.loads((ROOT / "outputs" / "s9_unseen_od" / "unseen_od_audit.json").read_text(encoding="utf-8"))
    b0, b1, t = acc["models"]["B0_S7W3"], acc["models"]["B1_S8"], acc["models"]["Teacher"]
    w3, s9 = reg["models"]["W3_S7W3"], reg["models"]["S8"]
    final_metrics = {
        "model": "S9",
        "release": "Supply-Aware Traveler Agent v2.0",
        "base": "S7-W3 Generic Behavioral Core v1.0",
        "status": "FROZEN",
        "supersedes": "S8 (DEPRECATED: walk/bike speed data error)",
        "source": {
            "accessibility": "outputs/s9_accessibility_eval/eval_metrics.json (test-only, unseen persona x unseen OD, paired bootstrap B=2000 seed=42)",
            "regression": "outputs/s9_regression/eval_metrics.json",
            "unseen_od": "outputs/s9_unseen_od/unseen_od_audit.json",
        },
        "accessibility": {
            "fidelity_all": {"B0_S7W3": b0["fidelity"]["all"], "B1_S9": b1["fidelity"]["all"]},
            "sensitivity_delta_P_pt": {"B0_S7W3": b0["sensitivity"]["delta_P_pt_best_minus_worst"],
                                       "B1_S9": b1["sensitivity"]["delta_P_pt_best_minus_worst"],
                                       "Teacher": t["sensitivity"]["delta_P_pt_best_minus_worst"]},
            "monotonicity": {"B0_S7W3": b0["monotonicity"], "B1_S9": b1["monotonicity"],
                             "Teacher": t["monotonicity"]},
            "fvr": {"B0_S7W3": b0["fvr"], "B1_S9": b1["fvr"]},
            "teacher_pt_prob_by_class": t["pt_prob_by_class"],
        },
        "unseen_od": {"od_holdout": uod["unseen_od_audit"]["od_holdout"],
                      "test_records": uod["unseen_od_audit"]["test_records"]},
        "legacy": {"W3_S7W3": w3["legacy"], "S9": s9["legacy"]},
        "multi_axis": {"seen_joint": {"W3_S7W3": w3["seen_joint"], "S9": s9["seen_joint"]},
                       "unseen_joint": {"W3_S7W3": w3["unseen_joint"], "S9": s9["unseen_joint"]},
                       "interaction_l1_error": {"W3_S7W3": w3["interaction_l1_error"], "S9": s9["interaction_l1_error"]}},
        "mechanism": {axis: {"W3_S7W3": reg["causal_mechanism"]["models"]["W3_S7W3"][axis],
                             "S9": reg["causal_mechanism"]["models"]["S8"][axis],
                             "delta_s9_vs_s7w3": reg["causal_mechanism"]["deltas_s8_vs_s7w3"].get(axis)}
                      for axis in ("congestion", "parking_cost")},
        "deltas_s9_vs_s7w3": {
            "accessibility": acc["deltas_s8_vs_s7w3"],
            "legacy_and_joint": reg["deltas_s8_vs_s7w3"],
        },
        "regression_gate": reg["regression_gate"],
    }
    write_json(RELEASE / "metrics" / "final_metrics.json", final_metrics)

    rows = []

    def unpack(m_dict):
        return (float(m_dict["mean"]), float(m_dict["ci_low"]), float(m_dict["ci_high"]))

    def add(name, s7, s9v, teacher=None, delta=None, source="accessibility_eval"):
        row = {"metric": name, "source": source}
        if s7 is not None:
            m, lo, hi = unpack(s7); row.update(s7w3_mean=m, s7w3_ci_low=lo, s7w3_ci_high=hi)
        if s9v is not None:
            m, lo, hi = unpack(s9v); row.update(s9_mean=m, s9_ci_low=lo, s9_ci_high=hi)
        if teacher is not None:
            m, lo, hi = unpack(teacher); row.update(teacher_mean=m, teacher_ci_low=lo, teacher_ci_high=hi)
        if delta is not None:
            row.update(delta_mean=float(delta["mean"]), delta_ci_low=float(delta["ci_low"]),
                       delta_ci_high=float(delta["ci_high"]),
                       delta_ci_excludes_zero=bool(delta.get("ci_excludes_zero")))
        rows.append(row)

    add("accessibility_kl", b0["fidelity"]["all"]["kl"], b1["fidelity"]["all"]["kl"])
    add("accessibility_probability_l1", b0["fidelity"]["all"]["probability_l1"], b1["fidelity"]["all"]["probability_l1"])
    add("pt_probability_mae", b0["fidelity"]["all"]["pt_probability_mae"], b1["fidelity"]["all"]["pt_probability_mae"],
        delta=acc["deltas_s8_vs_s7w3"]["pt_probability_mae_delta"])
    add("infeasible_mean_P_pt", b0["fvr"]["mean_P_pt_infeasible"], b1["fvr"]["mean_P_pt_infeasible"],
        delta=acc["deltas_s8_vs_s7w3"]["mean_P_pt_infeasible_delta"])
    add("fvr_rate", b0["fvr"]["rate"], b1["fvr"]["rate"], delta=acc["deltas_s8_vs_s7w3"]["fvr_rate_delta"])
    add("monotonicity_pair_agreement", b0["monotonicity"]["pair_agreement"], b1["monotonicity"]["pair_agreement"],
        teacher=t["monotonicity"]["pair_agreement"], delta=acc["deltas_s8_vs_s7w3"]["pair_monotonicity_delta"])
    add("monotonicity_triplet_agreement", b0["monotonicity"]["triplet_agreement"], b1["monotonicity"]["triplet_agreement"],
        teacher=t["monotonicity"]["triplet_agreement"])
    add("accessibility_sensitivity_delta_P_pt", b0["sensitivity"]["delta_P_pt_best_minus_worst"],
        b1["sensitivity"]["delta_P_pt_best_minus_worst"], teacher=t["sensitivity"]["delta_P_pt_best_minus_worst"])
    add("legacy_mode_accuracy", w3["legacy"]["mode_accuracy"], s9["legacy"]["mode_accuracy"], source="regression_eval")
    add("legacy_kl", w3["legacy"]["kl"], s9["legacy"]["kl"], delta=reg["deltas_s8_vs_s7w3"]["legacy_kl_delta"], source="regression_eval")
    add("seen_joint_kl", w3["seen_joint"]["kl"], s9["seen_joint"]["kl"], delta=reg["deltas_s8_vs_s7w3"]["seen_joint_kl_delta"], source="regression_eval")
    add("unseen_joint_kl", w3["unseen_joint"]["kl"], s9["unseen_joint"]["kl"], delta=reg["deltas_s8_vs_s7w3"]["unseen_joint_kl_delta"], source="regression_eval")
    add("interaction_l1_error", w3["interaction_l1_error"], s9["interaction_l1_error"], source="regression_eval")
    add("mechanism_congestion_G_med",
        reg["causal_mechanism"]["models"]["W3_S7W3"]["congestion"]["G_med"],
        reg["causal_mechanism"]["models"]["S8"]["congestion"]["G_med"],
        delta=reg["causal_mechanism"]["deltas_s8_vs_s7w3"]["congestion"]["G_med_delta"], source="regression_eval")
    add("mechanism_congestion_Gap_shortcut",
        reg["causal_mechanism"]["models"]["W3_S7W3"]["congestion"]["Gap_shortcut"],
        reg["causal_mechanism"]["models"]["S8"]["congestion"]["Gap_shortcut"], source="regression_eval")
    add("mechanism_parking_cost_G_med",
        reg["causal_mechanism"]["models"]["W3_S7W3"]["parking_cost"]["G_med"],
        reg["causal_mechanism"]["models"]["S8"]["parking_cost"]["G_med"],
        delta=reg["causal_mechanism"]["deltas_s8_vs_s7w3"]["parking_cost"]["G_med_delta"], source="regression_eval")

    csv_path = RELEASE / "metrics" / "comparison_s7_vs_s9.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "source", "s7w3_mean", "s7w3_ci_low", "s7w3_ci_high",
                  "s9_mean", "s9_ci_low", "s9_ci_high", "teacher_mean", "teacher_ci_low",
                  "teacher_ci_high", "delta_mean", "delta_ci_low", "delta_ci_high",
                  "delta_ci_excludes_zero"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def build_reports() -> None:
    copy_file(require(ROOT / "reports" / "S8_SCHEMA_AUDIT.md"), RELEASE / "reports" / "S8_SCHEMA_AUDIT.md")
    copy_file(require(S8_RELEASE / "reports" / "EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md"),
              RELEASE / "reports" / "EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY_DEPRECATED.md")


def build_code_manifest() -> None:
    cm = RELEASE / "code_manifest"
    for name in ("git_commit.txt", "git_status.txt", "environment.txt", "uncommitted.patch"):
        if not (cm / name).exists():
            raise FileNotFoundError(f"code_manifest/{name} missing — capture git/environment facts first")
    write_json(cm / "code_manifest_provenance.json", {
        "note": "S9 training/eval code = S8 commit 65e505f PLUS the S9 data-fix edits "
                "(gtfs_accessibility.py mode speeds, adapter.py teleported walk/bike) — see git log.",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": "3.12.13", "torch": "2.13.0+cpu",
                        "java": "OpenJDK 25.0.4", "matsim": "MATSim 2026.0"},
        "data_fix": DATA_FIX,
    })


def build_checksums() -> None:
    cks = RELEASE / "checksums"
    cks.mkdir(parents=True, exist_ok=True)
    targets = sorted(p for p in RELEASE.rglob("*") if p.is_file() and p.parent.name != "checksums")
    lines = ["# SHA256 checksums of the frozen S9 release.",
             "# Any change to a checksum after freeze means the version has been modified.",
             f"# Generated at {datetime.now(timezone.utc).isoformat()}"]
    for p in targets:
        rel = p.relative_to(RELEASE).as_posix()
        lines.append(f"{sha256(p)}  {rel}")
    (cks / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checksummed {len(targets)} files -> {cks / 'SHA256SUMS.txt'}")


def _load_gate_json(name: str):
    p = GATE_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def verify_gate(tol: float = 1e-4) -> int:
    acc_hist = json.loads((ROOT / "outputs" / "s9_accessibility_eval" / "eval_metrics.json").read_text(encoding="utf-8"))
    reg_hist = json.loads((ROOT / "outputs" / "s9_regression" / "eval_metrics.json").read_text(encoding="utf-8"))
    uod_hist = json.loads((ROOT / "outputs" / "s9_unseen_od" / "unseen_od_audit.json").read_text(encoding="utf-8"))
    acc_gate = _load_gate_json("accessibility_eval/eval_metrics.json")
    reg_gate = _load_gate_json("regression_eval/eval_metrics.json")
    uod_gate = _load_gate_json("unseen_od/unseen_od_audit.json")
    smoke = _load_gate_json("singapore_smoke_result.json")
    if acc_gate is None or reg_gate is None or uod_gate is None:
        print("MISSING reproduction artifacts (accessibility/regression/unseen_od)")
        return 1
    b1g = acc_gate["models"]["B1_S8"]
    b1h = acc_hist["models"]["B1_S8"]
    s9g = reg_gate["models"]["S8"]
    s9h = reg_hist["models"]["S8"]
    gates = [
        ("PT prob MAE (S9)", b1h["fidelity"]["all"]["pt_probability_mae"]["mean"], b1g["fidelity"]["all"]["pt_probability_mae"]["mean"]),
        ("mean P(PT|infeasible) (S9)", b1h["fvr"]["mean_P_pt_infeasible"]["mean"], b1g["fvr"]["mean_P_pt_infeasible"]["mean"]),
        ("FVR rate (S9)", b1h["fvr"]["rate"]["mean"], b1g["fvr"]["rate"]["mean"]),
        ("pair monotonicity (S9)", b1h["monotonicity"]["pair_agreement"]["mean"], b1g["monotonicity"]["pair_agreement"]["mean"]),
        ("sensitivity dP_PT (S9)", b1h["sensitivity"]["delta_P_pt_best_minus_worst"]["mean"], b1g["sensitivity"]["delta_P_pt_best_minus_worst"]["mean"]),
        ("legacy accuracy (S9)", s9h["legacy"]["mode_accuracy"]["mean"], s9g["legacy"]["mode_accuracy"]["mean"]),
        ("legacy KL (S9)", s9h["legacy"]["kl"]["mean"], s9g["legacy"]["kl"]["mean"]),
        ("seen joint KL (S9)", s9h["seen_joint"]["kl"]["mean"], s9g["seen_joint"]["kl"]["mean"]),
        ("unseen joint KL (S9)", s9h["unseen_joint"]["kl"]["mean"], s9g["unseen_joint"]["kl"]["mean"]),
        ("mechanism parking G_med (S9)", reg_hist["causal_mechanism"]["models"]["S8"]["parking_cost"]["G_med"]["mean"],
         reg_gate["causal_mechanism"]["models"]["S8"]["parking_cost"]["G_med"]["mean"]),
        ("mechanism congestion Gap_shortcut (S9)", reg_hist["causal_mechanism"]["models"]["S8"]["congestion"]["Gap_shortcut"]["mean"],
         reg_gate["causal_mechanism"]["models"]["S8"]["congestion"]["Gap_shortcut"]["mean"]),
        ("unseen OD n_test_states", float(uod_hist["unseen_od_audit"]["test_records"]["n"]),
         float(uod_gate["unseen_od_audit"]["test_records"]["n"])),
    ]
    ok = True
    print(f"{'gate':42s} {'historical':>12s} {'re-run':>12s} {'|delta|':>10s}  verdict")
    for name, h, g in gates:
        d = abs(h - g)
        passed = d <= tol
        ok &= passed
        print(f"{name:42s} {h:12.6f} {g:12.6f} {d:10.6f}  {'PASS' if passed else 'FAIL'}")
    print()
    if smoke is None:
        print("Singapore smoke artifact MISSING")
        ok = False
    else:
        print(f"Singapore smoke: schema_match={smoke.get('schema_check', {}).get('schema_match')}, "
              f"gate_pass={smoke.get('gate_pass')}, matsim_exit={smoke.get('matsim_exit_code')}")
        if not smoke.get("gate_pass"):
            ok = False
    for rel, name in ((S7_RELEASE, "S7-W3"), (S8_RELEASE, "S8 (deprecated, kept intact)")):
        ok &= _verify_release_intact(rel, name)
    print()
    print("REPRODUCTION GATE:", "PASS — frozen S9 checkpoint reproduces the historical values"
          if ok else "FAIL — pause the freeze and locate the cause")
    return 0 if ok else 1


def _verify_release_intact(release: Path, name: str) -> bool:
    sums = (release / "checksums" / "SHA256SUMS.txt").read_text(encoding="utf-8")
    ok = True
    n = 0
    for line in sums.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, rel = line.split(None, 1)
        p = release / rel
        if not p.exists() or sha256(p) != h:
            print(f"{name} release file MISSING/MODIFIED: {rel}")
            ok = False
        n += 1
    print(f"{name} release integrity: {n} files, {'ALL UNCHANGED' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "checksums", "verify-gate"])
    args = ap.parse_args()
    if args.stage == "build":
        build_checkpoint()
        build_config()
        build_schema()
        build_normalization()
        build_accessibility()
        build_provenance()
        build_data_manifest()
        build_metrics()
        build_reports()
        build_code_manifest()
        print(f"release built at {RELEASE}")
    elif args.stage == "checksums":
        build_checksums()
    elif args.stage == "verify-gate":
        return verify_gate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
