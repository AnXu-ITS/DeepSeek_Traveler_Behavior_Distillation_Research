#!/usr/bin/env python
"""S8 backup & freeze executor (S8_BACKUP_FREEZE_INSTRUCTIONS.md).

Mechanically implements the instruction steps:

  build         create releases/s8_supply_aware_v1/ with checkpoint, config,
                schema (Case B diff), normalization, accessibility definitions,
                Singapore supply provenance, teacher provenance, data manifests,
                metrics snapshot, reports (steps 1-7, 9-11)
  checksums     write checksums/SHA256SUMS.txt over every frozen file (step 13)
  verify-gate   compare the re-run reproduction gate artifacts (step 8) against
                the historical S8 eval artifacts, and re-verify the frozen S7-W3
                release is unmodified (freeze gate items)

Steps 12/14/15 (FINAL_S8_FREEZE.md, README.md, git commit/tag, read-only
protection) are hand-written / PowerShell-driven and must exist before
`checksums` runs.

Usage:
    python scripts/freeze_s8_release.py build
    python scripts/freeze_s8_release.py checksums
    python scripts/freeze_s8_release.py verify-gate
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

RELEASE = ROOT / "releases" / "s8_supply_aware_v1"
RELEASE_NAME = "s8_supply_aware_v1"
S8_OUT = ROOT / "outputs" / "student_s8"
S7_RELEASE = ROOT / "releases" / "s7_w3_generic_core_v1"
GATE_DIR = RELEASE / "reports" / "reproduction_gate"


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


# ----------------------------------------------------------------------------
# Step 1: checkpoint
# ----------------------------------------------------------------------------
def build_checkpoint() -> None:
    src_ckpt = require(S8_OUT / "checkpoints" / "best.pt")
    dst_ckpt = RELEASE / "checkpoint" / "model.pt"
    copy_file(src_ckpt, dst_ckpt)
    copy_file(S8_OUT / "training_history.json", RELEASE / "checkpoint" / "training_history.json")
    copy_file(S8_OUT / "val_metrics.json", RELEASE / "checkpoint" / "val_metrics.json")

    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    n_params = sum(v.numel() for v in ck["model_state"].values())
    meta = {
        "release": RELEASE_NAME,
        "release_name": "Supply-Aware Traveler Agent v1.0",
        "model": "S8",
        "status": "FROZEN",
        "checkpoint_source": str(src_ckpt.relative_to(ROOT)),
        "checkpoint_saved_at_utc": iso_mtime(src_ckpt),
        "seed": 42,
        "best_epoch": 26,
        "early_stop_epoch": 41,
        "runtime_seconds": json.loads((S8_OUT / "val_metrics.json").read_text(encoding="utf-8"))["runtime_seconds"],
        "parameters": n_params,
        "params_expected": 24562,
        "architecture": ck["config"],
        "arch_version": ck["arch_version"],
        "init_checkpoint": ck["init_checkpoint"],
        "init_checkpoint_note": "FROZEN S7-W3 Generic Behavioral Core v1.0 release checkpoint",
        "lambda_accessibility": ck["lambda_accessibility"],
        "lambda_mechanism": ck["lambda_mechanism"],
        "lambda_broken": ck["lambda_broken"],
        "checkpoint_keys": sorted(ck.keys()),
        "optimizer_state": "NOT PRESENT (S8 best.pt stores model weights only)",
        "scheduler_state": "NOT PRESENT",
        "training_state": "training_history.json + val_metrics.json (copied alongside)",
        "sha256": sha256(dst_ckpt),
        "ablation_rounds": {
            "R1": {"lambda_accessibility": 0.0, "best_epoch": 19, "output": "outputs/student_s8_r1_lam0/",
                   "note": "baseline fine-tune, lower monotonicity (pair 0.600 / triplet 0.217) — not selected"},
            "R2": {"lambda_accessibility": 1.0, "best_epoch": 26, "output": "outputs/student_s8/",
                   "note": "SELECTED: +accessibility response loss improves monotonicity and infeasible probability"},
        },
    }
    assert meta["parameters"] == meta["params_expected"], "parameter count mismatch"
    write_json(RELEASE / "checkpoint" / "model.pt.meta.json", meta)


# ----------------------------------------------------------------------------
# Step 2: config
# ----------------------------------------------------------------------------
def build_config() -> None:
    copy_file(require(ROOT / "configs" / "student_s8.yaml"), RELEASE / "config" / "student_s8.yaml")
    copy_file(require(ROOT / "configs" / "accessibility_features.yaml"),
              RELEASE / "config" / "accessibility_features.yaml")
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    write_json(RELEASE / "config" / "checkpoint_embedded_config.json",
               {"source": "checkpoint key 'config' (training-time snapshot)",
                "config": ck["config"]})

    # effective training config (configs/student_s8.yaml is the SELECTED R2 config)
    training_s8 = {
        "selected_round": "R2",
        "lambda_accessibility": 1.0,
        "rounds": [
            {"round": "R1", "lambda_accessibility": 0.0, "best_epoch": 19,
             "output": "outputs/student_s8_r1_lam0/",
             "note": "accessibility KL-only fine-tune; rejected: within-curve sensitivity/monotonicity below teacher"},
            {"round": "R2", "lambda_accessibility": 1.0, "best_epoch": 26,
             "early_stop_epoch": 41, "output": "outputs/student_s8/",
             "note": "SELECTED final S8 (accessibility response loss L_accessibility added)"},
        ],
        "training": {
            "seed": 42,
            "batch_size": 32,
            "mechanism_batch_size": 8,
            "learning_rate": 0.000125,
            "weight_decay": 0.0001,
            "max_epochs": 120,
            "patience": 15,
            "early_stopping": "selection guards + patience 15 (R2 stopped at epoch 41)",
            "replay_ratio": "2:1:1:1",
            "replay_note": "legacy : joint : mechanism : accessibility",
            "accessibility_per_epoch": 52,
            "het_pairs_per_epoch": 52,
        },
        "loss": {
            "lambda_action": 1.0, "lambda_distribution": 1.0, "lambda_departure": 1.0,
            "lambda_elasticity": 0.0, "lambda_direction": 1.0, "lambda_magnitude": 1.0,
            "direction_margin": 0.0, "lambda_heterogeneity": 1.0, "huber_delta": 1.0,
            "lambda_mechanism": 1.0, "lambda_broken": 1.0, "lambda_accessibility": 1.0,
            "loss_note": "L_accessibility = mean |E_T^acc - E_S^acc| over curve-group pairs (round 2 only)",
        },
        "split": {
            "strategy": "s8_triple_holdout",
            "persona_holdout": "s7_s3c_convention (28/6/6, test personas never in train)",
            "od_holdout": "true (train/val/test ODs disjoint 79/16/19)",
            "accessibility_holdout": "walk_burden_15min (access+egress >= 15 min only in test)",
        },
        "init_from": "releases/s7_w3_generic_core_v1/checkpoint/model.pt (FROZEN, byte-copied shared weights)",
        "schema_case": "Case B",
        "arch_version": "student_s8_v1",
    }
    write_json(RELEASE / "config" / "training_s8.json", training_s8)
    # YAML mirror of the effective training config
    import yaml

    (RELEASE / "config" / "training_s8.yaml").write_text(
        "# Effective S8 training config (frozen). Selected round = R2, lambda_accessibility = 1.0.\n"
        "# Full provenance: config/student_s8.yaml (verbatim) + checkpoint 'config' key.\n"
        + yaml.safe_dump(training_s8, sort_keys=False, allow_unicode=True),
        encoding="utf-8")


# ----------------------------------------------------------------------------
# Step 3: schema (Case B diff)
# ----------------------------------------------------------------------------
def build_schema() -> None:
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    spec = ck["feature_spec"]
    ext = ck["extractor_state"]

    input_schema = {
        "schema_version": "1.0",
        "arch_version": "student_s8_v1",
        "schema_case": "Case B (new input dimension: 6 added alternative numeric fields)",
        "frozen_source": "src/traveler_distillation/accessibility/accessibility_features.py (S8FeatureExtractor.encode)",
        "persona_features": {"categorical": PERSONA_CAT, "numeric": PERSONA_NUM},
        "trip_features": {"categorical": TRIP_CAT, "numeric": TRIP_NUM},
        "dynamic_context": {"categorical": CONTEXT_CAT, "numeric": CONTEXT_NUM},
        "mode_level_attributes": S8_ALT_NUM,
        "availability_representation": {
            "choice_set_encoding": "state.alternatives list order preserved; each mode -> mode_vocab index",
            "mask_convention": "alt_mask float tensor: 1.0 = available, 0.0 = unavailable/padded",
            "unavailable_mode_behavior": "masked alternatives receive -inf logits in masked_softmax -> zero probability mass",
            "s8_note": "pt stays AVAILABLE even when pt_feasible=0 (feasibility is a LEARNED behavior, FVR metric); sentinel travel_time 120 min for infeasible pt",
        },
        "encoding": {
            "categorical": "learned embedding; vocab index per field; <UNK> fallback = index 0",
            "numeric": "z-score; the 6 S7 fields reuse the FROZEN S7-W3 stats verbatim, the 6 new S8 fields are fit on the S8 TRAIN split only",
        },
        "tensor_layout": {
            "global_cat": {"dtype": "long", "n": len(GLOBAL_CAT), "order": GLOBAL_CAT},
            "global_num": {"dtype": "float32", "n": len(GLOBAL_NUM), "order": GLOBAL_NUM},
            "alt_mode_idx": {"dtype": "long", "per_alternative": "mode_vocab index"},
            "alt_num": {"dtype": "float32", "n_per_alternative": len(S8_ALT_NUM), "order": S8_ALT_NUM},
            "alt_mask": {"dtype": "float32", "per_alternative": "1.0/0.0 availability"},
        },
        "vocabularies": {"mode_vocab": ext["mode_vocab"], "cat_vocab_sizes": spec["cat_vocab_sizes"]},
    }
    write_json(RELEASE / "schema" / "input_schema_s8.json", input_schema)

    output_schema = {
        "schema_version": "1.0",
        "arch_version": "student_s8_v1",
        "frozen_source": "src/traveler_distillation/student/model.py (TravelerStudent.forward)",
        "outputs": {
            "mode_probabilities": {
                "type": "dict[str,float] over available modes",
                "definition": "masked softmax over available-alternative utilities",
                "constraint": "sums to 1 over available modes; unavailable modes carry 0",
            },
            "departure_time_shift_min": {
                "type": "float",
                "definition": "60 * tanh(departure_head([global_emb, pooled_alt_emb]))",
                "range": "(-60, +60) minutes by construction",
            },
            "utilities": {"type": "float per alternative", "definition": "raw scorer logits"},
        },
        "heads": {
            "mode_head": "scorer([global_emb, alt_emb]) -> 1 logit per alternative",
            "departure_head": "departure_head([global_emb, masked-mean-pooled alt_emb]) -> 1 logit",
        },
    }
    write_json(RELEASE / "schema" / "output_schema_s8.json", output_schema)

    lines = [
        "# Feature order for S8 (student_s8_v1) — MUST match the actual tensor input order.",
        "# Generated from src/traveler_distillation/accessibility/accessibility_features.py S8_ALT_NUM.",
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
    lines.append("# Canonical column order (one tensor per line, columns within a tensor in listed order):")
    lines.append("global_cat := " + ",".join(GLOBAL_CAT))
    lines.append("global_num := " + ",".join(GLOBAL_NUM))
    lines.append("alt_mode_idx := <mode_vocab index per alternative, order = state.alternatives>")
    lines.append("alt_num := " + ",".join(S8_ALT_NUM) + " (per alternative, order = state.alternatives)")
    lines.append("alt_mask := <1.0/0.0 per alternative, order = state.alternatives>")
    (RELEASE / "schema" / "feature_order_s8.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    schema_diff = {
        "schema_case": "Case B",
        "base": "S7-W3 Generic Behavioral Core v1.0 (releases/s7_w3_generic_core_v1/schema/)",
        "arch_version": "student_s8_v1",
        "inherited_features": {
            "alternative_numeric": ALT_NUM,
            "note": "reused verbatim with the FROZEN S7-W3 z-score stats",
        },
        "added_features": {
            "alternative_numeric": S8_NEW_ALT_NUM,
            "meaning": "city-independent real-supply transit accessibility attributes (S8 instructions §6-7)",
        },
        "added_dimension": {
            "alt_encoder.0.weight": "32x14 -> 32x20 (+6 columns, zero-initialized)",
            "n_alt_num": "6 -> 12",
            "parameters": "24,370 -> 24,562 (+192)",
        },
        "added_encoder": "none (S8 reuses alt_encoder; only the first layer input is extended)",
        "weight_migration": {
            "global_encoder/scorer/departure_head/embeddings/alt_encoder.1-3": "byte-copied verbatim from S7-W3",
            "alt_encoder.0.weight first 14 columns": "byte-copied verbatim from S7-W3",
            "alt_encoder.0.weight new 6 columns": "zero-initialized => S8-at-init reproduces S7-W3 outputs exactly (initialization invariant verified 4/4, scripts/audit_s8_schema.py)",
        },
        "normalization_change": {
            "old_6_fields": "UNCHANGED (frozen S7-W3 mean/std reused verbatim)",
            "new_6_fields": "mean/std fit on the S8 TRAIN split only (238*4 alternative rows)",
            "frozen_s7_release": "NOT modified (S7 release remains byte-identical)",
        },
        "documented_in": "reports/S8_SCHEMA_AUDIT.md",
    }
    write_json(RELEASE / "schema" / "schema_diff_vs_s7.json", schema_diff)


# ----------------------------------------------------------------------------
# Step 3b: normalization
# ----------------------------------------------------------------------------
def build_normalization() -> None:
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    ext = ck["extractor_state"]
    old_fields = {f: {"mean": ext["num_mean"][f], "std": ext["num_std"][f]} for f in ALT_NUM}
    new_fields = {f: {"mean": ext["num_mean"][f], "std": ext["num_std"][f]} for f in S8_NEW_ALT_NUM}
    write_json(RELEASE / "normalization" / "normalization.json", {
        "source": "checkpoint key 'extractor_state'",
        "numeric_fields": {"global": GLOBAL_NUM, "alternative": S8_ALT_NUM},
        "global_mean_std": {f: {"mean": ext["num_mean"][f], "std": ext["num_std"][f]}
                            for f in GLOBAL_NUM},
        "alternative_fields": {
            "s7_inherited": {"fields": ALT_NUM, "stats_fit_on": "FROZEN S7-W3 train split (reused verbatim, NOT refit)", "stats": old_fields},
            "s8_new": {"fields": S8_NEW_ALT_NUM, "stats_fit_on": "S8 train split only", "stats": new_fields},
        },
        "note": "x_z = (x - mean) / std; std floored at 1e-8 in code"})
    write_json(RELEASE / "normalization" / "mode_mapping.json",
               {"source": "checkpoint key 'extractor_state'.mode_vocab",
                "mode_vocab": ext["mode_vocab"]})
    write_json(RELEASE / "normalization" / "category_mapping.json",
               {"source": "checkpoint key 'extractor_state'.cat_vocabs",
                "cat_vocabs": ext["cat_vocabs"],
                "note": "index 0 = <UNK> fallback in every vocab"})


# ----------------------------------------------------------------------------
# Step 4: transit accessibility definitions
# ----------------------------------------------------------------------------
def build_accessibility() -> None:
    feature_schema = {
        "schema_version": "1.0",
        "source": "configs/accessibility_features.yaml (frozen) + data/singapore_accessibility/DATA_README.md",
        "fields": [
            {"name": "pt_feasible", "unit": "binary 0/1", "range": "[0,1]",
             "computation": "1.0 = a feasible itinerary exists in the departure window under the routing rules; 0.0 otherwise",
             "normalization": "z-score (S8 train fit)", "source": "plan_accessibility (GTFS + OSM network)"},
            {"name": "pt_access_time_min", "alias": "access_time_min (S7 field reused)",
             "unit": "minutes", "range": ">= 0",
             "computation": "network walk time origin -> first boarding stop (access radius 700 m)",
             "normalization": "z-score (FROZEN S7-W3 stats)", "source": "OSM network walk graph"},
            {"name": "pt_egress_time_min", "alias": "egress_time_min", "unit": "minutes", "range": ">= 0",
             "computation": "network walk time last alighting stop -> destination (egress radius 700 m, extended 1.5 km allowed)",
             "normalization": "z-score (S8 train fit)", "source": "OSM network walk graph"},
            {"name": "pt_wait_time_min", "alias": "wait_time_min", "unit": "minutes", "range": ">= 0",
             "computation": "first boarding departure - trip departure - access walk",
             "normalization": "z-score (S8 train fit)", "source": "GTFS departure times"},
            {"name": "pt_in_vehicle_time_min", "alias": "in_vehicle_time_min", "unit": "minutes", "range": ">= 0",
             "computation": "total time aboard transit vehicles (direct or 1-transfer itinerary)",
             "normalization": "z-score (S8 train fit)", "source": "GTFS stop times"},
            {"name": "pt_transfer_count", "alias": "transfers (S7 field reused)", "unit": "count", "range": "0 or 1 (planner cap)",
             "computation": "vehicle-to-vehicle transfer count of the itinerary",
             "normalization": "z-score (FROZEN S7-W3 stats)", "source": "GTFS itinerary search"},
            {"name": "pt_transfer_time_min", "alias": "transfer_time_min", "unit": "minutes", "range": ">= 0",
             "computation": "alight vehicle -> board next (walk + wait at transfer; connection window 180-2700 s)",
             "normalization": "z-score (S8 train fit)", "source": "GTFS itinerary search"},
            {"name": "pt_total_door_to_door_time_min", "alias": "travel_time_min of the pt alternative",
             "unit": "minutes", "range": "> 0",
             "computation": "T_PT = T_access + T_wait + T_invehicle + T_transfer + T_egress (infeasible sentinel = 120 min)",
             "normalization": "z-score (FROZEN S7-W3 stats)", "source": "plan_accessibility"},
            {"name": "pt_coverage_ratio", "alias": "coverage_ratio", "unit": "ratio", "range": "[0,1]",
             "computation": "R_coverage = D_PT,vehicle / D_OD (straight-line in-vehicle distance / OD distance), clipped [0,1]",
             "normalization": "z-score (S8 train fit)", "source": "GTFS stop geometry"},
            {"name": "pt_generalized_cost", "unit": "n/a",
             "computation": "NOT DEFINED — S8 §7.10: no invented value-of-time parameters; raw features only",
             "normalization": "n/a", "source": "n/a", "defined": False},
        ],
        "routing_rules": {
            "access_radius_m": 700, "egress_radius_m": 700, "extended_egress_m": 1500,
            "boarding_buffer_s": "max(300, access_walk_s + 60)", "max_transfers": 1,
            "connection_window_s": [180, 2700], "departure_window_s": 3600,
            "walk_speed_ms": 1.2, "walk_time": "network-derived length / freespeed",
        },
        "infeasible_convention": "pt stays in the choice set with pt_feasible=0 and sentinel travel_time 120 min (FVR is learned behavior, not a mask)",
        "identity": "ALL fields are city-independent numeric attributes; no stop_id/route_id/node id/place name ever enters the student/teacher input",
    }
    write_json(RELEASE / "accessibility" / "feature_schema.json", feature_schema)

    md = [
        "# S8 Transit Accessibility Feature Definitions (frozen)",
        "",
        "Source of truth: `configs/accessibility_features.yaml`, "
        "`data/singapore_accessibility/DATA_README.md`, "
        "`src/traveler_distillation/accessibility/gtfs_accessibility.py` "
        "(planner `plan_accessibility`), `accessibility/feature_schema.json`.",
        "",
        "## Field definitions",
        "",
        "| Field | Unit | Range | Computation | Normalization | Source |",
        "|---|---|---|---|---|---|",
    ]
    for f in feature_schema["fields"]:
        md.append(
            f"| {f['name']} ({f.get('alias', '')}) | {f['unit']} | {f.get('range', '')} | "
            f"{f['computation']} | {f['normalization']} | {f['source']} |"
        )
    md += [
        "",
        "## Routing rules (frozen for S8 features)",
        "",
        "| Rule | Value |",
        "|---|---|",
        "| access radius | boarding stop within 700 m of origin (no wider fallback) |",
        "| egress radius | alighting stop within 700 m of destination; extended egress <= 1.5 km |",
        "| max transfers | 1 vehicle-to-vehicle |",
        "| boarding buffer | access-aware: max(300 s, access_walk + 60 s) |",
        "| connection window | 180-2700 s between alight and next board |",
        "| departure window | first boarding within 1 h of trip departure |",
        "| walk time | network-derived length / freespeed (1.2 m/s reference) |",
        "",
        "## Feasibility convention",
        "",
        "`pt_feasible = 0` states keep pt in the choice set with a documented sentinel "
        "travel time (120 min) — feasibility is a LEARNED behavior (FVR metric), "
        "not an availability mask.",
        "",
        "## City-independence guarantee",
        "",
        "All fields are numeric accessibility attributes with no location identity. "
        "`pt_generalized_cost` is deliberately NOT defined (no invented value-of-time).",
        "",
    ]
    (RELEASE / "accessibility" / "feature_definitions.md").write_text("\n".join(md), encoding="utf-8")


# ----------------------------------------------------------------------------
# Step 5: Singapore supply provenance
# ----------------------------------------------------------------------------
def build_provenance() -> None:
    singapore_supply = {
        "study_area": "Tampines + Pasir Ris, Singapore",
        "bbox": {"lat_min": 1.330, "lat_max": 1.400, "lon_min": 103.900, "lon_max": 104.015},
        "crs": "UTM 48N (src/traveler_distillation/singapore/projection.py utm48n)",
        "network": "data/singapore/transit/network_with_transit.xml (OSM-derived; 191,617 links, 100,867 nodes, 3,520.58 km)",
        "stops": "859 served stops (full-day supply 05:00-23:00, 20,966 trips; 20,382 bus + 904 rail)",
        "od_candidate_nodes": "50,241 (car-accessible ∩ walk largest component)",
        "matsim": "MATSim 2026.0, qsim; Phase C frozen settings N*=10,000, flowCapacityFactor=storageCapacityFactor=0.3",
        "pt_planner": "direct + 1 transfer (separate legs) + extended 1.5 km walk egress (MATSimAdapter._plan_pt / plan_accessibility)",
        "honesty_note": "S8 uses REAL Singapore transport supply, but the Teacher behavior labels are NOT observed Singapore traveler behavior (synthetic persona LLM inferences). The paper must not claim 'real Singapore traveler behavior' or 'exact reproduction of Singapore demand'.",
        "phase_c_supply": "frozen Phase B.5 version (network_with_transit.xml + calibrated capacity factors; green-ratio sensitivity archive in outputs/singapore_phase_b5/calibration/)",
    }
    write_json(RELEASE / "provenance" / "singapore_supply.json", singapore_supply)

    osm_manifest = {
        "snapshot_date": "2026-08-25",
        "source": "Overpass API bbox extraction",
        "query": "data/singapore/osm/query.txt ([out:xml][timeout:600];(way[\"highway\"](1.330,103.900,1.400,104.015);>;);out body;)",
        "study_area": "Tampines + Pasir Ris (lat 1.330-1.400 x lon 103.900-104.015)",
        "clipping_boundary": "bbox lat 1.330-1.400 x lon 103.900-104.015 (+ margin 0.004 deg for GTFS cropping)",
        "crs": "UTM 48N",
        "conversion_version": "src/traveler_distillation/singapore/osm_network.py (build_network)",
        "license": "ODbL",
        "file": "data/singapore/osm/tampines_pasir_ris.osm",
        "file_sha256": sha256(ROOT / "data" / "singapore" / "osm" / "tampines_pasir_ris.osm"),
        "network_stats": json.loads((ROOT / "data" / "singapore" / "osm" / "network_stats.json").read_text(encoding="utf-8")),
        "traffic_signals": "659 highway=traffic_signals nodes (data/singapore/osm/traffic_signal_nodes.txt)",
        "known_approximations": [
            "MRT rail-on-road: no railway in OSM conversion; MRT trips routed on road graph (transportMode=rail), explicitly downgraded",
            "one-way transit reverse links: busr_/ai_ artificial links for bus/rail against one-way streets (pt2matsim-style)",
            "no junction delay model (capacity assigned by highway class saturation flow)",
        ],
    }
    write_json(RELEASE / "provenance" / "osm_manifest.json", osm_manifest)

    gtfs_manifest = {
        "source": "community-constructed singapore-gtfs-2025 (NOT official LTA DataMall)",
        "publisher": "Singapore GTFS, https://github.com/thecrapone/singapore-gtfs-2025",
        "version": "feed_version 1.0",
        "feed_dates": "2025-01-01 - 2030-12-31",
        "archived": "2026-08-24 (user-provided snapshot)",
        "file": "data/singapore/gtfs/raw/singapore-gtfs.zip",
        "file_size_bytes": 335668255,
        "checksum_sha256": "1fcc6f5f2766d34aab08d40adabbb094f9124feba6e39aba671e5f532d4eb3cd",
        "checksum_file": "data/singapore/gtfs/raw/checksum.sha256",
        "content": {"agencies": 6, "routes": 603, "bus_routes": 593, "mrt_routes": 9,
                    "stops": 5376, "trips": 230915, "stop_times": 8169065,
                    "calendar_patterns": 4},
        "cropping": {
            "service_id": "WD", "time_window": "05:00-23:00 (18000-82800 s)",
            "bbox": "lat 1.330-1.400 x lon 103.900-104.015 (margin 0.004 deg)",
            "supply_cropping": "longest in-region consecutive stop segment per trip",
            "trips_kept": 21286, "stops_kept": 859,
            "mode_counts": {"bus": 20382, "rail": 904},
        },
        "routing_assumptions": {
            "transfer_rule": "max 1 vehicle-to-vehicle transfer; connection window 180-2700 s",
            "walk_radius": "access 700 m / egress 700 m (extended 1.5 km)",
            "service_window": "full-day 05:00-23:00; departure window 1 h",
            "boarding_buffer_s": "max(300, access_walk_s + 60)",
        },
        "pt_validity_version": "Phase B.5 PT validity (outputs/singapore_phase_b5/pt_validity_v3/) — frozen validated PT pipeline",
        "honesty_note": "MRT schedules are frequency-based / synthetic; bus travel times include estimates. The paper must not describe this ZIP as an 'official measured GTFS timetable'.",
    }
    write_json(RELEASE / "provenance" / "gtfs_manifest.json", gtfs_manifest)

    gen = json.loads((ROOT / "data" / "singapore_accessibility" / "generation_manifest.json").read_text(encoding="utf-8"))
    teacher_s8 = {
        "prompt_version": gen["prompt_version"],
        "prompt_module": "src/traveler_distillation/teacher/prompts_s8.py",
        "model": gen["model"],
        "endpoint": gen["endpoint"],
        "k_policy": gen["k_policy"],
        "k_distribution": gen["k_distribution"],
        "aggregation_method": "mean_probability (aggregate_repeats over K valid repeats)",
        "output_schema": "AggregatedTeacherTarget (mode_probabilities over available modes + departure shift)",
        "clip_departure": gen.get("clip_departure"),
        "total_calls": gen["n_calls_planned"],
        "stats": gen["stats"],
        "incomplete_count": 0,
        "note": "1,468 total attempts across both labeling rounds (first-round process contributed ~38 attempts before the max_tokens fix); final 0 incomplete",
        "cache_bypass": "official DeepSeek endpoint has no gateway cache; cache_bypass=False",
        "labeling_script": "scripts/label_s8_teacher.py",
        "generated_at_utc": gen["timestamp"],
        "budget_note": "K=3 base + K=5 boundary samples (user-approved budget)",
    }
    write_json(RELEASE / "provenance" / "teacher_s8.json", teacher_s8)


# ----------------------------------------------------------------------------
# Step 6: dataset manifest
# ----------------------------------------------------------------------------
def build_data_manifest() -> None:
    sdir = ROOT / "data" / "singapore_accessibility"
    split = json.loads((sdir / "split_manifest.json").read_text(encoding="utf-8"))
    sanity = split["sanity"]
    dataset = {
        "dataset": "s8_singapore_accessibility",
        "dataset_version": "s8_singapore_accessibility",
        "n_states": 338,
        "class_distribution": sanity["class_counts"],
        "split_counts": sanity["split_counts"],
        "persona_holdout": {"counts": split["persona_split_counts"],
                            "note": "s7_s3c_convention; test personas never in train"},
        "od_holdout": {"counts": sanity["od_pool_sizes"], "disjoint": True,
                       "note": "train/val/test ODs pairwise disjoint; every test state uses only test ODs and test personas (double holdout)"},
        "accessibility_holdout": split["accessibility_holdout"],
        "curve_groups": {"n_groups": sanity["n_groups"],
                         "groups_with_lt3_classes": sanity["groups_with_lt3_classes"]},
        "teacher": {
            "prompt_version": "teacher_s8_accessibility_v0.1",
            "k_distribution": {"3": 111, "5": 227},
            "n_calls_planned": 1468,
            "incomplete": 0,
        },
        "identity_leakage": {
            "quote_level_hits": 0,
            "check": "build assertion in src/traveler_distillation/accessibility/accessibility_dataset.py "
                     "(every stop_id/route_id/node id string is forbidden in student/teacher-facing state)",
        },
        "sanity": {
            "door_to_door_identity_max_err_min": sanity["door_to_door_identity_max_err_min"],
        },
        "files": {
            "records": {"path": "data/singapore_accessibility/records.jsonl",
                        "sha256": sha256(sdir / "records.jsonl"),
                        "n_states": 338},
            "states_with_teacher": {"path": "data/singapore_accessibility/states_with_teacher.jsonl",
                                    "sha256": sha256(sdir / "states_with_teacher.jsonl")},
            "repeat_records": {"path": "data/singapore_accessibility/repeat_records.jsonl",
                               "sha256": sha256(sdir / "repeat_records.jsonl")},
            "split_manifest": {"path": "data/singapore_accessibility/split_manifest.json",
                               "sha256": sha256(sdir / "split_manifest.json")},
            "generation_manifest": {"path": "data/singapore_accessibility/generation_manifest.json",
                                    "sha256": sha256(sdir / "generation_manifest.json")},
        },
        "note": "Manifest records identities/checksums; bulk data files remain at their original paths (not duplicated).",
    }
    write_json(RELEASE / "data_manifest" / "s8_accessibility_dataset.json", dataset)
    copy_file(sdir / "split_manifest.json", RELEASE / "data_manifest" / "s8_split_manifest.json")
    copy_file(sdir / "generation_manifest.json", RELEASE / "data_manifest" / "s8_generation_manifest.json")
    copy_file(sdir / "DATA_README.md", RELEASE / "data_manifest" / "DATA_README_singapore_accessibility.md")


# ----------------------------------------------------------------------------
# Step 9: final metrics (verbatim projection of the historical eval artifacts)
# ----------------------------------------------------------------------------
def build_metrics() -> None:
    acc = json.loads((ROOT / "outputs" / "s8_accessibility_eval" / "eval_metrics.json").read_text(encoding="utf-8"))
    reg = json.loads((ROOT / "outputs" / "s8_regression" / "eval_metrics.json").read_text(encoding="utf-8"))
    uod = json.loads((ROOT / "outputs" / "s8_unseen_od" / "unseen_od_audit.json").read_text(encoding="utf-8"))

    b0, b1, t = acc["models"]["B0_S7W3"], acc["models"]["B1_S8"], acc["models"]["Teacher"]
    w3, s8 = reg["models"]["W3_S7W3"], reg["models"]["S8"]

    final_metrics = {
        "model": "S8",
        "release": "Supply-Aware Traveler Agent v1.0",
        "base": "S7-W3 Generic Behavioral Core v1.0",
        "status": "FROZEN",
        "source": {
            "accessibility": "outputs/s8_accessibility_eval/eval_metrics.json (test-only, unseen persona x unseen OD, paired bootstrap B=2000 seed=42)",
            "regression": "outputs/s8_regression/eval_metrics.json (same pipelines as S7, paired bootstrap)",
            "unseen_od": "outputs/s8_unseen_od/unseen_od_audit.json",
            "protocol": "test-only; paired bootstrap B=2000 seed=42; 95% CI",
        },
        "accessibility": {
            "fidelity_all": {"B0_S7W3": b0["fidelity"]["all"], "B1_S8": b1["fidelity"]["all"]},
            "sensitivity_delta_P_pt": {"B0_S7W3": b0["sensitivity"]["delta_P_pt_best_minus_worst"],
                                       "B1_S8": b1["sensitivity"]["delta_P_pt_best_minus_worst"],
                                       "Teacher": t["sensitivity"]["delta_P_pt_best_minus_worst"],
                                       "n_groups": b0["sensitivity"]["n_groups"]},
            "monotonicity": {"B0_S7W3": {k: v for k, v in b0["monotonicity"].items() if k in ("pair_agreement", "triplet_agreement", "n_pairs", "n_triplets")},
                             "B1_S8": {k: v for k, v in b1["monotonicity"].items() if k in ("pair_agreement", "triplet_agreement", "n_pairs", "n_triplets")},
                             "Teacher": {k: v for k, v in t["monotonicity"].items() if k in ("pair_agreement", "triplet_agreement", "n_pairs", "n_triplets")}},
            "fvr": {"B0_S7W3": b0["fvr"], "B1_S8": b1["fvr"]},
            "convenience_error_by_class": {"B0_S7W3": b0["convenience_error"], "B1_S8": b1["convenience_error"]},
            "teacher_pt_prob_by_class": t["pt_prob_by_class"],
        },
        "unseen_od": {
            "od_holdout": uod["unseen_od_audit"]["od_holdout"],
            "test_records": uod["unseen_od_audit"]["test_records"],
            "pt_probability_mae": {k: v["pt_probability_mae"] for k, v in uod["summary_from_accessibility_eval"]["models"].items()},
        },
        "legacy": {"W3_S7W3": w3["legacy"], "S8": s8["legacy"]},
        "multi_axis": {
            "seen_joint": {"W3_S7W3": w3["seen_joint"], "S8": s8["seen_joint"]},
            "unseen_joint": {"W3_S7W3": w3["unseen_joint"], "S8": s8["unseen_joint"]},
            "interaction_l1_error": {"W3_S7W3": w3["interaction_l1_error"], "S8": s8["interaction_l1_error"]},
        },
        "mechanism": {
            axis: {"W3_S7W3": reg["causal_mechanism"]["models"]["W3_S7W3"][axis],
                   "S8": reg["causal_mechanism"]["models"]["S8"][axis],
                   "delta_s8_vs_s7w3": reg["causal_mechanism"]["deltas_s8_vs_s7w3"].get(axis)}
            for axis in ("congestion", "parking_cost")
        },
        "deltas_s8_vs_s7w3": {
            "accessibility": acc["deltas_s8_vs_s7w3"],
            "legacy_and_joint": reg["deltas_s8_vs_s7w3"],
        },
        "regression_gate": reg["regression_gate"],
    }
    write_json(RELEASE / "metrics" / "final_metrics.json", final_metrics)

    # comparison_s7_vs_s8.csv (mechanical projection, no transcription)
    def unpack(m_dict):
        return (float(m_dict["mean"]), float(m_dict["ci_low"]), float(m_dict["ci_high"]))

    rows = []
    def add(name, s7, s8v, teacher=None, delta=None, source="accessibility_eval"):
        row = {"metric": name, "source": source}
        if s7 is not None:
            m, lo, hi = unpack(s7); row.update(s7w3_mean=m, s7w3_ci_low=lo, s7w3_ci_high=hi)
        if s8v is not None:
            m, lo, hi = unpack(s8v); row.update(s8_mean=m, s8_ci_low=lo, s8_ci_high=hi)
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
    add("legacy_mode_accuracy", w3["legacy"]["mode_accuracy"], s8["legacy"]["mode_accuracy"], source="regression_eval")
    add("legacy_kl", w3["legacy"]["kl"], s8["legacy"]["kl"], delta=reg["deltas_s8_vs_s7w3"]["legacy_kl_delta"], source="regression_eval")
    add("legacy_probability_l1", w3["legacy"]["probability_l1"], s8["legacy"]["probability_l1"], source="regression_eval")
    add("legacy_delta_p_gap", w3["legacy"]["delta_p_gap"], s8["legacy"]["delta_p_gap"], source="regression_eval")
    add("legacy_sign_agreement", w3["legacy"]["sign_agreement"], s8["legacy"]["sign_agreement"], source="regression_eval")
    add("seen_joint_kl", w3["seen_joint"]["kl"], s8["seen_joint"]["kl"], delta=reg["deltas_s8_vs_s7w3"]["seen_joint_kl_delta"], source="regression_eval")
    add("unseen_joint_kl", w3["unseen_joint"]["kl"], s8["unseen_joint"]["kl"], delta=reg["deltas_s8_vs_s7w3"]["unseen_joint_kl_delta"], source="regression_eval")
    add("interaction_l1_error", w3["interaction_l1_error"], s8["interaction_l1_error"], source="regression_eval")
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
    add("mechanism_parking_cost_Gap_shortcut",
        reg["causal_mechanism"]["models"]["W3_S7W3"]["parking_cost"]["Gap_shortcut"],
        reg["causal_mechanism"]["models"]["S8"]["parking_cost"]["Gap_shortcut"], source="regression_eval")

    csv_path = RELEASE / "metrics" / "comparison_s7_vs_s8.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["metric", "source", "s7w3_mean", "s7w3_ci_low", "s7w3_ci_high",
                  "s8_mean", "s8_ci_low", "s8_ci_high", "teacher_mean", "teacher_ci_low",
                  "teacher_ci_high", "delta_mean", "delta_ci_low", "delta_ci_high",
                  "delta_ci_excludes_zero"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------------------------------------------------------
# Step 10: reports
# ----------------------------------------------------------------------------
def build_reports() -> None:
    for name in ("S8_SCHEMA_AUDIT.md", "EXPERIMENT_REPORT_S8_TRANSIT_ACCESSIBILITY.md"):
        copy_file(require(ROOT / "reports" / name), RELEASE / "reports" / name)
    for name in ("EXPERIMENT_REPORT_S5_MULTI_AXIS.md",
                 "EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md",
                 "EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md"):
        copy_file(require(ROOT / "outputs" / name), RELEASE / "reports" / name)


# ----------------------------------------------------------------------------
# Step 11: code manifest (git facts captured by the PowerShell driver)
# ----------------------------------------------------------------------------
def build_code_manifest() -> None:
    cm = RELEASE / "code_manifest"
    for name in ("git_commit.txt", "git_status.txt", "environment.txt", "uncommitted.patch"):
        if not (cm / name).exists():
            raise FileNotFoundError(f"code_manifest/{name} missing — capture git/environment facts first")
    write_json(cm / "code_manifest_provenance.json", {
        "note": "git_commit.txt = code commit at which S8 was trained/evaluated (65e505f); "
                "git_status.txt = worktree status captured immediately before the freeze commit; "
                "uncommitted.patch = git diff of tracked files plus untracked-file inventory at that moment.",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": "3.12.13",
            "torch": "2.13.0+cpu",
            "java": "OpenJDK 25.0.4 (Temurin)",
            "matsim": "MATSim 2026.0 (C:/Users/xuan1/AppData/Local/Temp/matsim_rel/matsim-2026.0.jar)",
        },
    })


# ----------------------------------------------------------------------------
# Step 13: checksums
# ----------------------------------------------------------------------------
def build_checksums() -> None:
    cks = RELEASE / "checksums"
    cks.mkdir(parents=True, exist_ok=True)
    targets = sorted(
        p for p in RELEASE.rglob("*")
        if p.is_file() and p.parent.name != "checksums"
    )
    lines = ["# SHA256 checksums of the frozen S8 release.",
             "# Any change to a checksum after freeze means the version has been modified.",
             f"# Generated at {datetime.now(timezone.utc).isoformat()}"]
    for p in targets:
        rel = p.relative_to(RELEASE).as_posix()
        lines.append(f"{sha256(p)}  {rel}")
    (cks / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checksummed {len(targets)} files -> {cks / 'SHA256SUMS.txt'}")


# ----------------------------------------------------------------------------
# Step 8 gate verification: reproduction artifacts vs historical artifacts
# ----------------------------------------------------------------------------
def _load_gate_json(name: str):
    p = GATE_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def verify_gate(tol: float = 1e-4) -> int:
    acc_hist = json.loads((ROOT / "outputs" / "s8_accessibility_eval" / "eval_metrics.json").read_text(encoding="utf-8"))
    reg_hist = json.loads((ROOT / "outputs" / "s8_regression" / "eval_metrics.json").read_text(encoding="utf-8"))
    uod_hist = json.loads((ROOT / "outputs" / "s8_unseen_od" / "unseen_od_audit.json").read_text(encoding="utf-8"))

    acc_gate = _load_gate_json("accessibility_eval/eval_metrics.json")
    reg_gate = _load_gate_json("regression_eval/eval_metrics.json")
    uod_gate = _load_gate_json("unseen_od/unseen_od_audit.json")
    smoke = _load_gate_json("singapore_smoke_result.json")

    gates = []
    if acc_gate is None:
        print("MISSING accessibility reproduction artifact")
        return 1
    b0g, b1g = acc_gate["models"]["B0_S7W3"], acc_gate["models"]["B1_S8"]
    b0h, b1h = acc_hist["models"]["B0_S7W3"], acc_hist["models"]["B1_S8"]
    gates += [
        ("accessibility PT prob MAE (S8)", b1h["fidelity"]["all"]["pt_probability_mae"]["mean"],
         b1g["fidelity"]["all"]["pt_probability_mae"]["mean"]),
        ("accessibility mean P(PT|infeasible) (S8)", b1h["fvr"]["mean_P_pt_infeasible"]["mean"],
         b1g["fvr"]["mean_P_pt_infeasible"]["mean"]),
        ("accessibility pair monotonicity (S8)", b1h["monotonicity"]["pair_agreement"]["mean"],
         b1g["monotonicity"]["pair_agreement"]["mean"]),
        ("accessibility triplet monotonicity (S8)", b1h["monotonicity"]["triplet_agreement"]["mean"],
         b1g["monotonicity"]["triplet_agreement"]["mean"]),
        ("accessibility sensitivity dP_PT (S8)", b1h["sensitivity"]["delta_P_pt_best_minus_worst"]["mean"],
         b1g["sensitivity"]["delta_P_pt_best_minus_worst"]["mean"]),
        ("accessibility KL (S7-W3 B0)", b0h["fidelity"]["all"]["kl"]["mean"],
         b0g["fidelity"]["all"]["kl"]["mean"]),
        ("accessibility PT prob MAE (S7-W3 B0)", b0h["fidelity"]["all"]["pt_probability_mae"]["mean"],
         b0g["fidelity"]["all"]["pt_probability_mae"]["mean"]),
    ]
    if uod_gate is None:
        print("MISSING unseen-OD reproduction artifact")
        return 1
    gates += [
        ("unseen OD n_test_ods", float(uod_hist["unseen_od_audit"]["test_records"]["n"]),
         float(uod_gate["unseen_od_audit"]["test_records"]["n"])),
        ("unseen OD all_unseen", 1.0,
         1.0 if uod_gate["unseen_od_audit"]["test_records"]["all_unseen"] else 0.0),
    ]
    if reg_gate is None:
        print("MISSING regression reproduction artifact")
        return 1
    w3g, s8g = reg_gate["models"]["W3_S7W3"], reg_gate["models"]["S8"]
    w3h, s8h = reg_hist["models"]["W3_S7W3"], reg_hist["models"]["S8"]
    gates += [
        ("legacy accuracy (S8)", s8h["legacy"]["mode_accuracy"]["mean"], s8g["legacy"]["mode_accuracy"]["mean"]),
        ("legacy KL (S8)", s8h["legacy"]["kl"]["mean"], s8g["legacy"]["kl"]["mean"]),
        ("seen joint KL (S8)", s8h["seen_joint"]["kl"]["mean"], s8g["seen_joint"]["kl"]["mean"]),
        ("unseen joint KL (S8)", s8h["unseen_joint"]["kl"]["mean"], s8g["unseen_joint"]["kl"]["mean"]),
        ("legacy accuracy (S7-W3)", w3h["legacy"]["mode_accuracy"]["mean"], w3g["legacy"]["mode_accuracy"]["mean"]),
        ("mechanism parking G_med (S8)",
         reg_hist["causal_mechanism"]["models"]["S8"]["parking_cost"]["G_med"]["mean"],
         reg_gate["causal_mechanism"]["models"]["S8"]["parking_cost"]["G_med"]["mean"]),
        ("mechanism congestion Gap_shortcut (S8)",
         reg_hist["causal_mechanism"]["models"]["S8"]["congestion"]["Gap_shortcut"]["mean"],
         reg_gate["causal_mechanism"]["models"]["S8"]["congestion"]["Gap_shortcut"]["mean"]),
    ]

    ok = True
    print(f"{'gate':46s} {'historical':>12s} {'re-run':>12s} {'|delta|':>10s}  verdict")
    for name, h, g in gates:
        d = abs(h - g)
        passed = d <= tol
        ok &= passed
        print(f"{name:46s} {h:12.6f} {g:12.6f} {d:10.6f}  {'PASS' if passed else 'FAIL'}")

    # Singapore smoke
    print()
    if smoke is None:
        print("Singapore smoke artifact MISSING")
        ok = False
    else:
        print(f"Singapore smoke: schema_match={smoke.get('schema_check', {}).get('schema_match')}, "
              f"gate_pass={smoke.get('gate_pass')}, matsim_exit={smoke.get('matsim_exit_code')}")
        if not smoke.get("gate_pass"):
            ok = False

    # S7 release must be 0-modified
    s7_ok = _verify_s7_intact()
    ok &= s7_ok

    print()
    print("REPRODUCTION GATE:", "PASS — frozen S8 checkpoint reproduces the historical values"
          if ok else "FAIL — pause the freeze and locate the cause")
    return 0 if ok else 1


def _verify_s7_intact() -> bool:
    sums = (S7_RELEASE / "checksums" / "SHA256SUMS.txt").read_text(encoding="utf-8")
    ok = True
    n = 0
    for line in sums.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, rel = line.split(None, 1)
        p = S7_RELEASE / rel
        if not p.exists():
            print(f"S7 release file MISSING: {rel}")
            ok = False
            continue
        if sha256(p) != h:
            print(f"S7 release file MODIFIED: {rel}")
            ok = False
        n += 1
    print(f"S7-W3 release integrity: {n} files, {'ALL UNCHANGED' if ok else 'MISMATCH DETECTED'}")
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
