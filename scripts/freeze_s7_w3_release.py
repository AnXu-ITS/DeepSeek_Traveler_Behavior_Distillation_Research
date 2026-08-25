#!/usr/bin/env python
"""S7-W3 backup & freeze executor (S7_W3_BACKUP_FREEZE_INSTRUCTIONS.md).

Mechanically implements the instruction steps:
  build         create releases/s7_w3_generic_core_v1/ with checkpoint, config,
                schema, normalization, metrics, reports, data/code manifests
                (steps 1-6, 8-10, 12-13 prerequisites)
  checksums     write checksums/SHA256SUMS.txt over every frozen file (step 11)
  verify-smoke  compare releases/.../smoke/*.json against the historical S7
                eval artifacts (step 7 reproducibility gate)

Steps 12/13 (FINAL_S7_W3_FREEZE.md, README.md) are hand-written files that must
exist before `checksums` runs. Step 14 (git commit/tag) and step 15 (read-only
protection) are performed outside this script.

Usage:
    python scripts/freeze_s7_w3_release.py build
    python scripts/freeze_s7_w3_release.py checksums
    python scripts/freeze_s7_w3_release.py verify-smoke
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

RELEASE = ROOT / "releases" / "s7_w3_generic_core_v1"
RELEASE_NAME = "s7_w3_generic_core_v1"
S7_OUT = ROOT / "outputs" / "student_s7_w3"


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


def count_jsonl(path: Path) -> int:
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def load_jsonl(path: Path):
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            out.append(json.loads(ln))
    return out


# ----------------------------------------------------------------------------
# Step 1: checkpoint
# ----------------------------------------------------------------------------
def build_checkpoint() -> None:
    src_ckpt = require(S7_OUT / "checkpoints" / "best.pt")
    dst_ckpt = RELEASE / "checkpoint" / "model.pt"
    copy_file(src_ckpt, dst_ckpt)
    copy_file(S7_OUT / "training_history.json", RELEASE / "checkpoint" / "training_history.json")
    copy_file(S7_OUT / "val_metrics.json", RELEASE / "checkpoint" / "val_metrics.json")

    ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    n_params = sum(v.numel() for v in ck["model_state"].values())
    meta = {
        "release": RELEASE_NAME,
        "checkpoint_source": str(src_ckpt.relative_to(ROOT)),
        "checkpoint_saved_at_utc": iso_mtime(src_ckpt),
        "seed": 42,
        "best_epoch": 1,
        "runtime_seconds": json.loads((S7_OUT / "val_metrics.json").read_text(encoding="utf-8"))["runtime_seconds"],
        "parameters": n_params,
        "architecture": ck["config"],
        "init_checkpoint": ck["init_checkpoint"],
        "lambda_mechanism": ck["lambda_mechanism"],
        "lambda_broken": ck["lambda_broken"],
        "checkpoint_keys": sorted(ck.keys()),
        "optimizer_state": "NOT PRESENT (S7 best.pt stores model weights only)",
        "scheduler_state": "NOT PRESENT",
        "training_state": "training_history.json + val_metrics.json (copied alongside)",
        "sha256": sha256(dst_ckpt),
        "params_expected": 24370,
    }
    assert meta["parameters"] == meta["params_expected"], "parameter count mismatch"
    write_json(RELEASE / "checkpoint" / "model.pt.meta.json", meta)

    # 4-seed checkpoints (seed 42 = model.pt above)
    for seed in (7, 123, 2024):
        sdir = ROOT / "outputs" / f"student_s7_w3_seed{seed}"
        dst = RELEASE / "checkpoint" / "seeds" / f"seed{seed}"
        copy_file(require(sdir / "checkpoints" / "best.pt"), dst / "model.pt")
        copy_file(sdir / "training_history.json", dst / "training_history.json")
        copy_file(sdir / "val_metrics.json", dst / "val_metrics.json")
        sm = {
            "seed": seed,
            "checkpoint_source": str((sdir / "checkpoints" / "best.pt").relative_to(ROOT)),
            "checkpoint_saved_at_utc": iso_mtime(sdir / "checkpoints" / "best.pt"),
            "sha256": sha256(dst / "model.pt"),
        }
        write_json(dst / "model.pt.meta.json", sm)


# ----------------------------------------------------------------------------
# Step 2: config
# ----------------------------------------------------------------------------
def build_config() -> None:
    cfg = require(ROOT / "configs" / "student_s7_w3.yaml")
    copy_file(cfg, RELEASE / "config" / "student_s7_w3.yaml")
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    write_json(RELEASE / "config" / "checkpoint_embedded_config.json",
               {"source": "checkpoint key 'config' (training-time snapshot)",
                "config": ck["config"]})


# ----------------------------------------------------------------------------
# Step 3: input/output schema
# ----------------------------------------------------------------------------
def build_schema() -> None:
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    spec = ck["feature_spec"]
    ext = ck["extractor_state"]

    input_schema = {
        "schema_version": "1.0",
        "frozen_source": "src/traveler_distillation/student/features.py (FeatureExtractor.encode)",
        "persona_features": {"categorical": PERSONA_CAT, "numeric": PERSONA_NUM},
        "trip_features": {"categorical": TRIP_CAT, "numeric": TRIP_NUM},
        "dynamic_context": {"categorical": CONTEXT_CAT, "numeric": CONTEXT_NUM},
        "mode_level_attributes": ALT_NUM,
        "availability_representation": {
            "choice_set_encoding": "state.alternatives list order preserved; each mode -> mode_vocab index",
            "mask_convention": "alt_mask float tensor: 1.0 = available, 0.0 = unavailable/padded",
            "unavailable_mode_behavior": "masked alternatives receive -inf logits in masked_softmax -> zero probability mass",
        },
        "encoding": {
            "categorical": "learned embedding; vocab index per field; <UNK> fallback = index 0",
            "numeric": "z-score using per-field mean/std fit on the S7 TRAIN split only",
        },
        "tensor_layout": {
            "global_cat": {"dtype": "long", "n": len(GLOBAL_CAT), "order": GLOBAL_CAT},
            "global_num": {"dtype": "float32", "n": len(GLOBAL_NUM), "order": GLOBAL_NUM},
            "alt_mode_idx": {"dtype": "long", "per_alternative": "mode_vocab index"},
            "alt_num": {"dtype": "float32", "n_per_alternative": len(ALT_NUM), "order": ALT_NUM},
            "alt_mask": {"dtype": "float32", "per_alternative": "1.0/0.0 availability"},
        },
        "vocabularies": {"mode_vocab": ext["mode_vocab"], "cat_vocab_sizes": spec["cat_vocab_sizes"]},
    }
    write_json(RELEASE / "schema" / "input_schema.json", input_schema)

    output_schema = {
        "schema_version": "1.0",
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
    write_json(RELEASE / "schema" / "output_schema.json", output_schema)

    lines = [
        "# Feature order for TravelerStudent.forward — MUST match actual tensor input order (frozen S7-W3).",
        f"# Generated from src/traveler_distillation/student/features.py constants at freeze time.",
        f"# global_cat (torch.long, n={len(GLOBAL_CAT)}):",
    ]
    lines += [f"#   {i}: {name}" for i, name in enumerate(GLOBAL_CAT)]
    lines.append(f"# global_num (torch.float32, n={len(GLOBAL_NUM)}):")
    lines += [f"#   {i}: {name}" for i, name in enumerate(GLOBAL_NUM)]
    lines.append("# alt_mode_idx (torch.long, per alternative): mode_vocab index")
    lines.append(f"# alt_num (torch.float32, per alternative, n={len(ALT_NUM)}):")
    lines += [f"#   {i}: {name}" for i, name in enumerate(ALT_NUM)]
    lines.append("# alt_mask (torch.float32, per alternative): 1.0 = available, 0.0 = unavailable/padded")
    lines.append("")
    lines.append("# Canonical column order (one tensor per line, columns within a tensor in listed order):")
    lines.append("global_cat := " + ",".join(GLOBAL_CAT))
    lines.append("global_num := " + ",".join(GLOBAL_NUM))
    lines.append("alt_mode_idx := <mode_vocab index per alternative, order = state.alternatives>")
    lines.append("alt_num := " + ",".join(ALT_NUM) + " (per alternative, order = state.alternatives)")
    lines.append("alt_mask := <1.0/0.0 per alternative, order = state.alternatives>")
    (RELEASE / "schema").mkdir(parents=True, exist_ok=True)
    (RELEASE / "schema" / "feature_order.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------
# Step 4: normalization / encoding
# ----------------------------------------------------------------------------
def build_normalization() -> None:
    ck = torch.load(RELEASE / "checkpoint" / "model.pt", map_location="cpu", weights_only=False)
    ext = ck["extractor_state"]
    write_json(RELEASE / "normalization" / "normalization.json",
               {"source": "checkpoint key 'extractor_state' (fit on S7 TRAIN split only)",
                "numeric_fields": {"global": GLOBAL_NUM, "alternative": ALT_NUM},
                "mean": ext["num_mean"],
                "std": ext["num_std"],
                "note": "x_z = (x - mean) / std; std floored at 1e-8 in code"})
    write_json(RELEASE / "normalization" / "mode_mapping.json",
               {"source": "checkpoint key 'extractor_state'.mode_vocab",
                "mode_vocab": ext["mode_vocab"]})
    write_json(RELEASE / "normalization" / "category_mapping.json",
               {"source": "checkpoint key 'extractor_state'.cat_vocabs",
                "cat_vocabs": ext["cat_vocabs"],
                "note": "index 0 = <UNK> fallback in every vocab"})


# ----------------------------------------------------------------------------
# Step 8-10: metrics + seed robustness + reports
# ----------------------------------------------------------------------------
def build_metrics() -> None:
    # final_metrics.json is derived verbatim from the historical S7 eval artifacts
    # (no re-computation, no transcription): source of truth for the freeze.
    reg = json.loads((ROOT / "outputs" / "s7_regression" / "eval_metrics.json").read_text(encoding="utf-8"))
    cau = json.loads((ROOT / "outputs" / "s7_causal_eval" / "eval_metrics.json").read_text(encoding="utf-8"))
    seeds = json.loads((ROOT / "outputs" / "s7_seed_check" / "seed_summary.json").read_text(encoding="utf-8"))

    final_metrics = {
        "model": "S7-W3",
        "release": "Generic Behavioral Core v1.0",
        "status": "FROZEN",
        "source": {
            "legacy_and_joint": "outputs/s7_regression/eval_metrics.json (models.W3)",
            "causal_mechanism": "outputs/s7_causal_eval/eval_metrics.json (models.W3)",
            "seed_robustness": "outputs/s7_seed_check/seed_summary.json",
            "protocol": "test-only; paired bootstrap B=2000 seed=42; units = quadruplet group (causal) / state (regression); 95% CI",
        },
        "legacy": reg["models"]["W3"]["legacy"],
        "multi_axis": {
            "seen_joint": reg["models"]["W3"]["seen_joint"],
            "unseen_joint": reg["models"]["W3"]["unseen_joint"],
            "interaction_l1_error": reg["models"]["W3"]["interaction_l1_error"],
        },
        "causal_mechanism": {
            axis: {
                "G_med": cau["models"]["W3"][axis]["G_med"],
                "Gap_shortcut": cau["models"]["W3"][axis]["Gap_shortcut"],
                "R_shortcut": cau["models"]["W3"][axis]["R_shortcut"],
                "R_mediator": cau["models"]["W3"][axis]["R_mediator"],
            }
            for axis in ("congestion", "parking_cost")
        },
        "robustness": {
            "seeds": seeds["seeds"],
            "stable_across_seeds": seeds["stable_across_seeds"],
            "verdict": seeds["verdict"],
            "regressed_seeds": seeds["regressed_seeds"],
            "interpretation": seeds["interpretation"],
            "notes": [
                "4/4 seeds: parking G_med delta CI excludes 0 and is negative (repair)",
                "4/4 seeds: congestion Gap_shortcut delta CI excludes 0 and is negative (repair)",
                "4/4 seeds: legacy KL delta CI excludes 0 and is negative (no regression)",
                "4/4 seeds: seen joint KL delta direction negative; none significantly regressed",
            ],
        },
    }
    write_json(RELEASE / "metrics" / "final_metrics.json", final_metrics)

    # seed_robustness.csv (mechanical projection of seed_summary.json)
    rows = []
    for ps in seeds["per_seed"]:
        row = {"seed": ps["seed"]}
        for key, lab in [
            ("parking_cost_G_med", "parking_G_med_delta"),
            ("congestion_G_broken", "congestion_G_broken_delta"),
            ("congestion_Gap_shortcut", "congestion_Gap_shortcut_delta"),
            ("legacy_kl_delta", "legacy_kl_delta"),
            ("seen_joint_kl_delta", "seen_joint_kl_delta"),
        ]:
            row[f"{lab}_mean"] = ps[key]["mean"]
            row[f"{lab}_ci_low"] = ps[key]["ci_low"]
            row[f"{lab}_ci_high"] = ps[key]["ci_high"]
            row[f"{lab}_ci_excludes_zero"] = ps[key]["excl_zero"]
        rows.append(row)
    csv_path = RELEASE / "metrics" / "seed_robustness.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # seed eval artifacts verbatim
    for name in ("seed_summary.json", "causal_eval_seeds.json", "regression_eval_seeds.json"):
        copy_file(require(ROOT / "outputs" / "s7_seed_check" / name), RELEASE / "metrics" / "seeds" / name)
    copy_file(require(ROOT / "outputs" / "s7_selection" / "selection.json"),
              RELEASE / "metrics" / "selection" / "selection.json")
    copy_file(require(ROOT / "outputs" / "s7_selection" / "variant_distances.json"),
              RELEASE / "metrics" / "selection" / "variant_distances.json")

    # generated seed robustness report (mechanical markdown, no new claims)
    md_lines = [
        "# S7-W3 Seed Robustness Report (frozen artifact)",
        "",
        "Source: `outputs/s7_seed_check/seed_summary.json` (verbatim projection).",
        "W3 config retrained under 4 training seeds (42, 7, 123, 2024), zero API calls,",
        "evaluated on the same test-only pipeline with paired bootstrap CIs.",
        "",
        "| seed | parking G_med Δ | congestion G_broken Δ | congestion Gap_shortcut Δ | Δ legacy KL | Δ seen joint KL |",
        "|---|---|---|---|---|---|",
    ]
    for ps in seeds["per_seed"]:
        def fmt(key):
            v = ps[key]
            star = "*" if v["excl_zero"] else ""
            return f'{v["mean"]:+.4f} [{v["ci_low"]:+.4f}, {v["ci_high"]:+.4f}]{star}'
        md_lines.append(
            f'| {ps["seed"]} | {fmt("parking_cost_G_med")} | {fmt("congestion_G_broken")} | '
            f'{fmt("congestion_Gap_shortcut")} | {fmt("legacy_kl_delta")} | {fmt("seen_joint_kl_delta")} |'
        )
    md_lines += [
        "",
        f'**Verdict**: stable_across_seeds = {seeds["stable_across_seeds"]}; regressed_seeds = {seeds["regressed_seeds"]}',
        f'**Interpretation**: {seeds["interpretation"]}',
        "",
        "CIs are 95% paired bootstrap (B=2000, seed=42); * = CI excludes 0.",
    ]
    (RELEASE / "reports").mkdir(parents=True, exist_ok=True)
    (RELEASE / "reports" / "SEED_ROBUSTNESS_S7_W3.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def build_reports() -> None:
    for name in ("EXPERIMENT_REPORT_S5_MULTI_AXIS.md",
                 "EXPERIMENT_REPORT_S6_CAUSAL_AUDIT.md",
                 "EXPERIMENT_REPORT_S7_MECHANISM_AWARE.md"):
        copy_file(require(ROOT / "outputs" / name), RELEASE / "reports" / name)


# ----------------------------------------------------------------------------
# Step 5: data manifest
# ----------------------------------------------------------------------------
def _dataset_entry(kind: str, states_path: Path, extra: dict) -> dict:
    states_path = require(states_path)
    states = load_jsonl(states_path)
    personas = set()
    k_dist: dict = {}
    incomplete = 0
    for s in states:
        st = s.get("state") or {}
        p = st.get("persona") or {}
        pid = p.get("persona_id") or s.get("persona_id")
        if pid:
            personas.add(pid)
        if s.get("status", "complete") != "complete":
            incomplete += 1
        k = ((s.get("teacher_aggregate") or {}).get("k")
             or (s.get("aggregation_metadata") or {}).get("k"))
        if k is not None:
            k_dist[int(k)] = k_dist.get(int(k), 0) + 1
        elif "teacher_k" in s:
            k_dist[int(s["teacher_k"])] = k_dist.get(int(s["teacher_k"]), 0) + 1
        for member in (s.get("members") or {}).values():
            mk = member.get("teacher_k")
            if mk is not None:
                k_dist[int(mk)] = k_dist.get(int(mk), 0) + 1
    entry = {
        "kind": kind,
        "path": str(states_path.relative_to(ROOT)),
        "n_states": len(states),
        "n_personas": len(personas),
        "teacher_k_distribution": k_dist,
        "incomplete_states_in_file": incomplete,
        "sha256": sha256(states_path),
        "file_size_bytes": states_path.stat().st_size,
        "creation_date_utc": iso_mtime(states_path),
    }
    entry.update(extra)
    return entry


def build_data_manifest() -> None:
    manifest = {
        "release": RELEASE_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Manifest records identities/checksums; bulk data files remain at their original paths (not duplicated).",
        "datasets": [
            _dataset_entry("S3_legacy_single_axis", ROOT / "data" / "student_v0_3_s3" / "aggregated_teacher_dataset.jsonl",
                           {"source_model": "deepseek-v4-pro",
                            "teacher_k_default": 3,
                            "split": "persona holdout (80 split groups; test = 6 unseen personas / 226 states)",
                            "repeat_records": {
                                "path": "data/student_v0_3_s3/repeat_records.jsonl",
                                "n_repeats": count_jsonl(ROOT / "data" / "student_v0_3_s3" / "repeat_records.jsonl"),
                                "sha256": sha256(ROOT / "data" / "student_v0_3_s3" / "repeat_records.jsonl"),
                            },
                            "provenance": "merged S1-S4 axis extension batches over 40 personas (PROGRESS.md S3: 1513 states / 4539 repeats / 80 groups)"}),
            _dataset_entry("S5_joint_full", ROOT / "data" / "student_s5_joint" / "aggregated_teacher_dataset.jsonl",
                           {"source_model": "deepseek-v4-pro",
                            "combos": ["rain_x_congestion", "fare_x_congestion", "fare_x_transit_delay", "road_disruption_x_congestion"],
                            "k7_subset": "fare_x_transit_delay, 40 states",
                            "k5_view": {
                                "path": "data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl",
                                "n_states": count_jsonl(ROOT / "data" / "student_s5_joint" / "aggregated_teacher_dataset_k5.jsonl"),
                                "sha256": sha256(ROOT / "data" / "student_s5_joint" / "aggregated_teacher_dataset_k5.jsonl"),
                            },
                            "generation_manifest": "data/student_s5_joint/generation_manifest.json (n_joint_states=640, n_teacher_calls=2640)",
                            "split": "persona holdout; fare_x_congestion = unseen combination holdout",
                            "provenance": "PROGRESS.md S5: 640 joint states / 2640 teacher calls / K={3:320, 5:280, 7:40}"}),
            _dataset_entry("S6_causal_audit", ROOT / "data" / "causal_audit" / "states_with_teacher.jsonl",
                           {"source_model": "deepseek-v4-pro",
                            "teacher_k_note": "A/B states reuse S3 K=3 targets; C/D new states K=5 (2435 calls, 0 incomplete)",
                            "axes": ["congestion", "parking_cost", "transit_delay"],
                            "states_without_teacher": "data/causal_audit/states.jsonl",
                            "provenance": "PROGRESS.md S6: 960 audit states (A/B/C/D quadruplets); final S7 test evaluation uses only the 8 test quadruplet groups per axis",
                            "pre_teacher_path": "data/causal_audit/states.jsonl",
                            "pre_teacher_sha256": sha256(ROOT / "data" / "causal_audit" / "states.jsonl")}),
            _dataset_entry("S7_mechanism_quadruplets", ROOT / "data" / "student_s7_mechanism" / "quadruplets.jsonl",
                           {"source_model": "S6 verbatim (0 new API calls)",
                            "teacher_k_note": "A/B=K3 (S3 reuse), C/D=K5 (S6)",
                            "axes": ["congestion", "parking_cost"],
                            "split": "s3c_persona_holdout, car-available filtered; per axis train 26 / val 2 / test 8",
                            "split_manifest": "data/student_s7_mechanism/split_manifest.json",
                            "split_manifest_sha256": sha256(ROOT / "data" / "student_s7_mechanism" / "split_manifest.json"),
                            "provenance": "built by scripts/build_s7_mechanism_dataset.py from S6 states; test quadruplets never used for train/val/selection"}),
        ],
    }
    write_json(RELEASE / "data_manifest" / "data_manifest.json", manifest)


def build_code_manifest() -> None:
    # git/environment facts are captured by the PowerShell driver (git_commit.txt,
    # git_status.txt, environment.txt, uncommitted.patch) before this release is
    # committed; this function only asserts they exist and records provenance.
    cm = RELEASE / "code_manifest"
    for name in ("git_commit.txt", "git_status.txt", "environment.txt", "uncommitted.patch"):
        if not (cm / name).exists():
            raise FileNotFoundError(f"code_manifest/{name} missing — capture git/environment facts first")
    write_json(cm / "code_manifest_provenance.json", {
        "note": "git_commit.txt = code commit at which S7-W3 was trained/evaluated; "
                "git_status.txt = worktree status captured immediately before the freeze commit; "
                "uncommitted.patch = git diff of tracked files plus untracked-file inventory at that moment.",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    })


# ----------------------------------------------------------------------------
# Step 11: checksums
# ----------------------------------------------------------------------------
def build_checksums() -> None:
    cks = RELEASE / "checksums"
    cks.mkdir(parents=True, exist_ok=True)
    targets = sorted(
        p for p in RELEASE.rglob("*")
        if p.is_file() and p.parent.name != "checksums"
    )
    lines = ["# SHA256 checksums of the frozen S7-W3 release.",
             "# Any change to a checksum after freeze means the version has been modified.",
             f"# Generated at {datetime.now(timezone.utc).isoformat()}"]
    for p in targets:
        rel = p.relative_to(RELEASE).as_posix()
        lines.append(f"{sha256(p)}  {rel}")
    (cks / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checksummed {len(targets)} files -> {cks / 'SHA256SUMS.txt'}")


# ----------------------------------------------------------------------------
# Step 7 gate: smoke verification
# ----------------------------------------------------------------------------
def verify_smoke(tol: float = 1e-4) -> int:
    """Compare release smoke runs against historical S7 eval artifacts."""
    reg_hist = json.loads((ROOT / "outputs" / "s7_regression" / "eval_metrics.json").read_text(encoding="utf-8"))
    cau_hist = json.loads((ROOT / "outputs" / "s7_causal_eval" / "eval_metrics.json").read_text(encoding="utf-8"))
    reg_smoke = json.loads((RELEASE / "smoke" / "regression_smoke.json").read_text(encoding="utf-8"))
    cau_smoke = json.loads((RELEASE / "smoke" / "causal_smoke.json").read_text(encoding="utf-8"))

    gates = [
        ("legacy mode_accuracy mean", reg_hist["models"]["W3"]["legacy"]["mode_accuracy"]["mean"],
         reg_smoke["models"]["W3"]["legacy"]["mode_accuracy"]["mean"]),
        ("legacy KL mean", reg_hist["models"]["W3"]["legacy"]["kl"]["mean"],
         reg_smoke["models"]["W3"]["legacy"]["kl"]["mean"]),
        ("legacy probability L1 mean", reg_hist["models"]["W3"]["legacy"]["probability_l1"]["mean"],
         reg_smoke["models"]["W3"]["legacy"]["probability_l1"]["mean"]),
        ("seen joint KL mean", reg_hist["models"]["W3"]["seen_joint"]["kl"]["mean"],
         reg_smoke["models"]["W3"]["seen_joint"]["kl"]["mean"]),
        ("unseen joint KL mean", reg_hist["models"]["W3"]["unseen_joint"]["kl"]["mean"],
         reg_smoke["models"]["W3"]["unseen_joint"]["kl"]["mean"]),
        ("parking G_med mean", cau_hist["models"]["W3"]["parking_cost"]["G_med"]["mean"],
         cau_smoke["models"]["W3"]["parking_cost"]["G_med"]["mean"]),
        ("congestion Gap_shortcut mean", cau_hist["models"]["W3"]["congestion"]["Gap_shortcut"]["mean"],
         cau_smoke["models"]["W3"]["congestion"]["Gap_shortcut"]["mean"]),
    ]
    ok = True
    print(f"{'gate':38s} {'historical':>12s} {'smoke':>12s} {'|delta|':>10s}  verdict")
    for name, h, s in gates:
        d = abs(h - s)
        passed = d <= tol
        ok &= passed
        print(f"{name:38s} {h:12.6f} {s:12.6f} {d:10.6f}  {'PASS' if passed else 'FAIL'}")
    print("\nSMOKE GATE:", "PASS — reproduction consistent with historical values" if ok
          else "FAIL — pause the freeze and locate the cause")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "checksums", "verify-smoke"])
    args = ap.parse_args()
    if args.stage == "build":
        build_checkpoint()
        build_config()
        build_schema()
        build_normalization()
        build_metrics()
        build_reports()
        build_data_manifest()
        build_code_manifest()
        print(f"release built at {RELEASE}")
    elif args.stage == "checksums":
        build_checksums()
    elif args.stage == "verify-smoke":
        return verify_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
