#!/usr/bin/env python
"""S8 — transit accessibility adaptation training (Case B, from frozen S7-W3).

Initializes Student-S8 (24,562 params, 12 alternative numeric features) from
the FROZEN S7-W3 release via partial weight copy (new columns zero), then
fine-tunes on the replay mix (S8 §17):

    legacy single-axis : S5 joint : S7 mechanism : S8 accessibility = 2:1:1:1

plus one heterogeneity-pair pass per epoch. No accessibility-specific loss in
round 1 (S8 §19 — baseline fine-tune only).

HARD CONSTRAINTS:
  * The frozen release is only READ (never written); the output directory must
    pass assert_not_frozen_output (freeze step 15).
  * The S8 TEST split (unseen personas AND unseen ODs) is never loaded for
    training or validation — asserted at load time.
  * New-feature z-score statistics are fit on the S8 TRAIN split only; the six
    S7 features keep the frozen extractor statistics verbatim.

Usage:
    python scripts/train_student_s8.py \
        --base-dataset data/student_v0_3_s3/aggregated_teacher_dataset.jsonl \
        --joint-dataset data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl \
        --mechanism-dataset data/student_s7_mechanism/quadruplets.jsonl \
        --accessibility-dataset data/singapore_accessibility/states_with_teacher.jsonl \
        --split-manifest data/singapore_accessibility/split_manifest.json \
        --config configs/student_s8.yaml \
        --output outputs/student_s8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from traveler_distillation.accessibility.accessibility_features import (
    ARCH_VERSION,
    S8FeatureExtractor,
    TravelerStudentS8,
)
from traveler_distillation.config import load_yaml
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget
from traveler_distillation.generators import resolve_combination
from traveler_distillation.student import (
    AggregatedTeacherDataset,
    CounterfactualPairDataset,
    MechanismQuadrupletDataset,
    PersonaContrastPairDataset,
    TravelerStudent,
    build_baseline_index,
    broken_path_fidelity_loss,
    collate_batch,
    collate_contrast_pairs,
    collate_pairs,
    collate_quadruplets,
    elasticity_direction_loss,
    elasticity_l1,
    elasticity_magnitude_loss,
    heterogeneity_l1,
    kl_divergence,
    load_mechanism_quadruplets,
    make_counterfactual_pairs,
    make_persona_contrast_pairs,
    mechanism_fidelity_loss,
    student_loss,
)
from traveler_distillation.student.release_guard import assert_not_frozen_output

FROZEN_RELEASE_CKPT = _ROOT / "releases" / "s7_w3_generic_core_v1" / "checkpoint" / "model.pt"


def _load(path: Path) -> list[AggregatedTeacherTarget]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(AggregatedTeacherTarget.model_validate(json.loads(line)))
    return samples


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    n = 0
    sum_kl = 0.0
    sum_l1 = 0.0
    correct = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred_idx = out["mode_probabilities"].argmax(dim=1)
        correct += (pred_idx == batch["target_mode_idx"]).sum().item()
        sum_kl += kl_divergence(batch["target_probs"], out["mode_probabilities"]).sum().item()
        sum_l1 += (batch["target_probs"] - out["mode_probabilities"]).abs().sum(dim=1).sum().item()
        n += batch["target_probs"].shape[0]
    return {
        "mode_accuracy": correct / n if n else 0.0,
        "kl": sum_kl / n if n else 0.0,
        "probability_l1": sum_l1 / n if n else 0.0,
    }


def static_losses(out, batch, l_cfg) -> dict:
    return student_loss(
        out, batch["target_probs"], batch["target_mode_idx"], batch["target_departure"],
        lambda_action=l_cfg["lambda_action"], lambda_distribution=l_cfg["lambda_distribution"],
        lambda_departure=l_cfg["lambda_departure"], huber_delta=l_cfg.get("huber_delta", 1.0),
    )


def elast_pair_loss(base_out, cf_out, base_batch, cf_batch, l_cfg) -> dict:
    base_l = static_losses(base_out, base_batch, l_cfg)
    cf_l = static_losses(cf_out, cf_batch, l_cfg)
    teacher_delta = cf_batch["target_probs"] - base_batch["target_probs"]
    student_delta = cf_out["mode_probabilities"] - base_out["mode_probabilities"]
    mask = cf_batch["alt_mask"]
    l1 = elasticity_l1(teacher_delta, student_delta, mask=mask)
    d = elasticity_direction_loss(teacher_delta, student_delta, mask=mask, margin=l_cfg.get("direction_margin", 0.0))
    m = elasticity_magnitude_loss(teacher_delta, student_delta, mask=mask)
    elast = l_cfg.get("lambda_elasticity", 0.0) * l1 + l_cfg.get("lambda_direction", 0.0) * d + l_cfg.get("lambda_magnitude", 0.0) * m
    return {
        "total": base_l["total"] + cf_l["total"] + elast,
        "static": base_l["total"] + cf_l["total"],
        "elasticity": l1,
        "elasticity_direction": d,
        "elasticity_magnitude": m,
    }


def het_pair_loss(a_out, b_out, a_batch, b_batch, l_cfg) -> dict:
    teacher_diff = a_batch["target_probs"] - b_batch["target_probs"]
    student_diff = a_out["mode_probabilities"] - b_out["mode_probabilities"]
    union_mask = (a_batch["alt_mask"] + b_batch["alt_mask"]).clamp(min=0.0, max=1.0)
    het = heterogeneity_l1(teacher_diff, student_diff, mask=union_mask)
    return {"total": l_cfg.get("lambda_heterogeneity", 1.0) * het, "heterogeneity": het}


def quadruplet_losses(out, batch, l_cfg) -> dict:
    static = 0.0
    n = batch["A"]["target_probs"].shape[0]
    for key in ("A", "B", "C", "D"):
        static += static_losses(out[key], batch[key], l_cfg)["total"].sum()
    mask = batch["A"]["alt_mask"]
    mech = mechanism_fidelity_loss(
        out["A"]["mode_probabilities"], out["B"]["mode_probabilities"], out["D"]["mode_probabilities"],
        batch["A"]["target_probs"], batch["B"]["target_probs"], batch["D"]["target_probs"],
        mask=mask,
    ).sum()
    broken = broken_path_fidelity_loss(
        out["A"]["mode_probabilities"], out["C"]["mode_probabilities"],
        batch["A"]["target_probs"], batch["C"]["target_probs"],
        mask=mask,
    ).sum()
    lam_m = l_cfg.get("lambda_mechanism", 0.0)
    lam_b = l_cfg.get("lambda_broken", 0.0)
    total = static + lam_m * mech + lam_b * broken
    return {
        "total": total / n,
        "static": static / n,
        "mechanism": mech / n,
        "broken": broken / n,
    }


@torch.no_grad()
def val_mechanism_gap(model, quads, extractor, device) -> dict:
    from collections import defaultdict
    model.eval()
    acc = defaultdict(lambda: {"G_nat": [], "G_broken": [], "G_med": []})
    for quad in quads:
        ds = MechanismQuadrupletDataset([quad], extractor)
        batch = collate_quadruplets([ds[0]])
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        out = {k: model(batch[k]) for k in ("A", "B", "C", "D")}
        pA, pB, pC, pD = (out[k]["mode_probabilities"][0] for k in ("A", "B", "C", "D"))
        tA, tB, tC, tD = (batch[k]["target_probs"][0] for k in ("A", "B", "C", "D"))
        mask = batch["A"]["alt_mask"][0]
        denom = mask.sum().item()
        G_nat = ((tB - tA) - (pB - pA)).abs().mul(mask).sum().item() / denom
        G_broken = ((tC - tA) - (pC - pA)).abs().mul(mask).sum().item() / denom
        G_med = ((tD - tA) - (pD - pA)).abs().mul(mask).sum().item() / denom
        a = acc[quad.axis_id]
        a["G_nat"].append(G_nat)
        a["G_broken"].append(G_broken)
        a["G_med"].append(G_med)
    out = {}
    for axis, a in acc.items():
        out[axis] = {k: round(sum(v) / len(v), 4) for k, v in a.items() if v}
        out[axis]["n_quads"] = len(a["G_nat"])
    return out


def _quad_step(model, quad_loader, device, l_cfg, optimizer, run) -> None:
    model.train()
    for batch in quad_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        out = {k: model(batch[k]) for k in ("A", "B", "C", "D")}
        losses = quadruplet_losses(out, batch, l_cfg)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        for k in ("static", "mechanism", "broken"):
            run[k] += losses[k].item() * batch["A"]["target_probs"].shape[0]
        run["n_quad"] += batch["A"]["target_probs"].shape[0]


def _pair_step(model, pair_loader, device, l_cfg, optimizer, run, kind) -> None:
    model.train()
    for batch in pair_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        base_out = model(batch["base"])
        cf_out = model(batch["cf"])
        losses = elast_pair_loss(base_out, cf_out, batch["base"], batch["cf"], l_cfg)
        optimizer.zero_grad()
        losses["total"].mean().backward()
        optimizer.step()
        run[kind] += losses["total"].sum().item()


def _het_step(model, het_loader, device, l_cfg, optimizer, run) -> None:
    model.train()
    for batch in het_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        a_out = model(batch["a"])
        b_out = model(batch["b"])
        losses = het_pair_loss(a_out, b_out, batch["a"], batch["b"], l_cfg)
        optimizer.zero_grad()
        losses["total"].mean().backward()
        optimizer.step()
        run["het"] += losses["total"].sum().item()


def _acc_step(model, acc_loader, device, l_cfg, optimizer, run) -> None:
    """Static (action/distribution/departure) step on S8 accessibility states."""
    model.train()
    for batch in acc_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        losses = static_losses(out, batch, l_cfg)
        optimizer.zero_grad()
        losses["total"].mean().backward()
        optimizer.step()
        run["accessibility"] += losses["total"].sum().item()


@torch.no_grad()
def _val_loss(model, val_quad_loader, val_elast_loader, val_het_loader, val_acc_loader,
              device, l_cfg) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in val_quad_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        out = {k: model(batch[k]) for k in ("A", "B", "C", "D")}
        total += quadruplet_losses(out, batch, l_cfg)["total"].sum().item()
        n += 1
    for batch in val_elast_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        losses = elast_pair_loss(model(batch["base"]), model(batch["cf"]), batch["base"], batch["cf"], l_cfg)
        total += losses["total"].sum().item()
        n += 1
    for batch in val_het_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        losses = het_pair_loss(model(batch["a"]), model(batch["b"]), batch["a"], batch["b"], l_cfg)
        total += losses["total"].sum().item()
        n += 1
    for batch in val_acc_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        losses = static_losses(model(batch), batch, l_cfg)
        total += losses["total"].sum().item()
        n += 1
    return total / n if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-dataset", default="data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--mechanism-dataset", default="data/student_s7_mechanism/quadruplets.jsonl")
    ap.add_argument("--accessibility-dataset", default="data/singapore_accessibility/states_with_teacher.jsonl")
    ap.add_argument("--accessibility-records", default="data/singapore_accessibility/records.jsonl")
    ap.add_argument("--split-manifest", default="data/singapore_accessibility/split_manifest.json")
    ap.add_argument("--legacy-split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--config", default="configs/student_s8.yaml")
    ap.add_argument("--output", default="outputs/student_s8")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs, for pipeline smoke")
    args = ap.parse_args()

    # freeze step 15: S8 output must never land inside a frozen release
    assert_not_frozen_output(args.output)

    cfg = load_yaml(args.config)
    s_cfg = cfg.get("student", {})
    t_cfg = cfg.get("training", {})
    l_cfg = cfg.get("loss", {})
    j_cfg = load_yaml(args.joint_config)
    combos = j_cfg.get("joint_combinations", [])

    seed = args.seed if args.seed is not None else t_cfg.get("seed", 42)
    set_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- splits ----
    s8_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    s8_persona_split = s8_manifest["persona_split"]
    s8_split_of = {}
    for sname in ("train", "val", "test"):
        for pid in s8_persona_split[sname]:
            s8_split_of[pid] = sname

    legacy_manifest = json.loads(Path(args.legacy_split_manifest).read_text(encoding="utf-8"))
    legacy_split_of = {}
    for sname in ("train", "val", "test"):
        for pid in legacy_manifest["personas"][sname]:
            legacy_split_of[pid] = sname

    def _lsplit(sample) -> str:
        return legacy_split_of.get(sample.persona_group_id or sample.state.persona.persona_id, "drop")

    base = _load(Path(args.base_dataset))
    joint = _load(Path(args.joint_dataset))
    joint = [s for s in joint if (resolve_combination(s.perturbation.joint_axes, combos) or {}).get("seen_in_training", True)]
    train_base = [s for s in base if _lsplit(s) == "train"]
    train_joint = [s for s in joint if _lsplit(s) == "train"]
    val_base = [s for s in base if _lsplit(s) == "val"]
    val_joint = [s for s in joint if _lsplit(s) == "val"]
    train_all = train_base + train_joint

    train_quads = load_mechanism_quadruplets(args.mechanism_dataset, split="train")
    val_quads = load_mechanism_quadruplets(args.mechanism_dataset, split="val")
    test_quads = load_mechanism_quadruplets(args.mechanism_dataset, split="test")
    assert all(q.split == "train" for q in train_quads)
    assert all(q.split == "val" for q in val_quads)
    assert len(test_quads) == 16, f"unexpected test quadruplet count: {len(test_quads)}"
    test_quads = []
    n_quad = len(train_quads)

    # ---- S8 accessibility states (TRAIN/VAL only; test split asserted absent) ----
    acc_targets = _load(Path(args.accessibility_dataset))
    train_acc = [s for s in acc_targets if s8_split_of.get(s.persona_group_id) == "train"]
    val_acc = [s for s in acc_targets if s8_split_of.get(s.persona_group_id) == "val"]
    test_acc = [s for s in acc_targets if s8_split_of.get(s.persona_group_id) == "test"]
    assert not test_acc, "S8 test-split targets leaked into the training script"
    n_acc = len(train_acc)
    print(f"accessibility targets: train={n_acc} val={len(val_acc)} (test excluded by assertion)")

    # ---- S8 extractor: frozen S7 stats + new fields fit on S8 TRAIN states only ----
    acc_records = []
    for line in Path(args.accessibility_records).read_text(encoding="utf-8").splitlines():
        if line.strip():
            acc_records.append(json.loads(line))
    train_states = [AggregatedTeacherTarget.model_validate(
        {"sample_id": r["sample_id"], "persona_group_id": r["persona_id"],
         "state": r["state"], "perturbation": {"axis": "accessibility", "level": 0.0},
         "teacher_aggregate": {"mode_probabilities": {"car": 1.0}, "selected_mode": "car",
                               "departure_time_shift_min": 0.0, "confidence_mean": 0.0},
         "aggregation_metadata": {"k": 1, "prompt_version": "n/a", "model": "n/a"}}
    ).state for r in acc_records if s8_split_of.get(r["persona_id"]) == "train"]
    ck = torch.load(FROZEN_RELEASE_CKPT, map_location=device)
    extractor = S8FeatureExtractor.from_s7_and_train(ck["extractor_state"], train_states)
    s8_spec = extractor.spec
    print(f"S8 extractor: n_alt_num={s8_spec['n_alt_num']} (6 frozen S7 stats + 6 fit on S8 train)")

    bs = t_cfg.get("batch_size", 32)
    quad_bs = t_cfg.get("mechanism_batch_size", 8)

    quad_loader = DataLoader(MechanismQuadrupletDataset(train_quads, extractor),
                             batch_size=quad_bs, shuffle=True, collate_fn=collate_quadruplets)
    val_quad_loader = DataLoader(MechanismQuadrupletDataset(val_quads, extractor),
                                 batch_size=quad_bs, shuffle=False, collate_fn=collate_quadruplets)
    val_elast_loader = DataLoader(
        CounterfactualPairDataset(make_counterfactual_pairs(val_base, build_baseline_index(val_base)), extractor),
        batch_size=bs, shuffle=False, collate_fn=collate_pairs)
    val_het_loader = DataLoader(PersonaContrastPairDataset(make_persona_contrast_pairs(val_base + val_joint), extractor),
                                batch_size=bs, shuffle=False, collate_fn=collate_contrast_pairs)
    val_single_loader = DataLoader(AggregatedTeacherDataset(val_base, extractor),
                                   batch_size=bs, shuffle=False, collate_fn=collate_batch)
    val_joint_loader = DataLoader(AggregatedTeacherDataset(val_joint, extractor),
                                  batch_size=bs, shuffle=False, collate_fn=collate_batch)
    val_acc_loader = DataLoader(AggregatedTeacherDataset(val_acc, extractor),
                                batch_size=bs, shuffle=False, collate_fn=collate_batch)

    model = TravelerStudentS8.from_s7_weights(FROZEN_RELEASE_CKPT, s8_spec).to(device)
    n_params = model.count_parameters()
    print(f"device={device} trainable_params={n_params} arch={ARCH_VERSION} (init from frozen S7-W3)")

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("learning_rate", 0.000125),
                                 weight_decay=t_cfg.get("weight_decay", 0.0001))

    base_index = build_baseline_index(train_base)
    legacy_pairs = make_counterfactual_pairs(train_base, base_index)
    joint_pairs = [(base_index[s.baseline_sample_id], s) for s in train_joint
                   if s.baseline_sample_id in base_index]
    het_pairs = make_persona_contrast_pairs(train_all)

    acc_per_epoch = t_cfg.get("accessibility_per_epoch", 52)

    if args.smoke:
        train_acc = train_acc[:8]
        train_quads = train_quads[:6]
        legacy_pairs = legacy_pairs[:8]
        joint_pairs = joint_pairs[:4]
        het_pairs = het_pairs[:4]
        n_quad = len(train_quads)
        acc_per_epoch = 8
        t_cfg["max_epochs"] = 2
        t_cfg["patience"] = 10

    max_epochs = t_cfg.get("max_epochs", 120)
    patience = t_cfg.get("patience", 15)
    het_per_epoch = t_cfg.get("het_pairs_per_epoch", n_quad)
    legacy_per_epoch = 2 * acc_per_epoch
    joint_per_epoch = acc_per_epoch

    def _sample(pairs, k):
        k = min(k, len(pairs))
        return random.sample(pairs, k) if k else []

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        run = {"static": 0.0, "mechanism": 0.0, "broken": 0.0, "legacy": 0.0,
               "joint": 0.0, "het": 0.0, "accessibility": 0.0, "n_quad": 0}
        _quad_step(model, quad_loader, device, l_cfg, optimizer, run)

        legacy_l = DataLoader(CounterfactualPairDataset(_sample(legacy_pairs, legacy_per_epoch), extractor),
                              batch_size=bs, shuffle=True, collate_fn=collate_pairs)
        _pair_step(model, legacy_l, device, l_cfg, optimizer, run, "legacy")

        joint_l = DataLoader(CounterfactualPairDataset(_sample(joint_pairs, joint_per_epoch), extractor),
                             batch_size=bs, shuffle=True, collate_fn=collate_pairs)
        _pair_step(model, joint_l, device, l_cfg, optimizer, run, "joint")

        het_l = DataLoader(PersonaContrastPairDataset(_sample(het_pairs, het_per_epoch), extractor),
                           batch_size=bs, shuffle=True, collate_fn=collate_contrast_pairs)
        _het_step(model, het_l, device, l_cfg, optimizer, run)

        acc_l = DataLoader(AggregatedTeacherDataset(_sample(train_acc, acc_per_epoch), extractor),
                           batch_size=bs, shuffle=True, collate_fn=collate_batch)
        _acc_step(model, acc_l, device, l_cfg, optimizer, run)

        val_total = _val_loss(model, val_quad_loader, val_elast_loader, val_het_loader,
                              val_acc_loader, device, l_cfg)
        val_legacy = evaluate(model, val_single_loader, device)
        val_joint_m = evaluate(model, val_joint_loader, device)
        val_acc_m = evaluate(model, val_acc_loader, device)
        val_gap = val_mechanism_gap(model, val_quads, extractor, device)
        mean_gap = sum(v["G_nat"] + v["G_broken"] + v["G_med"] for v in val_gap.values()) / max(1, 3 * len(val_gap))

        history.append({
            "epoch": epoch,
            "train_mechanism_loss": round(run["mechanism"] / max(1, run["n_quad"]), 6),
            "train_broken_loss": round(run["broken"] / max(1, run["n_quad"]), 6),
            "train_accessibility_loss": round(run["accessibility"] / max(1, acc_per_epoch), 6),
            "val_total_loss": round(val_total, 6),
            "val_legacy_kl": round(val_legacy["kl"], 4),
            "val_legacy_acc": round(val_legacy["mode_accuracy"], 4),
            "val_seen_joint_kl": round(val_joint_m["kl"], 4),
            "val_accessibility_kl": round(val_acc_m["kl"], 4),
            "val_mean_mechanism_gap": round(mean_gap, 6),
        })
        if val_total < best_val - 1e-6:
            best_val = val_total
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} val_total={val_total:.4f} acc_kl={val_acc_m['kl']:.3f} "
                  f"legacy_kl={val_legacy['kl']:.3f} joint_kl={val_joint_m['kl']:.3f} "
                  f"mech_gap={mean_gap:.4f} best_epoch={best_epoch}")
        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    runtime = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    final_gap = val_mechanism_gap(model, val_quads, extractor, device)
    final_legacy = evaluate(model, val_single_loader, device)
    final_joint = evaluate(model, val_joint_loader, device)
    final_acc = evaluate(model, val_acc_loader, device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state": best_state or {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_spec": s8_spec,
        "extractor_state": extractor.state_dict(),
        "config": s_cfg,
        "arch_version": ARCH_VERSION,
        "init_checkpoint": str(FROZEN_RELEASE_CKPT),
        "lambda_mechanism": l_cfg.get("lambda_mechanism", 0.0),
        "lambda_broken": l_cfg.get("lambda_broken", 0.0),
        "lambda_accessibility": l_cfg.get("lambda_accessibility", 0.0),
    }, out / "checkpoints" / "best.pt")

    (out / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    val_metrics = {
        "variant": out.name,
        "arch_version": ARCH_VERSION,
        "parameters": n_params,
        "init_from": "releases/s7_w3_generic_core_v1/checkpoint/model.pt (FROZEN)",
        "learning_rate": t_cfg.get("learning_rate", 0.000125),
        "replay_ratio": t_cfg.get("replay_ratio", "2:1:1:1"),
        "n_train_accessibility": n_acc,
        "best_epoch": best_epoch,
        "runtime_seconds": round(runtime, 2),
        "val_mechanism_gap": final_gap,
        "val_legacy": final_legacy,
        "val_seen_joint": final_joint,
        "val_accessibility": final_acc,
        "selection_note": "model selection uses val only; the S8 test split (unseen personas + unseen ODs) was never loaded",
    }
    (out / "val_metrics.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nbest_epoch={best_epoch} runtime={runtime:.1f}s")
    print(json.dumps({"val_accessibility": final_acc, "val_legacy": final_legacy,
                      "val_seen_joint": final_joint, "val_mechanism_gap": final_gap},
                     ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
