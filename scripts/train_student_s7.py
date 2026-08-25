#!/usr/bin/env python
"""S7 mechanism-aware fine-tune: C1 (S5 joint M2) -> S7 student.

Fine-tunes the frozen S5-Joint-M2 checkpoint with the S7 loss
    L_S7 = L_base + lambda_mechanism * L_mechanism + lambda_broken * L_broken
on a replay mix (S7 instruction §16):
    legacy single-axis pairs : seen-joint pairs : mechanism quadruplets = 2:1:1
plus one heterogeneity-pair pass per epoch (L_base component, fixed budget).

HARD CONSTRAINTS (user-mandated, non-negotiable):
  * Teacher targets are reused VERBATIM from S6 data — this script makes NO
    API calls and loads no targets other than the S6 quadruplet file + the
    existing S3/S5 aggregated datasets.
  * The final causal test set (split == "test" in the mechanism dataset) is
    NEVER loaded for training or validation. Only split == "train" quadruplets
    enter the mechanism loss; split == "val" quadruplets drive early stopping
    and model selection. This is asserted at load time.
  * The feature extractor is frozen from the C1 checkpoint (S6 audit states
    introduce no new feature categories).

Usage:
    python scripts/train_student_s7.py \
        --base-dataset data/student_v0_3_s3/aggregated_teacher_dataset.jsonl \
        --joint-dataset data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl \
        --joint-config configs/joint_sampling.yaml \
        --mechanism-dataset data/student_s7_mechanism/quadruplets.jsonl \
        --init-checkpoint outputs/student_s5_joint_m2/checkpoints/best.pt \
        --split-manifest outputs/student_v0_3_s3_c/split_manifest.json \
        --config configs/student_s7_w2.yaml \
        --output outputs/student_s7_w2
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

from traveler_distillation.config import load_yaml
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget
from traveler_distillation.generators import resolve_combination
from traveler_distillation.student import (
    FeatureExtractor,
    TravelerStudent,
    student_loss,
    elasticity_l1,
    elasticity_direction_loss,
    elasticity_magnitude_loss,
    heterogeneity_l1,
    kl_divergence,
    mechanism_fidelity_loss,
    broken_path_fidelity_loss,
    build_baseline_index,
    make_counterfactual_pairs,
    make_persona_contrast_pairs,
    CounterfactualPairDataset,
    collate_pairs,
    PersonaContrastPairDataset,
    collate_contrast_pairs,
    AggregatedTeacherDataset,
    collate_batch,
    load_mechanism_quadruplets,
    MechanismQuadrupletDataset,
    collate_quadruplets,
)


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
    """L_base (static) on all four members + the two S7 mechanism losses."""
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
    """Raw per-axis mechanism gaps on VAL quadruplets (selection signal, no test)."""
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


@torch.no_grad()
def _val_loss(model, val_quad_loader, val_elast_loader, val_het_loader, device, l_cfg) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in val_quad_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        out = {k: model(batch[k]) for k in ("A", "B", "C", "D")}
        losses = quadruplet_losses(out, batch, l_cfg)
        total += losses["total"].sum().item()
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
    return total / n if n else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-dataset", default="data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--mechanism-dataset", default="data/student_s7_mechanism/quadruplets.jsonl")
    ap.add_argument("--init-checkpoint", default="outputs/student_s5_joint_m2/checkpoints/best.pt")
    ap.add_argument("--split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--config", default="configs/student_s7_w2.yaml")
    ap.add_argument("--output", default="outputs/student_s7_w2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 2 epochs, for pipeline smoke")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    s_cfg = cfg.get("student", {})
    t_cfg = cfg.get("training", {})
    l_cfg = cfg.get("loss", {})
    j_cfg = load_yaml(args.joint_config)
    combos = j_cfg.get("joint_combinations", [])

    seed = args.seed if args.seed is not None else t_cfg.get("seed", 42)
    set_seed(seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- splits (S3-C persona holdout, verbatim) ----
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    persona_sets = manifest["personas"]
    split_of = {}
    for sname in ("train", "val", "test"):
        for pid in persona_sets[sname]:
            split_of[pid] = sname

    def _split_of(sample) -> str:
        return split_of.get(sample.persona_group_id or sample.state.persona.persona_id, "drop")

    base = _load(Path(args.base_dataset))
    joint = _load(Path(args.joint_dataset))
    joint = [s for s in joint if (resolve_combination(s.perturbation.joint_axes, combos) or {}).get("seen_in_training", True)]
    train_base = [s for s in base if _split_of(s) == "train"]
    train_joint = [s for s in joint if _split_of(s) == "train"]
    val_base = [s for s in base if _split_of(s) == "val"]
    val_joint = [s for s in joint if _split_of(s) == "val"]
    train_all = train_base + train_joint

    # ---- mechanism quadruplets: TRAIN only for the loss, VAL only for selection ----
    train_quads = load_mechanism_quadruplets(args.mechanism_dataset, split="train")
    val_quads = load_mechanism_quadruplets(args.mechanism_dataset, split="val")
    test_quads = load_mechanism_quadruplets(args.mechanism_dataset, split="test")
    assert all(q.split == "train" for q in train_quads)
    assert all(q.split == "val" for q in val_quads)
    # HARD CONSTRAINT: the causal test set must never reach training code paths
    # beyond this load (it is only materialized to prove it exists and is excluded).
    assert len(test_quads) == 16, f"unexpected test quadruplet count: {len(test_quads)}"
    test_quads = []  # drop the reference entirely; nothing below can touch it
    n_quad = len(train_quads)

    # ---- replay pairs ----
    base_index = build_baseline_index(train_base)
    legacy_pairs = make_counterfactual_pairs(train_base, base_index)
    joint_pairs = [(base_index[s.baseline_sample_id], s) for s in train_joint
                   if s.baseline_sample_id in base_index]
    het_pairs = make_persona_contrast_pairs(train_all)
    val_elast = make_counterfactual_pairs(val_base, build_baseline_index(val_base))
    val_het = make_persona_contrast_pairs(val_base + val_joint)
    print(f"base(train)={len(train_base)} joint(train)={len(train_joint)} "
          f"quads train={n_quad} val={len(val_quads)}")
    print(f"pairs: legacy={len(legacy_pairs)} joint={len(joint_pairs)} het={len(het_pairs)} "
          f"val_elast={len(val_elast)} val_het={len(val_het)}")

    # smoke mode: shrink everything, 2 epochs
    if args.smoke:
        train_quads = train_quads[:6]
        legacy_pairs = legacy_pairs[:8]
        joint_pairs = joint_pairs[:4]
        het_pairs = het_pairs[:4]
        n_quad = len(train_quads)
        t_cfg["max_epochs"] = 2
        t_cfg["patience"] = 10

    # ---- frozen extractor from C1 checkpoint ----
    ckpt = torch.load(Path(args.init_checkpoint), map_location=device)
    extractor = FeatureExtractor().from_state_dict(ckpt["extractor_state"])
    bs = t_cfg.get("batch_size", 32)
    quad_bs = t_cfg.get("mechanism_batch_size", 8)

    quad_loader = DataLoader(MechanismQuadrupletDataset(train_quads, extractor),
                             batch_size=quad_bs, shuffle=True, collate_fn=collate_quadruplets)
    val_quad_loader = DataLoader(MechanismQuadrupletDataset(val_quads, extractor),
                                 batch_size=quad_bs, shuffle=False, collate_fn=collate_quadruplets)
    val_elast_loader = DataLoader(CounterfactualPairDataset(val_elast, extractor),
                                  batch_size=bs, shuffle=False, collate_fn=collate_pairs)
    val_het_loader = DataLoader(PersonaContrastPairDataset(val_het, extractor),
                                batch_size=bs, shuffle=False, collate_fn=collate_contrast_pairs)
    val_single_loader = DataLoader(AggregatedTeacherDataset(val_base, extractor),
                                   batch_size=bs, shuffle=False, collate_fn=collate_batch)
    val_joint_loader = DataLoader(AggregatedTeacherDataset(val_joint, extractor),
                                  batch_size=bs, shuffle=False, collate_fn=collate_batch)

    model = TravelerStudent(s_cfg, ckpt["feature_spec"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    n_params = model.count_parameters()
    print(f"device={device} trainable_params={n_params} (init from C1={args.init_checkpoint})")

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("learning_rate", 0.000125),
                                 weight_decay=t_cfg.get("weight_decay", 0.0001))

    max_epochs = t_cfg.get("max_epochs", 120)
    patience = t_cfg.get("patience", 15)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []
    het_per_epoch = t_cfg.get("het_pairs_per_epoch", n_quad)
    legacy_per_epoch = 2 * n_quad
    joint_per_epoch = n_quad

    def _sample(pairs, k):
        k = min(k, len(pairs))
        return random.sample(pairs, k) if k else []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        run = {"static": 0.0, "mechanism": 0.0, "broken": 0.0, "legacy": 0.0,
               "joint": 0.0, "het": 0.0, "n_quad": 0}
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

        val_total = _val_loss(model, val_quad_loader, val_elast_loader, val_het_loader, device, l_cfg)
        val_legacy = evaluate(model, val_single_loader, device)
        val_joint_m = evaluate(model, val_joint_loader, device)
        val_gap = val_mechanism_gap(model, val_quads, extractor, device)
        mean_gap = sum(v["G_nat"] + v["G_broken"] + v["G_med"] for v in val_gap.values()) / max(1, 3 * len(val_gap))

        history.append({
            "epoch": epoch,
            "train_mechanism_loss": round(run["mechanism"] / max(1, run["n_quad"]), 6),
            "train_broken_loss": round(run["broken"] / max(1, run["n_quad"]), 6),
            "val_total_loss": round(val_total, 6),
            "val_legacy_kl": round(val_legacy["kl"], 4),
            "val_legacy_acc": round(val_legacy["mode_accuracy"], 4),
            "val_seen_joint_kl": round(val_joint_m["kl"], 4),
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
            print(f"epoch {epoch:3d} val_total={val_total:.4f} mech_gap={mean_gap:.4f} "
                  f"legacy_kl={val_legacy['kl']:.3f} joint_kl={val_joint_m['kl']:.3f} "
                  f"best_epoch={best_epoch}")
        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    runtime = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # final val metrics (selection input; test set NOT touched)
    final_gap = val_mechanism_gap(model, val_quads, extractor, device)
    final_legacy = evaluate(model, val_single_loader, device)
    final_joint = evaluate(model, val_joint_loader, device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state": best_state or {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_spec": ckpt["feature_spec"],
        "extractor_state": extractor.state_dict(),
        "config": s_cfg,
        "init_checkpoint": str(Path(args.init_checkpoint)),
        "lambda_mechanism": l_cfg.get("lambda_mechanism", 0.0),
        "lambda_broken": l_cfg.get("lambda_broken", 0.0),
    }, out / "checkpoints" / "best.pt")

    (out / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    val_metrics = {
        "variant": out.name,
        "lambda_mechanism": l_cfg.get("lambda_mechanism", 0.0),
        "lambda_broken": l_cfg.get("lambda_broken", 0.0),
        "learning_rate": t_cfg.get("learning_rate", 0.000125),
        "replay_ratio": t_cfg.get("replay_ratio", "2:1:1"),
        "n_train_quadruplets": n_quad,
        "best_epoch": best_epoch,
        "runtime_seconds": round(runtime, 2),
        "val_mechanism_gap": final_gap,
        "val_mean_mechanism_gap": round(
            sum(v["G_nat"] + v["G_broken"] + v["G_med"] for v in final_gap.values()) / max(1, 3 * len(final_gap)), 4),
        "val_legacy": final_legacy,
        "val_seen_joint": final_joint,
        "selection_note": "model selection uses val only; the causal test set was never loaded for train/val",
    }
    (out / "val_metrics.json").write_text(json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nbest_epoch={best_epoch} runtime={runtime:.1f}s")
    print(json.dumps({"val_mechanism_gap": final_gap, "val_legacy": final_legacy,
                      "val_seen_joint": final_joint}, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
