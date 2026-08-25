#!/usr/bin/env python
"""S5 multi-axis joint fine-tune: S3-C student -> S5-Joint student.

Fine-tunes the frozen S3-C checkpoint on the MERGED dataset
(S3 single-axis + S5 SEEN joint combos) using the same v0.2-C loss
(L_action + L_distribution + decomposed elasticity + heterogeneity). The joint
states are ordinary counterfactuals linked to their baselines, so the existing
elasticity pairing already supervises the baseline -> joint response (the
design's L_joint), and the static distribution loss covers the joint targets.

The S3-C feature extractor is FROZEN (joint states introduce no new feature
categories), and the S3-C persona holdout is REUSED verbatim so M0/M1/M2 share
the same unseen test personas.

Usage:
    python scripts/train_student_s5_joint.py \
        --base-dataset data/student_v0_3_s3/aggregated_teacher_dataset.jsonl \
        --joint-dataset data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl \
        --init-checkpoint outputs/student_v0_3_s3_c/checkpoints/best.pt \
        --split-manifest outputs/student_v0_3_s3_c/split_manifest.json \
        --config configs/student_s5_joint.yaml \
        --joint-config configs/joint_sampling.yaml \
        --output outputs/student_s5_joint_m1
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
    persona_breakdown,
    counterfactual_sign_agreement,
    AggregatedTeacherDataset,
    CounterfactualPairDataset,
    collate_batch,
    collate_pairs,
    build_baseline_index,
    make_counterfactual_pairs,
    make_persona_contrast_pairs,
    PersonaContrastPairDataset,
    collate_contrast_pairs,
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
def evaluate(model, loader, device, loss_cfg) -> dict:
    model.eval()
    n = 0
    sum_ce = 0.0
    sum_kl = 0.0
    sum_l1 = 0.0
    correct = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred_idx = out["mode_probabilities"].argmax(dim=1)
        correct += (pred_idx == batch["target_mode_idx"]).sum().item()
        logp = torch.log(out["mode_probabilities"].clamp(min=1e-8))
        sum_ce += torch.nn.functional.nll_loss(logp, batch["target_mode_idx"], reduction="sum").item()
        sum_kl += kl_divergence(batch["target_probs"], out["mode_probabilities"]).sum().item()
        sum_l1 += (batch["target_probs"] - out["mode_probabilities"]).abs().sum(dim=1).sum().item()
        n += batch["target_probs"].shape[0]
    return {
        "mode_accuracy": correct / n if n else 0.0,
        "cross_entropy": sum_ce / n if n else 0.0,
        "kl": sum_kl / n if n else 0.0,
        "probability_l1": sum_l1 / n if n else 0.0,
    }


def _predict_probs(sample, model, extractor, device) -> dict[str, float]:
    feats = extractor.encode(sample.state)
    batch = collate_batch([{
        "global_cat": torch.tensor(feats["global_cat"], dtype=torch.long),
        "global_num": torch.tensor(feats["global_num"], dtype=torch.float32),
        "alt_mode_idx": torch.tensor(feats["alt_mode_idx"], dtype=torch.long),
        "alt_num": torch.tensor(feats["alt_num"], dtype=torch.float32),
        "alt_mask": torch.tensor(feats["alt_available"], dtype=torch.float32),
        "target_probs": torch.zeros(len(feats["alt_available"])),
        "target_mode_idx": torch.zeros((), dtype=torch.long),
        "target_departure": torch.zeros(()),
    }])
    batch = {k: v.to(device) for k, v in batch.items()}
    s_out = model(batch)["mode_probabilities"][0].cpu().numpy()
    return {alt.mode: float(s_out[i]) for i, alt in enumerate(sample.state.alternatives) if alt.available}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-dataset", default="data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--init-checkpoint", default="outputs/student_v0_3_s3_c/checkpoints/best.pt")
    ap.add_argument("--split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--config", default="configs/student_s5_joint.yaml")
    ap.add_argument("--output", default="outputs/student_s5_joint_m1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
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

    base = _load(Path(args.base_dataset))
    joint = _load(Path(args.joint_dataset))
    # keep only SEEN joint combos (the unseen combo is the compositional holdout)
    joint = [s for s in joint if (resolve_combination(s.perturbation.joint_axes, combos) or {}).get("seen_in_training", True)]
    print(f"base={len(base)} joint(seen)={len(joint)}")

    samples = base + joint
    # every sample must carry a persona_group_id for the manifest split
    for s in samples:
        if s.persona_group_id is None:
            s.persona_group_id = s.state.persona.persona_id

    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    persona_sets = manifest["personas"]
    split_of = {}
    for split_name in ("train", "val", "test"):
        for pid in persona_sets[split_name]:
            split_of[pid] = split_name
    train = [s for s in samples if split_of.get(s.persona_group_id) == "train"]
    val = [s for s in samples if split_of.get(s.persona_group_id) == "val"]
    test = [s for s in samples if split_of.get(s.persona_group_id) == "test"]
    print(f"split train={len(train)} val={len(val)} test={len(test)}")

    # frozen extractor from S3-C checkpoint
    ckpt = torch.load(Path(args.init_checkpoint), map_location=device)
    extractor = FeatureExtractor().from_state_dict(ckpt["extractor_state"])

    train_elast = make_counterfactual_pairs(train, build_baseline_index(train))
    val_elast = make_counterfactual_pairs(val, build_baseline_index(val))
    test_elast = make_counterfactual_pairs(test, build_baseline_index(test))
    train_het = make_persona_contrast_pairs(train)
    val_het = make_persona_contrast_pairs(val)
    test_het = make_persona_contrast_pairs(test)
    print(f"pairs: elast train={len(train_elast)} val={len(val_elast)} test={len(test_elast)}; "
          f"het train={len(train_het)} val={len(val_het)} test={len(test_het)}")

    bs = t_cfg.get("batch_size", 32)
    train_elast_loader = DataLoader(CounterfactualPairDataset(train_elast, extractor), batch_size=bs, shuffle=True, collate_fn=collate_pairs)
    val_elast_loader = DataLoader(CounterfactualPairDataset(val_elast, extractor), batch_size=bs, shuffle=False, collate_fn=collate_pairs)
    train_het_loader = DataLoader(PersonaContrastPairDataset(train_het, extractor), batch_size=bs, shuffle=True, collate_fn=collate_contrast_pairs)
    val_het_loader = DataLoader(PersonaContrastPairDataset(val_het, extractor), batch_size=bs, shuffle=False, collate_fn=collate_contrast_pairs)
    val_single_loader = DataLoader(AggregatedTeacherDataset(val, extractor), batch_size=bs, shuffle=False, collate_fn=collate_batch)
    test_single_loader = DataLoader(AggregatedTeacherDataset(test, extractor), batch_size=bs, shuffle=False, collate_fn=collate_batch)

    model = TravelerStudent(s_cfg, ckpt["feature_spec"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    n_params = model.count_parameters()
    print(f"device={device} trainable_params={n_params} (init from S3-C checkpoint)")

    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("learning_rate", 0.0005), weight_decay=t_cfg.get("weight_decay", 0.0001))

    max_epochs = t_cfg.get("max_epochs", 120)
    patience = t_cfg.get("patience", 15)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = []

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        run = {"total": 0.0, "static": 0.0, "elasticity": 0.0, "heterogeneity": 0.0,
               "elasticity_direction": 0.0, "elasticity_magnitude": 0.0}
        n = 0
        for batch in train_elast_loader:
            batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
            base_out = model(batch["base"])
            cf_out = model(batch["cf"])
            losses = elast_pair_loss(base_out, cf_out, batch["base"], batch["cf"], l_cfg)
            loss = losses["total"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run["total"] += losses["total"].sum().item()
            run["static"] += losses["static"].sum().item()
            run["elasticity"] += losses["elasticity"].sum().item()
            run["elasticity_direction"] += losses["elasticity_direction"].sum().item()
            run["elasticity_magnitude"] += losses["elasticity_magnitude"].sum().item()
            n += batch["cf"]["target_probs"].shape[0]
        for batch in train_het_loader:
            batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
            a_out = model(batch["a"])
            b_out = model(batch["b"])
            losses = het_pair_loss(a_out, b_out, batch["a"], batch["b"], l_cfg)
            loss = losses["total"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run["total"] += losses["total"].sum().item()
            run["heterogeneity"] += losses["heterogeneity"].sum().item()
        denom = n if n else 1
        for k in ("total", "static", "elasticity", "elasticity_direction", "elasticity_magnitude"):
            run[k] /= denom
        run["heterogeneity"] /= max(1, len(train_het))

        val_total = _val_loss_sum(model, val_elast_loader, val_het_loader, device, l_cfg)
        val_metrics = evaluate(model, val_single_loader, device, l_cfg)
        history.append({
            "epoch": epoch, "train_total_loss": round(run["total"], 6),
            "train_elasticity_loss": round(run["elasticity"], 6),
            "train_heterogeneity_loss": round(run["heterogeneity"], 6),
            "val_total_loss": round(val_total, 6),
            "val_mode_accuracy": round(val_metrics["mode_accuracy"], 4),
            "val_probability_l1": round(val_metrics["probability_l1"], 4),
        })
        if val_total < best_val - 1e-6:
            best_val = val_total
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} train_total={run['total']:.4f} val_total={val_total:.4f} "
                  f"val_acc={val_metrics['mode_accuracy']:.3f} elast={run['elasticity']:.4f} "
                  f"het={run['heterogeneity']:.4f}")
        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    runtime = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_single_loader, device, l_cfg)
    sign_agree = counterfactual_sign_agreement(test, model, extractor, device)
    p_breakdown = persona_breakdown(test, model, extractor, device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state": best_state or {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_spec": ckpt["feature_spec"],
        "extractor_state": extractor.state_dict(),
        "config": s_cfg,
        "init_checkpoint": str(Path(args.init_checkpoint)),
    }, out / "checkpoints" / "best.pt")

    persona_sets_out = {sname: sorted({s.state.persona.persona_id for s in split}) for sname, split in [("train", train), ("val", val), ("test", test)]}
    split_manifest = {
        "strategy": "s3c_persona_holdout",
        "seed": seed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "pairs_elasticity": {"train": len(train_elast), "val": len(val_elast), "test": len(test_elast)},
        "pairs_heterogeneity": {"train": len(train_het), "val": len(val_het), "test": len(test_het)},
        "personas": persona_sets_out,
        "persona_overlap": bool(set(persona_sets_out["train"]) & set(persona_sets_out["test"])),
    }
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_metrics = {
        "test": test_metrics,
        "counterfactual_sign_agreement": sign_agree,
        "persona_breakdown": p_breakdown,
        "trainable_parameters": n_params,
        "device": device,
        "seed": seed,
        "best_epoch": best_epoch,
        "runtime_seconds": round(runtime, 2),
    }
    (out / "evaluation_metrics.json").write_text(json.dumps(eval_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("TEST METRICS")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"sign agreement = {sign_agree['mean_sign_agreement']} ({sign_agree['n_pairs']} pairs)")
    print(f"best_epoch={best_epoch} runtime={runtime:.1f}s")
    print(f"wrote {out}")
    return 0


@torch.no_grad()
def _val_loss_sum(model, elast_loader, het_loader, device, l_cfg) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in elast_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        base_out = model(batch["base"])
        cf_out = model(batch["cf"])
        losses = elast_pair_loss(base_out, cf_out, batch["base"], batch["cf"], l_cfg)
        total += losses["total"].sum().item()
        n += batch["cf"]["target_probs"].shape[0]
    for batch in het_loader:
        batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
        a_out = model(batch["a"])
        b_out = model(batch["b"])
        losses = het_pair_loss(a_out, b_out, batch["a"], batch["b"], l_cfg)
        total += losses["total"].sum().item()
    return total / n if n else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
