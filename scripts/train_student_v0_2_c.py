#!/usr/bin/env python
"""Train the v0.2-C Student: L_action + L_distribution + L_elasticity + L_heterogeneity.

v0.2-C adds behavioral-heterogeneity preservation on top of v0.2-B. Static
(action + distribution) and elasticity losses are identical to v0.2-B (same
split, same lambdas) so the three runs are directly comparable; the only
addition is L_heterogeneity, which supervises the BETWEEN-PERSONA response gap:
two personas in the same travel situation should differ by the same amount as
the teacher says they differ.

Usage:
    python scripts/train_student_v0_2_c.py \
        --config configs/student_v0_2_c.yaml \
        --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl \
        --output outputs/student_v0_2_c
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from traveler_distillation.config import load_yaml
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget
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
    group_aware_split,
    group_overlap,
)


def _load_dataset(path: Path) -> list[AggregatedTeacherTarget]:
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
    sum_mae = 0.0
    sum_rmse = 0.0
    sum_huber = 0.0
    sign_agree = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        pred_idx = out["mode_probabilities"].argmax(dim=1)
        correct += (pred_idx == batch["target_mode_idx"]).sum().item()

        logp = torch.log(out["mode_probabilities"].clamp(min=1e-8))
        ce = torch.nn.functional.nll_loss(logp, batch["target_mode_idx"], reduction="sum")
        sum_ce += ce.item()
        sum_kl += kl_divergence(batch["target_probs"], out["mode_probabilities"]).sum().item()
        sum_l1 += (batch["target_probs"] - out["mode_probabilities"]).abs().sum(dim=1).sum().item()

        dep = out["departure_time_shift_min"]
        tgt = batch["target_departure"]
        sum_mae += (dep - tgt).abs().sum().item()
        sum_rmse += ((dep - tgt) ** 2).sum().item()
        sum_huber += torch.nn.functional.huber_loss(
            dep, tgt, delta=loss_cfg.get("huber_delta", 1.0), reduction="sum"
        ).item()

        e = loss_cfg.get("departure_sign_epsilon", 1.0)
        def cls(v):
            return (v > e).long() - (v < -e).long()
        sign_agree += (cls(dep) == cls(tgt)).sum().item()
        n += batch["target_probs"].shape[0]

    return {
        "mode_accuracy": correct / n if n else 0.0,
        "cross_entropy": sum_ce / n if n else 0.0,
        "kl": sum_kl / n if n else 0.0,
        "probability_l1": sum_l1 / n if n else 0.0,
        "departure_mae": sum_mae / n if n else 0.0,
        "departure_rmse": (sum_rmse / n) ** 0.5 if n else 0.0,
        "departure_huber": sum_huber / n if n else 0.0,
        "departure_sign_agreement": sign_agree / n if n else 0.0,
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
    return {
        alt.mode: float(s_out[i])
        for i, alt in enumerate(sample.state.alternatives)
        if alt.available
    }


def counterfactual_preview(samples, model, extractor, device) -> dict:
    """Teacher vs Student DeltaP per counterfactual group (evaluation only)."""
    model.eval()
    by_id = {s.sample_id: s for s in samples}
    by_group: dict[str, list] = {}
    for s in samples:
        gid = s.counterfactual_group_id
        if gid:
            by_group.setdefault(gid, []).append(s)

    deltas = []
    with torch.no_grad():
        for gid, members in by_group.items():
            baseline = None
            for s in members:
                bid = s.baseline_sample_id
                if bid and bid in by_id:
                    baseline = by_id[bid]
                    break
            if baseline is None:
                continue
            t_base = baseline.teacher_aggregate.mode_probabilities
            s_base = _predict_probs(baseline, model, extractor, device)
            for s in members:
                if s.perturbation.axis == "baseline":
                    continue
                t_cf = s.teacher_aggregate.mode_probabilities
                s_cf = _predict_probs(s, model, extractor, device)
                modes = set(t_base) | set(t_cf) | set(s_base) | set(s_cf)
                dt = {m: t_cf.get(m, 0.0) - t_base.get(m, 0.0) for m in modes}
                ds = {m: s_cf.get(m, 0.0) - s_base.get(m, 0.0) for m in modes}
                mad = sum(abs(dt[m] - ds[m]) for m in modes) / len(modes) if modes else 0.0
                deltas.append(
                    {
                        "group_id": gid,
                        "sample_id": s.sample_id,
                        "axis": s.perturbation.axis,
                        "level": s.perturbation.level,
                        "teacher_delta_p": dt,
                        "student_delta_p": ds,
                        "abs_delta_p_diff": round(mad, 4),
                    }
                )

    mean_abs = sum(d["abs_delta_p_diff"] for d in deltas) / len(deltas) if deltas else 0.0
    return {"mean_abs_delta_p_diff": round(mean_abs, 4), "per_group": deltas}


def heterogeneity_preview(pairs, model, extractor, device) -> dict:
    """Teacher vs Student between-persona gap (evaluation only)."""
    model.eval()
    deltas = []
    with torch.no_grad():
        for a, b in pairs:
            t_a = a.teacher_aggregate.mode_probabilities
            t_b = b.teacher_aggregate.mode_probabilities
            s_a = _predict_probs(a, model, extractor, device)
            s_b = _predict_probs(b, model, extractor, device)
            modes = set(t_a) | set(t_b) | set(s_a) | set(s_b)
            dt = {m: t_a.get(m, 0.0) - t_b.get(m, 0.0) for m in modes}
            ds = {m: s_a.get(m, 0.0) - s_b.get(m, 0.0) for m in modes}
            mad = sum(abs(dt[m] - ds[m]) for m in modes) / len(modes) if modes else 0.0
            deltas.append(
                {
                    "persona_a": a.state.persona.persona_id,
                    "persona_b": b.state.persona.persona_id,
                    "axis": a.perturbation.axis,
                    "level": a.perturbation.level,
                    "abs_delta_p_diff": round(mad, 4),
                }
            )

    mean_abs = sum(d["abs_delta_p_diff"] for d in deltas) / len(deltas) if deltas else 0.0
    return {"mean_abs_delta_p_diff": round(mean_abs, 4), "per_pair": deltas}


def static_losses(out, batch, l_cfg) -> dict:
    return student_loss(
        out, batch["target_probs"], batch["target_mode_idx"], batch["target_departure"],
        lambda_action=l_cfg["lambda_action"], lambda_distribution=l_cfg["lambda_distribution"],
        lambda_departure=l_cfg["lambda_departure"], huber_delta=l_cfg.get("huber_delta", 1.0),
    )


def elast_pair_loss(base_out, cf_out, base_batch, cf_batch, l_cfg) -> dict:
    """Static losses on both states + (decomposable) elasticity loss."""
    base_l = static_losses(base_out, base_batch, l_cfg)
    cf_l = static_losses(cf_out, cf_batch, l_cfg)
    teacher_delta = cf_batch["target_probs"] - base_batch["target_probs"]
    student_delta = cf_out["mode_probabilities"] - base_out["mode_probabilities"]
    mask = cf_batch["alt_mask"]

    elast_l1 = elasticity_l1(teacher_delta, student_delta, mask=mask)
    elast_dir = elasticity_direction_loss(
        teacher_delta, student_delta, mask=mask,
        margin=l_cfg.get("direction_margin", 0.0),
    )
    elast_mag = elasticity_magnitude_loss(teacher_delta, student_delta, mask=mask)

    elast_total = (
        l_cfg.get("lambda_elasticity", 0.0) * elast_l1
        + l_cfg.get("lambda_direction", 0.0) * elast_dir
        + l_cfg.get("lambda_magnitude", 0.0) * elast_mag
    )
    return {
        "total": base_l["total"] + cf_l["total"] + elast_total,
        "static": base_l["total"] + cf_l["total"],
        "elasticity": elast_l1,
        "elasticity_direction": elast_dir,
        "elasticity_magnitude": elast_mag,
    }


def het_pair_loss(a_out, b_out, a_batch, b_batch, l_cfg) -> dict:
    teacher_diff = a_batch["target_probs"] - b_batch["target_probs"]
    student_diff = a_out["mode_probabilities"] - b_out["mode_probabilities"]
    union_mask = (a_batch["alt_mask"] + b_batch["alt_mask"]).clamp(min=0.0, max=1.0)
    het = heterogeneity_l1(teacher_diff, student_diff, mask=union_mask)
    return {"total": l_cfg.get("lambda_heterogeneity", 1.0) * het, "heterogeneity": het}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/student_v0_2_c.yaml")
    ap.add_argument("--dataset", default="data/student_v0_2_a/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="outputs/student_v0_2_c")
    ap.add_argument("--tiny-overfit", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    s_cfg = cfg.get("student", {})
    t_cfg = cfg.get("training", {})
    l_cfg = cfg.get("loss", {})
    sp_cfg = cfg.get("split", {})

    seed = args.seed if args.seed is not None else t_cfg.get("seed", 42)
    set_seed(seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    samples = _load_dataset(Path(args.dataset))
    print(f"loaded {len(samples)} aggregated samples")

    if args.tiny_overfit:
        samples = samples[:64]

    group_field = sp_cfg.get("group_field", "split_group_id")
    if group_field == "split_group_id":
        for s in samples:
            if s.split_group_id is None:
                s.split_group_id = s.counterfactual_group_id
    elif group_field == "persona_group_id":
        # persona-holdout: every state of a persona stays in one split
        for s in samples:
            if s.persona_group_id is None:
                s.persona_group_id = s.state.persona.persona_id

    train, val, test = group_aware_split(
        samples,
        group_field=group_field,
        train_ratio=sp_cfg.get("train_ratio", 0.7),
        val_ratio=sp_cfg.get("val_ratio", 0.15),
        test_ratio=sp_cfg.get("test_ratio", 0.15),
        seed=seed,
    )
    assert not group_overlap(train, val, test, group_field=group_field)

    extractor = FeatureExtractor().fit([s.state for s in train])

    train_elast = make_counterfactual_pairs(train, build_baseline_index(train))
    val_elast = make_counterfactual_pairs(val, build_baseline_index(val))
    test_elast = make_counterfactual_pairs(test, build_baseline_index(test))
    train_het = make_persona_contrast_pairs(train)
    val_het = make_persona_contrast_pairs(val)
    test_het = make_persona_contrast_pairs(test)
    print(f"pairs: elast train={len(train_elast)} val={len(val_elast)} test={len(test_elast)}; "
          f"het train={len(train_het)} val={len(val_het)} test={len(test_het)}")

    bs = t_cfg.get("batch_size", 32)
    train_elast_loader = DataLoader(
        CounterfactualPairDataset(train_elast, extractor), batch_size=bs,
        shuffle=True, collate_fn=collate_pairs,
    )
    val_elast_loader = DataLoader(
        CounterfactualPairDataset(val_elast, extractor), batch_size=bs,
        shuffle=False, collate_fn=collate_pairs,
    )
    train_het_loader = DataLoader(
        PersonaContrastPairDataset(train_het, extractor), batch_size=bs,
        shuffle=True, collate_fn=collate_contrast_pairs,
    )
    val_het_loader = DataLoader(
        PersonaContrastPairDataset(val_het, extractor), batch_size=bs,
        shuffle=False, collate_fn=collate_contrast_pairs,
    )
    val_single_loader = DataLoader(
        AggregatedTeacherDataset(val, extractor), batch_size=bs,
        shuffle=False, collate_fn=collate_batch,
    )
    test_single_loader = DataLoader(
        AggregatedTeacherDataset(test, extractor), batch_size=bs,
        shuffle=False, collate_fn=collate_batch,
    )

    model = TravelerStudent(s_cfg, extractor.spec).to(device)
    n_params = model.count_parameters()
    print(f"device={device} trainable_params={n_params}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=t_cfg.get("learning_rate", 0.001),
        weight_decay=t_cfg.get("weight_decay", 0.0001),
    )

    max_epochs = t_cfg.get("max_epochs", 200)
    patience = t_cfg.get("patience", 20)
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

        history.append(
            {
                "epoch": epoch,
                "train_total_loss": round(run["total"], 6),
                "train_static_loss": round(run["static"], 6),
                "train_elasticity_loss": round(run["elasticity"], 6),
                "train_elasticity_direction_loss": round(run["elasticity_direction"], 6),
                "train_elasticity_magnitude_loss": round(run["elasticity_magnitude"], 6),
                "train_heterogeneity_loss": round(run["heterogeneity"], 6),
                "val_total_loss": round(val_total, 6),
                "val_mode_accuracy": round(val_metrics["mode_accuracy"], 4),
                "val_probability_l1": round(val_metrics["probability_l1"], 4),
            }
        )

        if val_total < best_val - 1e-6:
            best_val = val_total
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} train_total={run['total']:.4f} "
                  f"val_total={val_total:.4f} val_acc={val_metrics['mode_accuracy']:.3f} "
                  f"elast={run['elasticity']:.4f} dir={run['elasticity_direction']:.4f} "
                  f"mag={run['elasticity_magnitude']:.4f} het={run['heterogeneity']:.4f}")

        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    runtime = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_single_loader, device, l_cfg)
    cf_preview = counterfactual_preview(test, model, extractor, device)
    het_preview = heterogeneity_preview(test_het, model, extractor, device)
    sign_agree = counterfactual_sign_agreement(test, model, extractor, device)
    p_breakdown = persona_breakdown(test, model, extractor, device)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    if best_state is not None:
        torch.save(
            {
                "model_state": best_state,
                "feature_spec": extractor.spec,
                "extractor_state": extractor.state_dict(),
                "config": s_cfg,
            },
            out / "checkpoints" / "best.pt",
        )

    persona_sets = {
        split_name: sorted({s.state.persona.persona_id for s in split})
        for split_name, split in [("train", train), ("val", val), ("test", test)]
    }
    split_manifest = {
        "strategy": f"group_field={group_field}",
        "seed": seed,
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "pairs_elasticity": {"train": len(train_elast), "val": len(val_elast), "test": len(test_elast)},
        "pairs_heterogeneity": {"train": len(train_het), "val": len(val_het), "test": len(test_het)},
        "split_group_overlap": group_overlap(train, val, test, group_field=group_field),
        "personas": persona_sets,
        "persona_overlap": bool(
            set(persona_sets["train"]) & set(persona_sets["test"])
            or set(persona_sets["train"]) & set(persona_sets["val"])
            or set(persona_sets["val"]) & set(persona_sets["test"])
        ),
    }
    (out / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    eval_metrics = {
        "test": test_metrics,
        "counterfactual_preview": cf_preview,
        "heterogeneity_preview": het_preview,
        "counterfactual_sign_agreement": sign_agree,
        "persona_breakdown": p_breakdown,
        "trainable_parameters": n_params,
        "device": device,
        "seed": seed,
        "best_epoch": best_epoch,
        "runtime_seconds": round(runtime, 2),
    }
    (out / "evaluation_metrics.json").write_text(json.dumps(eval_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = _build_report(cfg, split_manifest, history, test_metrics, cf_preview, het_preview, sign_agree, p_breakdown, n_params, device, seed, best_epoch, runtime)
    (out / "student_v0_2_c_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print("TEST METRICS")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"\nbest_epoch={best_epoch} runtime={runtime:.1f}s params={n_params} device={device}")
    print(f"counterfactual mean |delta_P_T - delta_P_S| = {cf_preview['mean_abs_delta_p_diff']}")
    print(f"counterfactual sign agreement = {sign_agree['mean_sign_agreement']} ({sign_agree['n_pairs']} pairs)")
    print(f"heterogeneity  mean |delta_P_T - delta_P_S| = {het_preview['mean_abs_delta_p_diff']} ({len(het_preview['per_pair'])} pairs)")
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


def _build_report(cfg, split, history, test, cf, het, sign_agree, p_breakdown, n_params, device, seed, best_epoch, runtime) -> str:
    l = []
    a = l.append
    a("# Student Distillation v0.2-C Report\n")
    a("## Model\n")
    a(f"- architecture: lightweight variable-choice-set MLP student")
    a(f"- trainable parameters: {n_params}")
    a(f"- device: {device}, seed: {seed}")
    lc = cfg["loss"]
    a(f"- loss: L_action + L_distribution + decomposed L_elasticity + lambda_H * L_heterogeneity "
      f"(lambda_A={lc['lambda_action']}, lambda_D={lc['lambda_distribution']}, "
      f"lambda_t={lc['lambda_departure']}, lambda_E={lc.get('lambda_elasticity', 0.0)}, "
      f"lambda_dir={lc.get('lambda_direction', 0.0)}, lambda_mag={lc.get('lambda_magnitude', 0.0)}, "
      f"lambda_H={lc.get('lambda_heterogeneity', 1.0)})\n")
    a("## Split\n")
    a(f"- strategy = {split['strategy']}")
    a(f"- train/val/test = {split['counts']}")
    a(f"- elasticity pairs = {split['pairs_elasticity']}")
    a(f"- heterogeneity pairs = {split['pairs_heterogeneity']}")
    a(f"- split group overlap = {split['split_group_overlap']} (must be False)")
    a(f"- persona overlap = {split['persona_overlap']} "
      f"({'holdout: unseen personas in test' if not split['persona_overlap'] else 'reported for transparency'})\n")
    a("## Training\n")
    a(f"- best epoch: {best_epoch}, runtime: {runtime:.1f}s")
    if history:
        last = history[-1]
        a(f"- final train total loss: {last['train_total_loss']}")
        a(f"- final train elasticity loss (L1): {last['train_elasticity_loss']}")
        a(f"- final train elasticity direction loss: {last['train_elasticity_direction_loss']}")
        a(f"- final train elasticity magnitude loss: {last['train_elasticity_magnitude_loss']}")
        a(f"- final train heterogeneity loss: {last['train_heterogeneity_loss']}")
        a(f"- best val total loss: {min(h['val_total_loss'] for h in history)}\n")
    a("## Test Metrics\n")
    for k, v in test.items():
        a(f"- {k}: {v}")
    a("\n## Counterfactual / Elasticity Preview\n")
    a(f"- mean |delta_P_T - delta_P_S| = {cf['mean_abs_delta_p_diff']} ({len(cf['per_group'])} pairs)")
    a(f"- mean sign agreement = {sign_agree['mean_sign_agreement']} ({sign_agree['n_pairs']} pairs)\n")
    a("\n## Heterogeneity Preview (between-persona)\n")
    a(f"- mean |delta_P_T - delta_P_S| = {het['mean_abs_delta_p_diff']} ({len(het['per_pair'])} pairs)\n")
    a("\n## Per-Persona Test Breakdown\n")
    a("| persona | n | mode_acc | prob_L1 |")
    a("|---|---|---|---|")
    for row in p_breakdown:
        a(f"| {row['persona_id']} | {row['n_samples']} | {row['mode_accuracy']} | {row['probability_l1']} |")
    a("\n## Known Limitations\n")
    a("- development dataset (synthetic states, no real-world calibration)")
    a("- persona overlap = True (unseen-persona generalization is a separate future experiment)")
    a("- no MATSim")
    return "\n".join(l) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
