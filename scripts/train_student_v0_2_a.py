#!/usr/bin/env python
"""Train the v0.2-A Student baseline (L_action + L_distribution).

Usage:
    python scripts/train_student_v0_2_a.py \
        --config configs/student_v0_2_a.yaml \
        --dataset data/student_v0_2_a/aggregated_teacher_dataset.jsonl \
        --output outputs/student_v0_2_a
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
    persona_breakdown,
    counterfactual_sign_agreement,
    AggregatedTeacherDataset,
    collate_batch,
    group_aware_split,
    group_overlap,
)
from traveler_distillation.student.losses import kl_divergence


def _load_dataset(path: Path) -> list[AggregatedTeacherTarget]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(AggregatedTeacherTarget.model_validate(json.loads(line)))
    return samples


def _teacher_noise_reference(dataset_path: Path) -> dict:
    """Teacher sampling-noise reference from the repeat records.

    Computes the per-state mean pairwise L1 across the K repeats (the teacher's
    own same-state variance = the best a student could hope to reach). Falls
    back to the historical gateway measurement (0.0472 K3<->K5) when no repeat
    records are available next to the dataset.
    """
    repeats_path = dataset_path.parent / "repeat_records.jsonl"
    if not repeats_path.exists():
        return {"k3_vs_k5_mean_l1": 0.0472, "source": "historical_gateway"}

    from collections import defaultdict
    by_state: dict[str, list] = defaultdict(list)
    for line in repeats_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("action") is not None:
            by_state[rec["sample_id"]].append(rec["action"]["mode_probabilities"])

    pairwise = []
    for probs_list in by_state.values():
        if len(probs_list) < 2:
            continue
        l1s = []
        for i in range(len(probs_list)):
            for j in range(i + 1, len(probs_list)):
                modes = set(probs_list[i]) | set(probs_list[j])
                l1s.append(sum(abs(probs_list[i].get(m, 0.0) - probs_list[j].get(m, 0.0)) for m in modes))
        pairwise.append(sum(l1s) / len(l1s))

    mean_l1 = sum(pairwise) / len(pairwise) if pairwise else 0.0
    return {
        "mean_pairwise_l1": round(mean_l1, 4),
        "n_states": len(pairwise),
        "source": "dataset_repeat_records",
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, loss_cfg) -> dict:
    model.eval()
    eps = 1e-6
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
        sum_huber += torch.nn.functional.huber_loss(dep, tgt, delta=loss_cfg.get("huber_delta", 1.0), reduction="sum").item()

        # sign agreement: earlier / unchanged / later with epsilon tolerance
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


def _loss_sum(model, loader, device, loss_cfg) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            losses = student_loss(
                out,
                batch["target_probs"],
                batch["target_mode_idx"],
                batch["target_departure"],
                lambda_action=loss_cfg["lambda_action"],
                lambda_distribution=loss_cfg["lambda_distribution"],
                lambda_departure=loss_cfg["lambda_departure"],
                huber_delta=loss_cfg.get("huber_delta", 1.0),
            )
            total += losses["total"].sum().item()
            n += batch["target_probs"].shape[0]
    return total / n if n else 0.0


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


def counterfactual_preview(samples: list[AggregatedTeacherTarget], model, extractor, device) -> dict:
    """Teacher vs Student ΔP from baseline, per counterfactual group (evaluation only)."""
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
            # Resolve the shared baseline via baseline_sample_id linkage (the
            # baseline is NOT a member of the counterfactual group).
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/student_v0_2_a.yaml")
    ap.add_argument("--dataset", default="data/student_v0_2_a/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", default="outputs/student_v0_2_a")
    ap.add_argument("--tiny-overfit", action="store_true", help="train on a tiny subset to check overfit")
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
    noise_ref = _teacher_noise_reference(Path(args.dataset))

    if args.tiny_overfit:
        samples = samples[:16]

    # Split on the split_group_id key (persona::trip), which keeps a baseline
    # sample and ALL of its counterfactual curves in one split. Fall back to the
    # legacy counterfactual_group_id for older data files that predate the key.
    # persona_group_id selects a persona-holdout split (unseen personas in test).
    group_field = sp_cfg.get("group_field", "split_group_id")
    if group_field == "split_group_id":
        for s in samples:
            if s.split_group_id is None:
                s.split_group_id = s.counterfactual_group_id
    elif group_field == "persona_group_id":
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

    # normalization statistics from TRAIN split only
    extractor = FeatureExtractor().fit([s.state for s in train])

    train_ds = AggregatedTeacherDataset(train, extractor)
    val_ds = AggregatedTeacherDataset(val, extractor)
    test_ds = AggregatedTeacherDataset(test, extractor)

    batch_size = t_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

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
        train_losses = {"total": 0.0, "action": 0.0, "kl": 0.0}
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            losses = student_loss(
                out,
                batch["target_probs"],
                batch["target_mode_idx"],
                batch["target_departure"],
                lambda_action=l_cfg["lambda_action"],
                lambda_distribution=l_cfg["lambda_distribution"],
                lambda_departure=l_cfg["lambda_departure"],
                huber_delta=l_cfg.get("huber_delta", 1.0),
            )
            loss = losses["total"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for k in ("total", "action", "kl"):
                train_losses[k] += losses[k].sum().item()
            n += batch["target_probs"].shape[0]

        for k in train_losses:
            train_losses[k] /= n if n else 1

        val_total = _loss_sum(model, val_loader, device, l_cfg)
        val_metrics = evaluate(model, val_loader, device, l_cfg)

        history.append(
            {
                "epoch": epoch,
                "train_total_loss": round(train_losses["total"], 6),
                "train_action_loss": round(train_losses["action"], 6),
                "train_distribution_loss": round(train_losses["kl"], 6),
                "val_total_loss": round(val_total, 6),
                "val_action_loss": round(val_metrics["cross_entropy"] + l_cfg["lambda_departure"] * val_metrics["departure_huber"], 6),
                "val_distribution_loss": round(val_metrics["kl"], 6),
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
            print(f"epoch {epoch:3d} train_total={train_losses['total']:.4f} "
                  f"val_total={val_total:.4f} val_acc={val_metrics['mode_accuracy']:.3f} "
                  f"val_l1={val_metrics['probability_l1']:.4f}")

        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    runtime = time.time() - t0

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device, l_cfg)
    cf_preview = counterfactual_preview(test, model, extractor, device)
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
        "split_group_overlap": group_overlap(train, val, test, group_field=group_field),
        "counterfactual_group_overlap": group_overlap(
            train, val, test, group_field="counterfactual_group_id"
        ),
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
        "teacher_aggregation_noise_reference": noise_ref,
        "counterfactual_preview": cf_preview,
        "counterfactual_sign_agreement": sign_agree,
        "persona_breakdown": p_breakdown,
        "trainable_parameters": n_params,
        "device": device,
        "seed": seed,
        "best_epoch": best_epoch,
        "runtime_seconds": round(runtime, 2),
    }
    (out / "evaluation_metrics.json").write_text(json.dumps(eval_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = _build_report(cfg, split_manifest, history, test_metrics, cf_preview, sign_agree, p_breakdown, n_params, device, seed, best_epoch, runtime, noise_ref)
    (out / "student_v0_2_a_report.md").write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print("TEST METRICS")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"\nbest_epoch={best_epoch} runtime={runtime:.1f}s params={n_params} device={device}")
    print(f"counterfactual mean |delta_P_T - delta_P_S| = {cf_preview['mean_abs_delta_p_diff']}")
    print(f"counterfactual sign agreement = {sign_agree['mean_sign_agreement']} ({sign_agree['n_pairs']} pairs)")
    print(f"wrote {out}")
    return 0


def _build_report(cfg, split, history, test, cf, sign_agree, p_breakdown, n_params, device, seed, best_epoch, runtime, noise_ref) -> str:
    l = []
    a = l.append
    a("# Student Distillation v0.2-A Report\n")
    a("## Model\n")
    a(f"- architecture: lightweight variable-choice-set MLP student")
    a(f"- trainable parameters: {n_params}")
    a(f"- device: {device}, seed: {seed}")
    a(f"- loss: L = lambda_A * L_action + lambda_D * L_distribution (lambda_A={cfg['loss']['lambda_action']}, lambda_D={cfg['loss']['lambda_distribution']}, lambda_t={cfg['loss']['lambda_departure']})\n")
    a("## Split\n")
    a(f"- strategy = {split['strategy']}")
    a(f"- train/val/test = {split['counts']}")
    a(f"- split group overlap = {split['split_group_overlap']} (must be False)")
    a(f"- counterfactual group overlap = {split['counterfactual_group_overlap']} (informational)")
    a(f"- persona overlap = {split['persona_overlap']} "
      f"({'holdout: unseen personas in test' if not split['persona_overlap'] else 'allowed in v0.2-A, reported for transparency'})\n")
    a("## Training\n")
    a(f"- best epoch: {best_epoch}, runtime: {runtime:.1f}s")
    if history:
        last = history[-1]
        a(f"- final train total loss: {last['train_total_loss']}")
        a(f"- best val total loss: {min(h['val_total_loss'] for h in history)}\n")
    a("## Test Metrics\n")
    for k, v in test.items():
        a(f"- {k}: {v}")
    if "mean_pairwise_l1" in noise_ref:
        a(f"\nTeacher sampling-noise reference: mean pairwise L1 = {noise_ref['mean_pairwise_l1']} "
          f"({noise_ref['n_states']} states, from dataset repeat records)\n")
    else:
        a(f"\nTeacher aggregation noise reference (K3<->K5 mean L1) = {noise_ref.get('k3_vs_k5_mean_l1')}\n")
    a("\n## Counterfactual Preview (evaluation only, no elasticity training)\n")
    a(f"- mean |delta_P_T - delta_P_S| = {cf['mean_abs_delta_p_diff']}")
    a(f"- mean sign agreement = {sign_agree['mean_sign_agreement']} ({sign_agree['n_pairs']} pairs)\n")
    a("\n## Per-Persona Test Breakdown\n")
    a("| persona | n | mode_acc | prob_L1 |")
    a("|---|---|---|---|")
    for row in p_breakdown:
        a(f"| {row['persona_id']} | {row['n_samples']} | {row['mode_accuracy']} | {row['probability_l1']} |")
    a("\n## Known Limitations\n")
    a("- small development dataset (not a final model)")
    a("- no L_elasticity / L_heterogeneity training")
    a("- no real-world calibration, no MATSim")
    return "\n".join(l) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
