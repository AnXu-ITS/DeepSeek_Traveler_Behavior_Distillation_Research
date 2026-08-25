#!/usr/bin/env python
"""Compute S7 val-set metrics for one checkpoint (model-selection input only).

Mirrors the val computations inside train_student_s7.py so that the C1 baseline
and every S7 variant share the same val measurement protocol. Selection uses
val ONLY — the final causal test set is never touched here.

Usage:
    python scripts/eval_s7_val_metrics.py \
        --checkpoint outputs/student_s5_joint_m2/checkpoints/best.pt \
        --base-dataset data/student_v0_3_s3/aggregated_teacher_dataset.jsonl \
        --joint-dataset data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl \
        --joint-config configs/joint_sampling.yaml \
        --mechanism-dataset data/student_s7_mechanism/quadruplets.jsonl \
        --split-manifest outputs/student_v0_3_s3_c/split_manifest.json \
        --output outputs/s7_selection/c1_val_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch

from traveler_distillation.config import load_yaml
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget
from traveler_distillation.generators import resolve_combination
from traveler_distillation.student import (
    FeatureExtractor,
    TravelerStudent,
    kl_divergence,
    collate_batch,
    load_mechanism_quadruplets,
    MechanismQuadrupletDataset,
    collate_quadruplets,
)


def _load(path: Path) -> list[AggregatedTeacherTarget]:
    return [AggregatedTeacherTarget.model_validate(json.loads(l))
            for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-dataset", default="data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--mechanism-dataset", default="data/student_s7_mechanism/quadruplets.jsonl")
    ap.add_argument("--split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--output", default="outputs/s7_selection/val_metrics.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    combos = load_yaml(args.joint_config).get("joint_combinations", [])
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    persona_sets = manifest["personas"]
    split_of = {}
    for sname in ("train", "val", "test"):
        for pid in persona_sets[sname]:
            split_of[pid] = sname

    ckpt = torch.load(Path(args.checkpoint), map_location=device)
    extractor = FeatureExtractor().from_state_dict(ckpt["extractor_state"])
    model = TravelerStudent(ckpt.get("config", {}), ckpt["feature_spec"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    base = _load(Path(args.base_dataset))
    joint = _load(Path(args.joint_dataset))
    joint = [s for s in joint if (resolve_combination(s.perturbation.joint_axes, combos) or {}).get("seen_in_training", True)]
    val_base = [s for s in base if split_of.get(s.persona_group_id or s.state.persona.persona_id) == "val"]
    val_joint = [s for s in joint if split_of.get(s.persona_group_id or s.state.persona.persona_id) == "val"]
    val_quads = load_mechanism_quadruplets(args.mechanism_dataset, split="val")
    assert all(q.split == "val" for q in val_quads)

    @torch.no_grad()
    def _eval_states(samples):
        n = 0
        sum_kl = 0.0
        sum_l1 = 0.0
        correct = 0
        for s in samples:
            feats = extractor.encode(s.state)
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
            out = model(batch)
            probs = out["mode_probabilities"][0]
            target = torch.tensor([
                float(s.teacher_aggregate.mode_probabilities.get(a.mode, 0.0)) if a.available else 0.0
                for a in s.state.alternatives
            ])
            pred = s.state.alternatives[int(probs.argmax())].mode
            correct += 1 if pred == s.teacher_aggregate.selected_mode else 0
            sum_kl += kl_divergence(target.unsqueeze(0), probs.unsqueeze(0)).item()
            sum_l1 += (target - probs.cpu()).abs().sum().item()
            n += 1
        return {"mode_accuracy": round(correct / n, 4) if n else None,
                "kl": round(sum_kl / n, 4) if n else None,
                "probability_l1": round(sum_l1 / n, 4) if n else None, "n": n}

    @torch.no_grad()
    def _quad_gaps(quads):
        acc = defaultdict(lambda: {"G_nat": [], "G_broken": [], "G_med": []})
        for quad in quads:
            ds = MechanismQuadrupletDataset([quad], extractor)
            batch = collate_quadruplets([ds[0]])
            batch = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in batch.items()}
            out = {k: model(batch[k])["mode_probabilities"][0] for k in ("A", "B", "C", "D")}
            mask = batch["A"]["alt_mask"][0]
            denom = mask.sum().item()
            t = {k: batch[k]["target_probs"][0] for k in ("A", "B", "C", "D")}
            G_nat = ((t["B"] - t["A"]) - (out["B"] - out["A"])).abs().mul(mask).sum().item() / denom
            G_broken = ((t["C"] - t["A"]) - (out["C"] - out["A"])).abs().mul(mask).sum().item() / denom
            G_med = ((t["D"] - t["A"]) - (out["D"] - out["A"])).abs().mul(mask).sum().item() / denom
            a = acc[quad.axis_id]
            a["G_nat"].append(G_nat)
            a["G_broken"].append(G_broken)
            a["G_med"].append(G_med)
        out = {}
        for axis, a in acc.items():
            out[axis] = {k: round(sum(v) / len(v), 4) for k, v in a.items() if v}
            out[axis]["n_quads"] = len(a["G_nat"])
        return out

    gap = _quad_gaps(val_quads)
    mean_gap = sum(v["G_nat"] + v["G_broken"] + v["G_med"] for v in gap.values()) / max(1, 3 * len(gap))
    result = {
        "checkpoint": args.checkpoint,
        "val_mechanism_gap": gap,
        "val_mean_mechanism_gap": round(mean_gap, 4),
        "val_legacy": _eval_states(val_base),
        "val_seen_joint": _eval_states(val_joint),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
