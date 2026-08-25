#!/usr/bin/env python
"""S5 multi-axis joint evaluation: seen/unseen combination + interaction.

Compares M0 (S3-C single-axis), M1 (joint fine-tune, base-K targets) and M2
(joint fine-tune, K=7-stabilized targets) on THREE evidence layers:

  1. Seen-combination joint fidelity    (combos rain/cong, fare/delay, disrupt/cong)
  2. Unseen-combination joint fidelity  (combo fare/cong held out of training)
  3. Interaction effect error           I_ij = P_ij - P_i - P_j + P_0  (T vs S)
  4. Legacy single-axis regression      (S3 test set must not degrade)

All evaluation is restricted to the S3-C UNSEEN test personas (persona holdout).

Usage:
    python scripts/eval_s5_joint.py \
        --base-dataset data/student_v0_3_s3/aggregated_teacher_dataset.jsonl \
        --joint-k5 data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl \
        --joint-full data/student_s5_joint/aggregated_teacher_dataset.jsonl \
        --joint-config configs/joint_sampling.yaml \
        --split-manifest outputs/student_v0_3_s3_c/split_manifest.json \
        --m0 outputs/student_v0_3_s3_c/checkpoints/best.pt \
        --m1 outputs/student_s5_joint_m1/checkpoints/best.pt \
        --m2 outputs/student_s5_joint_m2/checkpoints/best.pt \
        --output outputs/s5_joint_eval
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
from traveler_distillation.student import FeatureExtractor, TravelerStudent, collate_batch


def _load(path: Path) -> list[AggregatedTeacherTarget]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(AggregatedTeacherTarget.model_validate(json.loads(line)))
    return samples


def _load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    extractor = FeatureExtractor().from_state_dict(ckpt["extractor_state"])
    model = TravelerStudent(ckpt.get("config", {}), ckpt["feature_spec"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, extractor


@torch.no_grad()
def _predict(sample, model, extractor, device) -> dict[str, float]:
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


def _kl(p, q) -> float:
    """KL(P || Q) over the union of modes."""
    modes = set(p) | set(q)
    kl = 0.0
    for m in modes:
        pm = p.get(m, 0.0)
        qm = max(q.get(m, 0.0), 1e-9)
        if pm > 0:
            kl += pm * (__import__("math").log(pm / qm))
    return kl


def _l1(p, q) -> float:
    modes = set(p) | set(q)
    return sum(abs(p.get(m, 0.0) - q.get(m, 0.0)) for m in modes)


def _level_matches(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return a == b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-k5", default="data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl")
    ap.add_argument("--joint-full", default="data/student_s5_joint/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--m0", default="outputs/student_v0_3_s3_c/checkpoints/best.pt")
    ap.add_argument("--m1", default="outputs/student_s5_joint_m1/checkpoints/best.pt")
    ap.add_argument("--m2", default="outputs/student_s5_joint_m2/checkpoints/best.pt")
    ap.add_argument("--output", default="outputs/s5_joint_eval")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    combos = load_yaml(args.joint_config).get("joint_combinations", [])
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    test_personas = set(manifest["personas"]["test"])

    base = _load(Path(args.base_dataset))
    joint_k5 = _load(Path(args.joint_k5))
    joint_full = _load(Path(args.joint_full))

    # index S3 single-axis states by split_group_id for interaction linking
    s3_by_group = defaultdict(list)
    for s in base:
        s3_by_group[s.split_group_id].append(s)

    def _test_joint(samples):
        out = []
        for s in samples:
            pid = s.persona_group_id or s.state.persona.persona_id
            if pid in test_personas:
                out.append(s)
        return out

    test_joint_k5 = _test_joint(joint_k5)
    test_joint_full = _test_joint(joint_full)

    # legacy single-axis test set (baseline + single-axis CF) for regression
    legacy_test = [s for s in base if (s.persona_group_id or s.state.persona.persona_id) in test_personas]

    models = {
        "M0": _load_model(Path(args.m0), device),
        "M1": _load_model(Path(args.m1), device),
        "M2": _load_model(Path(args.m2), device),
    }

    def joint_fidelity(samples, model, extractor):
        per_combo = defaultdict(lambda: {"kl": [], "l1": [], "acc": 0, "n": 0})
        for s in samples:
            combo = resolve_combination(s.perturbation.joint_axes, combos)
            cid = combo["combination_id"] if combo else "unknown"
            seen = combo.get("seen_in_training", True) if combo else None
            t = s.teacher_aggregate.mode_probabilities
            p = _predict(s, model, extractor, device)
            row = per_combo[cid]
            row["kl"].append(_kl(t, p))
            row["l1"].append(_l1(t, p))
            row["acc"] += 1 if max(p, key=p.get) == s.teacher_aggregate.selected_mode else 0
            row["n"] += 1
            row["seen"] = seen
        result = {}
        for cid, row in per_combo.items():
            result[cid] = {
                "n": row["n"],
                "seen_in_training": row["seen"],
                "joint_kl": round(sum(row["kl"]) / len(row["kl"]), 4),
                "joint_l1": round(sum(row["l1"]) / len(row["l1"]), 4),
                "mode_accuracy": round(row["acc"] / row["n"], 4),
            }
        # aggregate seen vs unseen
        agg = {"seen": {"kl": [], "l1": [], "n": 0}, "unseen": {"kl": [], "l1": [], "n": 0}}
        for cid, row in result.items():
            bucket = "seen" if row["seen_in_training"] else "unseen"
            agg[bucket]["kl"].extend([row["joint_kl"]] * row["n"])
            agg[bucket]["l1"].extend([row["joint_l1"]] * row["n"])
            agg[bucket]["n"] += row["n"]
        summary = {}
        for bucket, d in agg.items():
            summary[bucket] = {
                "n": d["n"],
                "joint_kl": round(sum(d["kl"]) / len(d["kl"]), 4) if d["kl"] else None,
                "joint_l1": round(sum(d["l1"]) / len(d["l1"]), 4) if d["l1"] else None,
            }
        return {"per_combo": result, "aggregate": summary}

    def interaction_error(samples, model, extractor):
        errs = defaultdict(list)
        n_linked = 0
        n_total = 0
        for s in samples:
            n_total += 1
            combo = resolve_combination(s.perturbation.joint_axes, combos)
            cid = combo["combination_id"] if combo else "unknown"
            base = next((x for x in s3_by_group[s.split_group_id] if x.perturbation.axis == "baseline"), None)
            singles = []
            ok = True
            for ja in s.perturbation.joint_axes:
                single = next((x for x in s3_by_group[s.split_group_id]
                               if x.perturbation.axis == ja["axis"] and _level_matches(x.perturbation.level, ja["level"])), None)
                if single is None:
                    ok = False
                    break
                singles.append(single)
            if base is None or not ok:
                continue
            n_linked += 1
            modes = set()
            P0_t = base.teacher_aggregate.mode_probabilities
            Pis_t = [x.teacher_aggregate.mode_probabilities for x in singles]
            Pij_t = s.teacher_aggregate.mode_probabilities
            P0_s = _predict(base, model, extractor, device)
            Pis_s = [_predict(x, model, extractor, device) for x in singles]
            Pij_s = _predict(s, model, extractor, device)
            modes |= set(P0_t) | set(Pij_t) | set(P0_s) | set(Pij_s)
            for x in Pis_t:
                modes |= set(x)
            for x in Pis_s:
                modes |= set(x)
            total = 0.0
            for m in modes:
                I_t = Pij_t.get(m, 0.0) - sum(x.get(m, 0.0) for x in Pis_t) + P0_t.get(m, 0.0)
                I_s = Pij_s.get(m, 0.0) - sum(x.get(m, 0.0) for x in Pis_s) + P0_s.get(m, 0.0)
                total += abs(I_t - I_s)
            errs[cid].append(total / len(modes) if modes else 0.0)
        out = {}
        for cid, vals in errs.items():
            out[cid] = {"n": len(vals), "interaction_l1_error": round(sum(vals) / len(vals), 4)}
        mean = round(sum(v for vals in errs.values() for v in vals) / sum(len(v) for v in errs.values()), 4) if errs else None
        return {"per_combo": out, "mean_interaction_l1_error": mean, "n_linked": n_linked, "n_total": n_total}

    def legacy_regression(model, extractor):
        acc = 0
        kl = 0.0
        l1 = 0.0
        n = len(legacy_test)
        for s in legacy_test:
            t = s.teacher_aggregate.mode_probabilities
            p = _predict(s, model, extractor, device)
            acc += 1 if max(p, key=p.get) == s.teacher_aggregate.selected_mode else 0
            kl += _kl(t, p)
            l1 += _l1(t, p)
        return {
            "n": n,
            "mode_accuracy": round(acc / n, 4) if n else None,
            "kl": round(kl / n, 4) if n else None,
            "probability_l1": round(l1 / n, 4) if n else None,
        }

    report = {"models": {}}
    for name, (model, extractor) in models.items():
        print(f"\n=== {name} ===", flush=True)
        jf_k5 = joint_fidelity(test_joint_k5, model, extractor)
        jf_full = joint_fidelity(test_joint_full, model, extractor)
        inter_k5 = interaction_error(test_joint_k5, model, extractor)
        inter_full = interaction_error(test_joint_full, model, extractor)
        legacy = legacy_regression(model, extractor)
        report["models"][name] = {
            "joint_fidelity": jf_k5,       # M1 targets for M1; identical to full except K=7 subset
            "joint_fidelity_full": jf_full,
            "interaction": inter_k5,
            "interaction_full": inter_full,
            "legacy_single_axis": legacy,
        }
        print(json.dumps({"joint_fidelity": jf_k5, "interaction": inter_k5, "legacy": legacy},
                         ensure_ascii=False, indent=2))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'eval_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
