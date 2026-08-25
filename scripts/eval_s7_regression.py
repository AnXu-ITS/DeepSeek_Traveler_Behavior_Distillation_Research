#!/usr/bin/env python
"""S7 regression evaluation: legacy six-axis + S5 seen/unseen joint retention.

Checks the S7 constraint "no degradation": every S7 variant is compared with
C1 (the S5 joint M2 start point) on
  1. the S3 legacy six-axis test set (persona holdout, same test personas as
     S3/S5): mode accuracy, KL, probability L1, counterfactual ΔP gap, sign
     agreement;
  2. the S5 joint test set: seen-combination KL/L1, unseen fare×congestion
     holdout KL/L1, interaction L1 error.
Deltas vs C1 carry 95% paired bootstrap CIs (states as units, B=2000, fixed
seed) — user-mandated for final results. This script makes no API calls and
never touches the S7 causal quadruplets.

Usage:
    python scripts/eval_s7_regression.py \
        --base-dataset data/student_v0_3_s3/aggregated_teacher_dataset.jsonl \
        --joint-k5 data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl \
        --joint-config configs/joint_sampling.yaml \
        --split-manifest outputs/student_v0_3_s3_c/split_manifest.json \
        --c1 outputs/student_s5_joint_m2/checkpoints/best.pt \
        --variant W1=outputs/student_s7_w1/checkpoints/best.pt \
        --variant W2=outputs/student_s7_w2/checkpoints/best.pt \
        --output outputs/s7_regression/eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import torch

from traveler_distillation.config import load_yaml
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget
from traveler_distillation.generators import resolve_combination
from traveler_distillation.student import (
    FeatureExtractor,
    TravelerStudent,
    collate_batch,
    build_baseline_index,
    make_counterfactual_pairs,
    elasticity_l1,
)

B = 2000
SEED = 42


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
    import math
    modes = set(p) | set(q)
    kl = 0.0
    for m in modes:
        pm = p.get(m, 0.0)
        qm = max(q.get(m, 0.0), 1e-9)
        if pm > 0:
            kl += pm * math.log(pm / qm)
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


def _bootstrap(values: np.ndarray, n_boot: int = B, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        means[i] = values[idx].mean()
    return {
        "mean": round(float(values.mean()), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
    }


def _bootstrap_delta(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired bootstrap CI of mean(a - b) (a = variant, b = C1)."""
    rng = np.random.default_rng(SEED)
    deltas = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(a), size=len(a))
        deltas[i] = (a[idx] - b[idx]).mean()
    return {
        "mean": round(float((a - b).mean()), 4),
        "ci_low": round(float(np.percentile(deltas, 2.5)), 4),
        "ci_high": round(float(np.percentile(deltas, 97.5)), 4),
        "ci_excludes_zero": bool(np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-k5", default="data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--c1", default="outputs/student_s5_joint_m2/checkpoints/best.pt")
    ap.add_argument("--variant", action="append", default=[],
                    help="name=path of an S7 checkpoint (repeatable)")
    ap.add_argument("--output", default="outputs/s7_regression/eval_metrics.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    combos = load_yaml(args.joint_config).get("joint_combinations", [])
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    test_personas = set(manifest["personas"]["test"])

    base = _load(Path(args.base_dataset))
    joint = _load(Path(args.joint_k5))

    def _pid(s):
        return s.persona_group_id or s.state.persona.persona_id

    legacy_test = [s for s in base if _pid(s) in test_personas]
    test_joint = [s for s in joint if _pid(s) in test_personas]
    seen_joint = [s for s in test_joint if (resolve_combination(s.perturbation.joint_axes, combos) or {}).get("seen_in_training", True)]
    unseen_joint = [s for s in test_joint if not (resolve_combination(s.perturbation.joint_axes, combos) or {}).get("seen_in_training", True)]
    print(f"legacy_test={len(legacy_test)} seen_joint={len(seen_joint)} unseen_joint={len(unseen_joint)}")

    s3_by_group = defaultdict(list)
    for s in base:
        s3_by_group[s.split_group_id].append(s)

    models: dict[str, object] = {"C1_S5": _load_model(Path(args.c1), device)}
    for spec in args.variant:
        name, _, path = spec.partition("=")
        models[name] = _load_model(Path(path), device)

    report = {"bootstrap": {"B": B, "seed": SEED, "unit": "state (paired over states)"},
              "models": {}, "deltas_vs_c1": {}}

    for name, (model, extractor) in models.items():
        # --- legacy single-axis ---
        per_state = []
        for s in legacy_test:
            t = s.teacher_aggregate.mode_probabilities
            p = _predict(s, model, extractor, device)
            per_state.append({
                "acc": 1.0 if max(p, key=p.get) == s.teacher_aggregate.selected_mode else 0.0,
                "kl": _kl(t, p),
                "l1": _l1(t, p),
            })
        acc = np.array([r["acc"] for r in per_state])
        kl = np.array([r["kl"] for r in per_state])
        l1 = np.array([r["l1"] for r in per_state])
        # counterfactual ΔP gap + sign agreement on the legacy test pairs
        base_index = build_baseline_index(legacy_test)
        pairs = make_counterfactual_pairs(legacy_test, base_index)
        dp_gaps, sign_agree = [], []
        for b, c in pairs:
            tb, tc = b.teacher_aggregate.mode_probabilities, c.teacher_aggregate.mode_probabilities
            pb = _predict(b, model, extractor, device)
            pc = _predict(c, model, extractor, device)
            modes = set(tb) | set(tc) | set(pb) | set(pc)
            gap = 0.0
            agree = 0
            for m in modes:
                dt = tc.get(m, 0.0) - tb.get(m, 0.0)
                ds = pc.get(m, 0.0) - pb.get(m, 0.0)
                gap += abs(dt - ds)
                if abs(dt) < 1e-3 or (dt > 0) == (ds > 0):
                    agree += 1
            dp_gaps.append(gap / len(modes) if modes else 0.0)
            sign_agree.append(agree / len(modes) if modes else 1.0)
        dp_gap = np.array(dp_gaps)
        sign = np.array(sign_agree)
        legacy = {
            "n": len(per_state),
            "mode_accuracy": _bootstrap(acc),
            "kl": _bootstrap(kl),
            "probability_l1": _bootstrap(l1),
            "delta_p_gap": _bootstrap(dp_gap),
            "sign_agreement": _bootstrap(sign),
        }
        # --- seen / unseen joint ---
        def joint_metrics(samples):
            kl_v, l1_v = [], []
            for s in samples:
                t = s.teacher_aggregate.mode_probabilities
                p = _predict(s, model, extractor, device)
                kl_v.append(_kl(t, p))
                l1_v.append(_l1(t, p))
            return (np.array(kl_v), np.array(l1_v)) if kl_v else (np.array([]), np.array([]))

        seen_kl, seen_l1 = joint_metrics(seen_joint)
        unseen_kl, unseen_l1 = joint_metrics(unseen_joint)
        # interaction error (seen combos only, like S5)
        inter_errs = []
        for s in seen_joint:
            base_s = next((x for x in s3_by_group[s.split_group_id] if x.perturbation.axis == "baseline"), None)
            singles = []
            ok = base_s is not None
            for ja in s.perturbation.joint_axes:
                single = next((x for x in s3_by_group[s.split_group_id]
                               if x.perturbation.axis == ja["axis"] and _level_matches(x.perturbation.level, ja["level"])), None)
                if single is None:
                    ok = False
                    break
                singles.append(single)
            if not ok:
                continue
            P0_t = base_s.teacher_aggregate.mode_probabilities
            Pis_t = [x.teacher_aggregate.mode_probabilities for x in singles]
            Pij_t = s.teacher_aggregate.mode_probabilities
            P0_s = _predict(base_s, model, extractor, device)
            Pis_s = [_predict(x, model, extractor, device) for x in singles]
            Pij_s = _predict(s, model, extractor, device)
            modes = set(P0_t) | set(Pij_t) | set(P0_s) | set(Pij_s)
            for x in Pis_t:
                modes |= set(x)
            for x in Pis_s:
                modes |= set(x)
            total = 0.0
            for m in modes:
                I_t = Pij_t.get(m, 0.0) - sum(x.get(m, 0.0) for x in Pis_t) + P0_t.get(m, 0.0)
                I_s = Pij_s.get(m, 0.0) - sum(x.get(m, 0.0) for x in Pis_s) + P0_s.get(m, 0.0)
                total += abs(I_t - I_s)
            inter_errs.append(total / len(modes) if modes else 0.0)
        inter = np.array(inter_errs)

        model_report = {
            "legacy": legacy,
            "seen_joint": {
                "n": len(seen_joint),
                "kl": _bootstrap(seen_kl) if len(seen_kl) else None,
                "l1": _bootstrap(seen_l1) if len(seen_l1) else None,
            },
            "unseen_joint": {
                "n": len(unseen_joint),
                "kl": _bootstrap(unseen_kl) if len(unseen_kl) else None,
                "l1": _bootstrap(unseen_l1) if len(unseen_l1) else None,
            },
            "interaction_l1_error": _bootstrap(inter) if len(inter) else None,
        }
        report["models"][name] = model_report

    # --- paired deltas vs C1 ---
    c1 = report["models"]["C1_S5"]
    for name in models:
        if name == "C1_S5":
            continue
        m = report["models"][name]
        # recompute raw arrays for deltas (per-state values needed)
        c1_model, c1_extr = models["C1_S5"]
        (model, extractor) = models[name]
        rows_a, rows_b = [], []
        for s in legacy_test:
            t = s.teacher_aggregate.mode_probabilities
            rows_a.append(_kl(t, _predict(s, model, extractor, device)))
            rows_b.append(_kl(t, _predict(s, c1_model, c1_extr, device)))
        legacy_kl_delta = _bootstrap_delta(np.array(rows_a), np.array(rows_b))

        def joint_kl_delta(samples):
            a = [_kl(s.teacher_aggregate.mode_probabilities, _predict(s, model, extractor, device)) for s in samples]
            b = [_kl(s.teacher_aggregate.mode_probabilities, _predict(s, c1_model, c1_extr, device)) for s in samples]
            return _bootstrap_delta(np.array(a), np.array(b)) if a else None

        report["deltas_vs_c1"][name] = {
            "legacy_kl_delta": legacy_kl_delta,
            "seen_joint_kl_delta": joint_kl_delta(seen_joint),
            "unseen_joint_kl_delta": joint_kl_delta(unseen_joint),
            "note": "delta = variant - C1; negative KL delta = no degradation (CI excludes 0 => significant)",
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
