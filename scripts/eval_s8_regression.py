#!/usr/bin/env python
"""S8 generic-capability regression gate (S8 instructions §25-26).

Re-runs the S7 evaluation pipelines with the S8 model against the FROZEN
S7-W3 baseline on:
  * S3 legacy test (226 states): accuracy / KL / probability L1 / dP gap / sign;
  * S5 joint test: seen / unseen combination KL + L1, interaction L1 error;
  * S6/S7 mechanism test (16 quadruplets): parking G_med, congestion Gap_shortcut.
Metric definitions are IMPORTED from the S7 eval scripts (identical formulas).
All deltas are S8 - S7W3 with 95% paired bootstrap CIs (B=2000, seed=42).

Regression gate (S8 §26):
    legacy accuracy drop <= 1 pp; legacy/seen/unseen joint KL increase <= 10%;
    mechanism indicators must not clearly regress.

Usage:
    python scripts/eval_s8_regression.py \
        --s8-checkpoint outputs/student_s8/checkpoints/best.pt \
        --output outputs/s8_regression
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

import numpy as np
import torch

from eval_s7_regression import (  # noqa: E402
    _bootstrap,
    _bootstrap_delta,
    _kl,
    _l1,
    _level_matches,
    _load,
    _predict,
)
from traveler_distillation.accessibility.accessibility_features import (  # noqa: E402
    S8FeatureExtractor,
    TravelerStudentS8,
)
from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import resolve_combination  # noqa: E402
from traveler_distillation.student import (  # noqa: E402
    FeatureExtractor,
    TravelerStudent,
    build_baseline_index,
    collate_batch,
    load_mechanism_quadruplets,
    make_counterfactual_pairs,
)

FROZEN_RELEASE_CKPT = _ROOT / "releases" / "s7_w3_generic_core_v1" / "checkpoint" / "model.pt"
B = 2000
SEED = 42
_EPS = 1e-6


def _load_s7(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    extractor = FeatureExtractor().from_state_dict(ckpt["extractor_state"])
    model = TravelerStudent(ckpt.get("config", {}), ckpt["feature_spec"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, extractor


def _load_s8(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    extractor = S8FeatureExtractor().from_state_dict(ckpt["extractor_state"])
    model = TravelerStudentS8(ckpt.get("config", {}), ckpt["feature_spec"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, extractor


@torch.no_grad()
def _predict_state(state, model, extractor, device) -> dict[str, float]:
    """State-level prediction (used by the causal/mechanism part)."""
    feats = extractor.encode(state)
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
    out = model(batch)["mode_probabilities"][0].cpu().numpy()
    return {alt.mode: float(out[i]) for i, alt in enumerate(state.alternatives) if alt.available}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", default="data/student_v0_3_s3/aggregated_teacher_dataset.jsonl")
    ap.add_argument("--joint-k5", default="data/student_s5_joint/aggregated_teacher_dataset_k5.jsonl")
    ap.add_argument("--joint-config", default="configs/joint_sampling.yaml")
    ap.add_argument("--split-manifest", default="outputs/student_v0_3_s3_c/split_manifest.json")
    ap.add_argument("--mechanism-dataset", default="data/student_s7_mechanism/quadruplets.jsonl")
    ap.add_argument("--s8-checkpoint", default="outputs/student_s8/checkpoints/best.pt")
    ap.add_argument("--output", default="outputs/s8_regression")
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

    models = {
        "W3_S7W3": _load_s7(FROZEN_RELEASE_CKPT, device),
        "S8": _load_s8(Path(args.s8_checkpoint), device),
    }
    report = {"bootstrap": {"B": B, "seed": SEED, "unit": "state (paired over states)"},
              "models": {}, "deltas_s8_vs_s7w3": {}}

    for name, (model, extractor) in models.items():
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
        kl_v = np.array([r["kl"] for r in per_state])
        l1_v = np.array([r["l1"] for r in per_state])
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
        legacy = {
            "n": len(per_state),
            "mode_accuracy": _bootstrap(acc),
            "kl": _bootstrap(kl_v),
            "probability_l1": _bootstrap(l1_v),
            "delta_p_gap": _bootstrap(np.array(dp_gaps)),
            "sign_agreement": _bootstrap(np.array(sign_agree)),
        }

        def joint_metrics(samples):
            kl_s, l1_s = [], []
            for s in samples:
                t = s.teacher_aggregate.mode_probabilities
                p = _predict(s, model, extractor, device)
                kl_s.append(_kl(t, p))
                l1_s.append(_l1(t, p))
            return (np.array(kl_s), np.array(l1_s)) if kl_s else (np.array([]), np.array([]))

        seen_kl, seen_l1 = joint_metrics(seen_joint)
        unseen_kl, unseen_l1 = joint_metrics(unseen_joint)
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

        report["models"][name] = {
            "legacy": legacy,
            "seen_joint": {"n": len(seen_joint),
                           "kl": _bootstrap(seen_kl) if len(seen_kl) else None,
                           "l1": _bootstrap(seen_l1) if len(seen_l1) else None},
            "unseen_joint": {"n": len(unseen_joint),
                             "kl": _bootstrap(unseen_kl) if len(unseen_kl) else None,
                             "l1": _bootstrap(unseen_l1) if len(unseen_l1) else None},
            "interaction_l1_error": _bootstrap(np.array(inter_errs)) if inter_errs else None,
        }

    # ---- paired deltas S8 - S7W3 ----
    s8, s7 = models["S8"], models["W3_S7W3"]
    rows_a, rows_b = [], []
    for s in legacy_test:
        t = s.teacher_aggregate.mode_probabilities
        rows_a.append(_kl(t, _predict(s, s8[0], s8[1], device)))
        rows_b.append(_kl(t, _predict(s, s7[0], s7[1], device)))
    legacy_kl_delta = _bootstrap_delta(np.array(rows_a), np.array(rows_b))

    def joint_kl_delta(samples):
        a = [_kl(s.teacher_aggregate.mode_probabilities, _predict(s, s8[0], s8[1], device)) for s in samples]
        b = [_kl(s.teacher_aggregate.mode_probabilities, _predict(s, s7[0], s7[1], device)) for s in samples]
        return _bootstrap_delta(np.array(a), np.array(b)) if a else None

    report["deltas_s8_vs_s7w3"] = {
        "legacy_kl_delta": legacy_kl_delta,
        "seen_joint_kl_delta": joint_kl_delta(seen_joint),
        "unseen_joint_kl_delta": joint_kl_delta(unseen_joint),
        "note": "delta = S8 - S7W3; KL increase <= 10% of the S7-W3 value required (§26)",
    }

    # ---- causal / mechanism part (16 test quadruplets) ----
    quads = load_mechanism_quadruplets(args.mechanism_dataset, split="test")
    assert all(q.split == "test" for q in quads) and len(quads) == 16
    group_metrics = defaultdict(lambda: defaultdict(dict))
    axes = defaultdict(list)
    for quad in quads:
        axis = quad.axis_id
        gid = quad.audit_group_id
        axes[axis].append(gid)
        members = quad.members
        PA_t = members["baseline"].teacher_probs
        PB_t = members["natural"].teacher_probs
        PC_t = members["broken"].teacher_probs
        PD_t = members["mediator"].teacher_probs
        for name, (model, extractor) in models.items():
            PA = _predict_state(members["baseline"].state, model, extractor, device)
            PB = _predict_state(members["natural"].state, model, extractor, device)
            PC = _predict_state(members["broken"].state, model, extractor, device)
            PD = _predict_state(members["mediator"].state, model, extractor, device)
            En, Eb, Em = _l1(PB, PA), _l1(PC, PA), _l1(PD, PA)
            group_metrics[name][axis][gid] = {
                "E_natural": En, "E_broken": Eb, "E_mediator": Em,
                "R_shortcut": Eb / (En + _EPS), "R_mediator": Em / (En + _EPS),
            }
        En_t, Eb_t, Em_t = _l1(PB_t, PA_t), _l1(PC_t, PA_t), _l1(PD_t, PA_t)
        group_metrics["Teacher"][axis][gid] = {
            "E_natural": En_t, "E_broken": Eb_t, "E_mediator": Em_t,
            "R_shortcut": Eb_t / (En_t + _EPS), "R_mediator": Em_t / (En_t + _EPS),
        }

    causal = {"n_test_groups_per_axis": {ax: len(g) for ax, g in axes.items()},
              "models": {}, "deltas_s8_vs_s7w3": {}}
    for name in list(models) + ["Teacher"]:
        mrep = {}
        for axis, gids in axes.items():
            rows = [group_metrics[name][axis][g] for g in gids]
            t_rows = [group_metrics["Teacher"][axis][g] for g in gids]
            out = {}
            for field in ("R_shortcut", "R_mediator"):
                out[field] = _bootstrap(np.array([r[field] for r in rows]))
            for field, tfield in (("G_med", "E_mediator"), ("Gap_shortcut", "R_shortcut")):
                gap = np.abs(np.array([r[tfield] for r in rows]) - np.array([r[tfield] for r in t_rows]))
                out[field] = _bootstrap(gap)
            mrep[axis] = out
        causal["models"][name] = mrep
    for axis, gids in axes.items():
        a = np.abs(np.array([group_metrics["S8"][axis][g]["E_mediator"] for g in gids])
                   - np.array([group_metrics["Teacher"][axis][g]["E_mediator"] for g in gids]))
        b = np.abs(np.array([group_metrics["W3_S7W3"][axis][g]["E_mediator"] for g in gids])
                   - np.array([group_metrics["Teacher"][axis][g]["E_mediator"] for g in gids]))
        causal["deltas_s8_vs_s7w3"][axis] = {"G_med_delta": _bootstrap_delta(a, b)}
    report["causal_mechanism"] = causal

    # ---- regression gate (§26) ----
    s7_legacy = report["models"]["W3_S7W3"]["legacy"]
    s8_legacy = report["models"]["S8"]["legacy"]
    s7_seen = report["models"]["W3_S7W3"]["seen_joint"]["kl"]["mean"]
    s8_seen = report["models"]["S8"]["seen_joint"]["kl"]["mean"]
    s7_unseen = report["models"]["W3_S7W3"]["unseen_joint"]["kl"]["mean"]
    s8_unseen = report["models"]["S8"]["unseen_joint"]["kl"]["mean"]
    gate = {
        "legacy_accuracy_drop_pp": round((s7_legacy["mode_accuracy"]["mean"] - s8_legacy["mode_accuracy"]["mean"]) * 100, 4),
        "legacy_accuracy_gate_1pp": bool((s7_legacy["mode_accuracy"]["mean"] - s8_legacy["mode_accuracy"]["mean"]) <= 0.01),
        "legacy_kl_increase_pct": round((s8_legacy["kl"]["mean"] - s7_legacy["kl"]["mean"]) / max(s7_legacy["kl"]["mean"], 1e-9) * 100, 2),
        "legacy_kl_gate_10pct": bool((s8_legacy["kl"]["mean"] - s7_legacy["kl"]["mean"]) <= 0.10 * s7_legacy["kl"]["mean"]),
        "seen_joint_kl_increase_pct": round((s8_seen - s7_seen) / max(s7_seen, 1e-9) * 100, 2),
        "seen_joint_kl_gate_10pct": bool((s8_seen - s7_seen) <= 0.10 * s7_seen),
        "unseen_joint_kl_increase_pct": round((s8_unseen - s7_unseen) / max(s7_unseen, 1e-9) * 100, 2),
        "unseen_joint_kl_gate_10pct": bool((s8_unseen - s7_unseen) <= 0.10 * s7_unseen),
        "note": "mechanism indicators judged separately (must not clearly regress)",
    }
    report["regression_gate"] = gate

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:6000])
    print(f"\nwrote {out / 'eval_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
