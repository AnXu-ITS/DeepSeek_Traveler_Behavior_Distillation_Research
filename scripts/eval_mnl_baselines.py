#!/usr/bin/env python
"""E1 S3 — decision-level evaluation of MNL-B (B1/B2/B3) + G2 consistency gate.

B1/B2 mirror scripts/eval_s8_accessibility.py exactly (same metric formulas,
same bootstrap B=2000 seed=42, same test-split assertions) with MNL-B
predictions. The frozen S9 checkpoint is recomputed on the same 51 test states
for the G2 gate (must reproduce outputs/s9_accessibility_eval/eval_metrics.json
at 4-decimal rounding) and to fill the mode-accuracy row the frozen file lacks.

B3 mirrors scripts/eval_s8_regression.py (legacy 226 / seen joint 72 / unseen
joint 24 states + interaction error) with MNL-B predictions, compared against
the frozen S9 row ("S8" key) in outputs/s9_regression/eval_metrics.json.

Usage:
    python scripts/eval_mnl_baselines.py
"""
from __future__ import annotations

import json
import math
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
)
from traveler_distillation.accessibility.accessibility_features import (  # noqa: E402
    S8FeatureExtractor,
    TravelerStudentS8,
)
from traveler_distillation.accessibility.gtfs_accessibility import (  # noqa: E402
    CLASS_A,
    CLASS_B,
    CLASS_C,
    CLASS_D,
    CLASS_E,
)
from traveler_distillation.baselines.mnl import MNLModel  # noqa: E402
from traveler_distillation.config import load_yaml  # noqa: E402
from traveler_distillation.generators import resolve_combination  # noqa: E402
from traveler_distillation.student import (  # noqa: E402
    build_baseline_index,
    collate_batch,
    make_counterfactual_pairs,
)

B = 2000
SEED = 42
CLASS_ORDER = [CLASS_A, CLASS_B, CLASS_C, CLASS_D, CLASS_E]
CLASS_LABELS = {CLASS_A: "A_excellent", CLASS_B: "B_good", CLASS_C: "C_moderate",
                CLASS_D: "D_poor", CLASS_E: "E_infeasible"}
FROZEN_S9_CKPT = _ROOT / "releases" / "s9_supply_aware_v2" / "checkpoint" / "model.pt"
ACC_DATASET = _ROOT / "data" / "singapore_accessibility" / "states_with_teacher.jsonl"
ACC_RECORDS = _ROOT / "data" / "singapore_accessibility" / "records.jsonl"
SPLIT_MANIFEST = _ROOT / "data" / "singapore_accessibility" / "split_manifest.json"
FROZEN_ACC_EVAL = _ROOT / "outputs" / "s9_accessibility_eval" / "eval_metrics.json"
FROZEN_REG_EVAL = _ROOT / "outputs" / "s9_regression" / "eval_metrics.json"
BASE_DATASET = _ROOT / "data" / "student_v0_3_s3" / "aggregated_teacher_dataset.jsonl"
JOINT_K5 = _ROOT / "data" / "student_s5_joint" / "aggregated_teacher_dataset_k5.jsonl"
JOINT_CONFIG = _ROOT / "configs" / "joint_sampling.yaml"
S3_SPLIT = _ROOT / "outputs" / "student_v0_3_s3_c" / "split_manifest.json"
MNLS_COEFS = _ROOT / "outputs" / "e1_mnl" / "mnl_b_coefs.json"
OUT_DIR = _ROOT / "outputs" / "e1_mnl"


# ----------------------------------------------------------------------------
# predictors
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_s9(model, extractor, state, device) -> dict[str, float]:
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


def load_s9(device: str):
    ck = torch.load(FROZEN_S9_CKPT, map_location=device)
    extractor = S8FeatureExtractor().from_state_dict(ck["extractor_state"])
    model = TravelerStudentS8(ck["config"], ck["feature_spec"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, extractor


# ----------------------------------------------------------------------------
# B1 / B2 — accessibility benchmark mirror
# ----------------------------------------------------------------------------
def _model_report(preds: dict[str, dict[str, float]], test_targets, class_of, curve_of) -> dict:
    def pt_prob(p: dict) -> float:
        return p.get("pt", 0.0)

    m_report = {"fidelity": {}, "fvr": {}, "convenience_error": {},
                "pt_prob_by_class": {}, "sensitivity": {}, "monotonicity": {},
                "mode_accuracy": None}
    all_kl, all_l1, all_mae, all_acc = [], [], [], []
    fvr_vals, pt_infeasible = [], []
    per_class = defaultdict(lambda: {"kl": [], "l1": [], "mae": [], "conv": [], "pt": []})
    for t in test_targets:
        sid = t.sample_id
        tp = t.teacher_aggregate.mode_probabilities
        sp = preds[sid]
        cls = class_of[sid]
        k = _kl(tp, sp)
        l = _l1(tp, sp)
        mae = abs(pt_prob(tp) - pt_prob(sp))
        all_kl.append(k)
        all_l1.append(l)
        all_mae.append(mae)
        all_acc.append(1.0 if max(sp, key=sp.get) == t.teacher_aggregate.selected_mode else 0.0)
        per_class[cls]["kl"].append(k)
        per_class[cls]["l1"].append(l)
        per_class[cls]["mae"].append(mae)
        per_class[cls]["conv"].append(mae)
        per_class[cls]["pt"].append(pt_prob(sp))
        if cls == CLASS_E:
            fvr_vals.append(1.0 if max(sp, key=sp.get) == "pt" else 0.0)
            pt_infeasible.append(pt_prob(sp))
    m_report["fidelity"]["all"] = {
        "kl": _bootstrap(np.array(all_kl)),
        "probability_l1": _bootstrap(np.array(all_l1)),
        "pt_probability_mae": _bootstrap(np.array(all_mae)),
    }
    m_report["mode_accuracy"] = _bootstrap(np.array(all_acc))
    for cls in CLASS_ORDER:
        if per_class[cls]["kl"]:
            m_report["fidelity"][CLASS_LABELS[cls]] = {
                "n": len(per_class[cls]["kl"]),
                "kl": _bootstrap(np.array(per_class[cls]["kl"])),
                "probability_l1": _bootstrap(np.array(per_class[cls]["l1"])),
                "pt_probability_mae": _bootstrap(np.array(per_class[cls]["mae"])),
            }
            m_report["convenience_error"][CLASS_LABELS[cls]] = _bootstrap(np.array(per_class[cls]["conv"]))
            m_report["pt_prob_by_class"][CLASS_LABELS[cls]] = _bootstrap(np.array(per_class[cls]["pt"]))
    if fvr_vals:
        m_report["fvr"] = {
            "rate": _bootstrap(np.array(fvr_vals)),
            "mean_P_pt_infeasible": _bootstrap(np.array(pt_infeasible)),
            "n_infeasible": len(fvr_vals),
        }
    by_group = defaultdict(lambda: defaultdict(list))
    for sid, p in preds.items():
        by_group[curve_of[sid]][class_of[sid]].append(pt_prob(p))
    deltas, mono_pairs, mono_triplets = [], [], []
    for gid, cls_probs in by_group.items():
        present = [c for c in CLASS_ORDER if c in cls_probs]
        means = {c: float(np.mean(cls_probs[c])) for c in present}
        if len(present) >= 2:
            deltas.append(means[present[0]] - means[present[-1]])
        for c1, c2 in zip(present, present[1:]):
            mono_pairs.append(1.0 if means[c1] > means[c2] else 0.0)
        for c1, c2, c3 in zip(present, present[1:], present[2:]):
            mono_triplets.append(1.0 if means[c1] > means[c2] > means[c3] else 0.0)
    m_report["sensitivity"] = {
        "delta_P_pt_best_minus_worst": _bootstrap(np.array(deltas)),
        "n_groups": len(deltas),
    }
    m_report["monotonicity"] = {
        "pair_agreement": _bootstrap(np.array(mono_pairs)) if mono_pairs else None,
        "triplet_agreement": _bootstrap(np.array(mono_triplets)) if mono_triplets else None,
        "n_pairs": len(mono_pairs), "n_triplets": len(mono_triplets),
        "raw_pairs": [float(v) for v in mono_pairs],
        "raw_triplets": [float(v) for v in mono_triplets],
    }
    return m_report


def _paired_delta(a: np.ndarray, b: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    d = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(a), size=len(a))
        d[i] = (a[idx] - b[idx]).mean()
    return {"mean": round(float((a - b).mean()), 4),
            "ci_low": round(float(np.percentile(d, 2.5)), 4),
            "ci_high": round(float(np.percentile(d, 97.5)), 4),
            "ci_excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}


def run_accessibility_eval(mnl: MNLModel, device: str) -> dict:
    targets = _load(ACC_DATASET)
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    test_personas = set(manifest["persona_split"]["test"])
    test_ods = set(manifest["od_split"]["test"])
    records = {}
    for line in ACC_RECORDS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["sample_id"]] = r
    test_targets = [t for t in targets if t.persona_group_id in test_personas]
    for t in test_targets:
        assert records[t.sample_id]["od_index"] in test_ods
        assert records[t.sample_id]["persona_id"] in test_personas
    print(f"B1/B2 test states: {len(test_targets)} (unseen personas x unseen ODs asserted)")

    class_of = {t.sample_id: records[t.sample_id]["accessibility_class"] for t in test_targets}
    curve_of = {t.sample_id: records[t.sample_id]["curve_group"] for t in test_targets}

    mnl_preds = {t.sample_id: mnl.predict_probs(t.state) for t in test_targets}
    model_s9, ext_s9 = load_s9(device)
    s9_preds = {t.sample_id: predict_s9(model_s9, ext_s9, t.state, device) for t in test_targets}

    mnl_report = _model_report(mnl_preds, test_targets, class_of, curve_of)
    s9_report = _model_report(s9_preds, test_targets, class_of, curve_of)

    # G2: recomputed S9 must match the frozen eval at 4-decimal rounding
    frozen = json.loads(FROZEN_ACC_EVAL.read_text(encoding="utf-8"))
    f9 = frozen["models"]["B1_S8"]
    checks = {}

    def _cmp(prefix: str, fresh: dict | None, frozen_v: dict | None):
        if fresh is None or frozen_v is None:
            checks[prefix] = {"fresh": fresh, "frozen": frozen_v, "pass": fresh == frozen_v}
            return
        for key in ("mean",):
            a, b = round(float(fresh.get(key, float("nan"))), 4), round(float(frozen_v.get(key, float("nan"))), 4)
            checks[f"{prefix}.{key}"] = {"fresh": a, "frozen": b, "pass": a == b}

    _cmp("fidelity.all.kl", s9_report["fidelity"]["all"]["kl"], f9["fidelity"]["all"]["kl"])
    _cmp("fidelity.all.l1", s9_report["fidelity"]["all"]["probability_l1"], f9["fidelity"]["all"]["probability_l1"])
    _cmp("fidelity.all.pt_mae", s9_report["fidelity"]["all"]["pt_probability_mae"], f9["fidelity"]["all"]["pt_probability_mae"])
    _cmp("fvr.rate", s9_report["fvr"]["rate"], f9["fvr"]["rate"])
    _cmp("fvr.mean_P_pt_inf", s9_report["fvr"]["mean_P_pt_infeasible"], f9["fvr"]["mean_P_pt_infeasible"])
    _cmp("sensitivity", s9_report["sensitivity"]["delta_P_pt_best_minus_worst"], f9["sensitivity"]["delta_P_pt_best_minus_worst"])
    _cmp("mono.pair", s9_report["monotonicity"]["pair_agreement"], f9["monotonicity"]["pair_agreement"])
    _cmp("mono.triplet", s9_report["monotonicity"]["triplet_agreement"], f9["monotonicity"]["triplet_agreement"])
    for cls in CLASS_LABELS.values():
        _cmp(f"fidelity.{cls}.kl", s9_report["fidelity"][cls]["kl"], f9["fidelity"][cls]["kl"])
        _cmp(f"fidelity.{cls}.l1", s9_report["fidelity"][cls]["probability_l1"], f9["fidelity"][cls]["probability_l1"])
        _cmp(f"fidelity.{cls}.pt_mae", s9_report["fidelity"][cls]["pt_probability_mae"], f9["fidelity"][cls]["pt_probability_mae"])
        _cmp(f"pt_prob.{cls}", s9_report["pt_prob_by_class"][cls], f9["pt_prob_by_class"][cls])
    g2_pass = all(c.get("pass", False) for c in checks.values())

    # paired deltas MNL - S9 (recomputed)
    sid_order = [t.sample_id for t in test_targets]
    mae_mnl = np.array([abs(t.teacher_aggregate.mode_probabilities.get("pt", 0.0) - mnl_preds[s].get("pt", 0.0))
                        for s, t in ((x.sample_id, x) for x in test_targets)])
    mae_s9 = np.array([abs(t.teacher_aggregate.mode_probabilities.get("pt", 0.0) - s9_preds[s].get("pt", 0.0))
                       for s, t in ((x.sample_id, x) for x in test_targets)])
    e_sids = [s for s in sid_order if class_of[s] == CLASS_E]
    deltas = {
        "pt_probability_mae_delta": _paired_delta(mae_mnl, mae_s9),
        "mean_P_pt_infeasible_delta": _paired_delta(
            np.array([mnl_preds[s].get("pt", 0.0) for s in e_sids]),
            np.array([s9_preds[s].get("pt", 0.0) for s in e_sids])),
        "fvr_rate_delta": _paired_delta(
            np.array([1.0 if max(mnl_preds[s], key=mnl_preds[s].get) == "pt" else 0.0 for s in e_sids]),
            np.array([1.0 if max(s9_preds[s], key=s9_preds[s].get) == "pt" else 0.0 for s in e_sids])),
        "pair_monotonicity_delta": _paired_delta(
            np.array(mnl_report["monotonicity"]["raw_pairs"]),
            np.array(s9_report["monotonicity"]["raw_pairs"])),
        "note": "positive MAE/P(PT|inf)/FVR delta = MNL worse than S9",
    }

    out = {
        "benchmark": "B1/B2 accessibility (51 test states, unseen persona x unseen OD)",
        "n_test_states": len(test_targets),
        "bootstrap": {"B": B, "seed": SEED},
        "models": {"MNL_B": mnl_report, "S9_recomputed": s9_report},
        "g2_consistency_gate": {"pass": g2_pass, "checks": checks},
        "deltas_mnl_vs_s9": deltas,
        "frozen_reference": {
            "S9": {k: f9[k] for k in ("fidelity", "fvr", "pt_prob_by_class", "sensitivity", "monotonicity")},
            "Teacher": frozen["models"]["Teacher"],
        },
        "s9_checkpoint": str(FROZEN_S9_CKPT),
        "mnl_coefs": str(MNLS_COEFS),
    }
    (OUT_DIR / "eval_accessibility.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"B1/B2 done; G2 gate {'PASS' if g2_pass else 'FAIL'}; wrote eval_accessibility.json")
    return out


# ----------------------------------------------------------------------------
# B3 — regression benchmark mirror
# ----------------------------------------------------------------------------
def run_regression_eval(mnl: MNLModel) -> dict:
    combos = load_yaml(JOINT_CONFIG).get("joint_combinations", [])
    manifest = json.loads(S3_SPLIT.read_text(encoding="utf-8"))
    test_personas = set(manifest["personas"]["test"])
    base = _load(BASE_DATASET)
    joint = _load(JOINT_K5)

    def _pid(s):
        return s.persona_group_id or s.state.persona.persona_id

    legacy_test = [s for s in base if _pid(s) in test_personas]
    test_joint = [s for s in joint if _pid(s) in test_personas]
    seen_joint = [s for s in test_joint if (resolve_combination(s.perturbation.joint_axes, combos) or {}).get("seen_in_training", True)]
    unseen_joint = [s for s in test_joint if not (resolve_combination(s.perturbation.joint_axes, combos) or {}).get("seen_in_training", True)]
    print(f"B3 legacy={len(legacy_test)} seen_joint={len(seen_joint)} unseen_joint={len(unseen_joint)}")

    s3_by_group = defaultdict(list)
    for s in base:
        s3_by_group[s.split_group_id].append(s)

    def predict(s):
        return mnl.predict_probs(s.state)

    per_state = []
    for s in legacy_test:
        t = s.teacher_aggregate.mode_probabilities
        p = predict(s)
        per_state.append({
            "acc": 1.0 if max(p, key=p.get) == s.teacher_aggregate.selected_mode else 0.0,
            "kl": _kl(t, p),
            "l1": _l1(t, p),
        })
    acc = np.array([r["acc"] for r in per_state])
    kl = np.array([r["kl"] for r in per_state])
    l1 = np.array([r["l1"] for r in per_state])
    base_index = build_baseline_index(legacy_test)
    pairs = make_counterfactual_pairs(legacy_test, base_index)
    dp_gaps, sign_agree = [], []
    for b, c in pairs:
        tb, tc = b.teacher_aggregate.mode_probabilities, c.teacher_aggregate.mode_probabilities
        pb, pc = predict(b), predict(c)
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
        "kl": _bootstrap(kl),
        "probability_l1": _bootstrap(l1),
        "delta_p_gap": _bootstrap(np.array(dp_gaps)),
        "sign_agreement": _bootstrap(np.array(sign_agree)),
    }

    def joint_metrics(samples):
        kl_v, l1_v = [], []
        for s in samples:
            t = s.teacher_aggregate.mode_probabilities
            p = predict(s)
            kl_v.append(_kl(t, p))
            l1_v.append(_l1(t, p))
        return (np.array(kl_v), np.array(l1_v)) if kl_v else (np.array([]), np.array([]))

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
        P0_s = predict(base_s)
        Pis_s = [predict(x) for x in singles]
        Pij_s = predict(s)
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

    mnl_row = {
        "legacy": legacy,
        "seen_joint": {"n": len(seen_joint),
                       "kl": _bootstrap(seen_kl) if len(seen_kl) else None,
                       "l1": _bootstrap(seen_l1) if len(seen_l1) else None},
        "unseen_joint": {"n": len(unseen_joint),
                         "kl": _bootstrap(unseen_kl) if len(unseen_kl) else None,
                         "l1": _bootstrap(unseen_l1) if len(unseen_l1) else None},
        "interaction_l1_error": _bootstrap(np.array(inter_errs)) if len(inter_errs) else None,
    }
    frozen = json.loads(FROZEN_REG_EVAL.read_text(encoding="utf-8"))
    out = {
        "benchmark": "B3 regression (frozen S3 legacy 226 + S5 joint 72/24, Teacher labels frozen)",
        "bootstrap": {"B": B, "seed": SEED, "unit": "state (paired over states)"},
        "models": {"MNL_B": mnl_row},
        "frozen_reference": {"S9": frozen["models"]["S8"]},
        "mnl_coefs": str(MNLS_COEFS),
    }
    (OUT_DIR / "eval_regression.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("B3 done; wrote eval_regression.json")
    return out


def main() -> int:
    device = "cpu"
    coefs = json.loads(MNLS_COEFS.read_text(encoding="utf-8"))
    mnl = MNLModel.from_json(coefs)
    print(f"MNL-B spec {mnl.spec}, k={len(mnl.theta)} (frozen {MNLS_COEFS})")
    acc_out = run_accessibility_eval(mnl, device)
    reg_out = run_regression_eval(mnl)

    print("\n=== T1/T2 summary (mean, 95% CI) ===")
    m = acc_out["models"]["MNL_B"]
    print("MNL KL={} L1={} PT-MAE={} FVR={} P(PT|inf)={} acc={}".format(
        m["fidelity"]["all"]["kl"]["mean"], m["fidelity"]["all"]["probability_l1"]["mean"],
        m["fidelity"]["all"]["pt_probability_mae"]["mean"], m["fvr"]["rate"]["mean"],
        m["fvr"]["mean_P_pt_infeasible"]["mean"], m["mode_accuracy"]["mean"]))
    print("MNL dP={} pair={} triplet={}".format(
        m["sensitivity"]["delta_P_pt_best_minus_worst"]["mean"],
        m["monotonicity"]["pair_agreement"]["mean"], m["monotonicity"]["triplet_agreement"]["mean"]))
    print("G2 gate:", acc_out["g2_consistency_gate"]["pass"])
    print("\n=== T3 summary ===")
    r = reg_out["models"]["MNL_B"]
    print("legacy acc={} KL={} L1={} dPgap={} sign={}".format(
        r["legacy"]["mode_accuracy"]["mean"], r["legacy"]["kl"]["mean"],
        r["legacy"]["probability_l1"]["mean"], r["legacy"]["delta_p_gap"]["mean"],
        r["legacy"]["sign_agreement"]["mean"]))
    print("seen KL={} L1={} | unseen KL={} L1={} | inter_err={}".format(
        r["seen_joint"]["kl"]["mean"], r["seen_joint"]["l1"]["mean"],
        r["unseen_joint"]["kl"]["mean"], r["unseen_joint"]["l1"]["mean"],
        r["interaction_l1_error"]["mean"]))
    return 0 if acc_out["g2_consistency_gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
