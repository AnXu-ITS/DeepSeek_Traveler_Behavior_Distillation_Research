#!/usr/bin/env python
"""S8 accessibility evaluation (test-only, unseen personas AND unseen ODs).

Evaluates Teacher (reference), B0 = frozen S7-W3, and B1 = Student-S8 on the
S8 TEST split (double holdout by construction: unseen personas + unseen ODs)
with the S8 primary metrics (§20-22):

  * PT probability fidelity: KL, probability L1, PT-prob MAE (per class + all);
  * accessibility sensitivity: dP_PT = P(PT|best) - P(PT|worst) within each
    curve group (paired over groups);
  * monotonicity: pair/triplet agreement of P(PT) along the class ordering;
  * PT feasibility violation rate (FVR): share of infeasible states whose
    argmax mode is pt, plus mean P(PT|infeasible);
  * convenience error E_conv = |P_T(PT) - P_S(PT)| per class A-E.

All aggregates carry 95% paired bootstrap CIs (B=2000, seed=42).

Usage:
    python scripts/eval_s8_accessibility.py \
        --accessibility-dataset data/singapore_accessibility/states_with_teacher.jsonl \
        --s8-checkpoint outputs/student_s8/checkpoints/best.pt \
        --output outputs/s8_accessibility_eval
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

from traveler_distillation.accessibility.accessibility_features import (
    S8FeatureExtractor,
    TravelerStudentS8,
)
from traveler_distillation.accessibility.gtfs_accessibility import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    CLASS_D,
    CLASS_E,
)
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget
from traveler_distillation.student import FeatureExtractor, TravelerStudent, collate_batch

FROZEN_RELEASE_CKPT = _ROOT / "releases" / "s7_w3_generic_core_v1" / "checkpoint" / "model.pt"
B = 2000
SEED = 42
CLASS_ORDER = [CLASS_A, CLASS_B, CLASS_C, CLASS_D, CLASS_E]
CLASS_LABELS = {CLASS_A: "A_excellent", CLASS_B: "B_good", CLASS_C: "C_moderate",
                CLASS_D: "D_poor", CLASS_E: "E_infeasible"}


def _load(path: Path) -> list[AggregatedTeacherTarget]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(AggregatedTeacherTarget.model_validate(json.loads(line)))
    return samples


def _bootstrap(values: np.ndarray, n_boot: int = B, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        means[i] = values[idx].mean()
    return {"mean": round(float(values.mean()), 4),
            "ci_low": round(float(np.percentile(means, 2.5)), 4),
            "ci_high": round(float(np.percentile(means, 97.5)), 4)}


@torch.no_grad()
def _predict(model, extractor, state, device) -> dict[str, float]:
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
    ap.add_argument("--accessibility-dataset",
                    default="data/singapore_accessibility/states_with_teacher.jsonl")
    ap.add_argument("--accessibility-records",
                    default="data/singapore_accessibility/records.jsonl")
    ap.add_argument("--split-manifest", default="data/singapore_accessibility/split_manifest.json")
    ap.add_argument("--s8-checkpoint", default="outputs/student_s8/checkpoints/best.pt")
    ap.add_argument("--output", default="outputs/s8_accessibility_eval")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    targets = _load(Path(args.accessibility_dataset))
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    test_personas = set(manifest["persona_split"]["test"])
    test_ods = set(manifest["od_split"]["test"])
    records = {}
    for line in Path(args.accessibility_records).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["sample_id"]] = r

    # TEST split only: unseen personas AND unseen ODs (double holdout)
    test_targets = [t for t in targets if t.persona_group_id in test_personas]
    assert test_targets, "no test-split targets found"
    for t in test_targets:
        od = records[t.sample_id]["od_index"]
        assert od in test_ods, f"test state {t.sample_id} uses non-test OD {od}"
        assert records[t.sample_id]["persona_id"] in test_personas
    print(f"test states: {len(test_targets)} (unseen personas x unseen ODs asserted)")

    # models: Teacher (from targets), B0 = S7-W3 frozen, B1 = S8
    models = {"Teacher": None}
    ck_s7 = torch.load(FROZEN_RELEASE_CKPT, map_location=device)
    ext_s7 = FeatureExtractor().from_state_dict(ck_s7["extractor_state"])
    model_s7 = TravelerStudent(ck_s7["config"], ck_s7["feature_spec"]).to(device)
    model_s7.load_state_dict(ck_s7["model_state"])
    model_s7.eval()
    models["B0_S7W3"] = (model_s7, ext_s7)

    ck_s8 = torch.load(Path(args.s8_checkpoint), map_location=device)
    ext_s8 = S8FeatureExtractor().from_state_dict(ck_s8["extractor_state"])
    model_s8 = TravelerStudentS8(ck_s8["config"], ck_s8["feature_spec"]).to(device)
    model_s8.load_state_dict(ck_s8["model_state"])
    model_s8.eval()
    models["B1_S8"] = (model_s8, ext_s8)

    # ---- per-state predictions ----
    preds = {name: {} for name in models}
    class_of = {}
    curve_of = {}
    for t in test_targets:
        sid = t.sample_id
        class_of[sid] = records[sid]["accessibility_class"]
        curve_of[sid] = records[sid]["curve_group"]
        preds["Teacher"][sid] = t.teacher_aggregate.mode_probabilities
        for name in ("B0_S7W3", "B1_S8"):
            m, e = models[name]
            preds[name][sid] = _predict(m, e, t.state, device)

    def pt_prob(p: dict) -> float:
        return p.get("pt", 0.0)

    def kl(p, q) -> float:
        import math
        modes = set(p) | set(q)
        klv = 0.0
        for m in modes:
            pm = p.get(m, 0.0)
            qm = max(q.get(m, 0.0), 1e-9)
            if pm > 0:
                klv += pm * math.log(pm / qm)
        return klv

    def l1(p, q) -> float:
        modes = set(p) | set(q)
        return sum(abs(p.get(m, 0.0) - q.get(m, 0.0)) for m in modes)

    report: dict = {
        "n_test_states": len(test_targets),
        "n_test_personas": len(test_personas),
        "n_test_ods": len(test_ods),
        "bootstrap": {"B": B, "seed": SEED},
        "models": {},
    }

    # ---- fidelity + FVR + convenience per class ----
    for name in ("B0_S7W3", "B1_S8"):
        m_report = {"fidelity": {}, "fvr": {}, "convenience_error": {},
                    "pt_prob_by_class": {}, "sensitivity": {}, "monotonicity": {}}
        all_kl, all_l1, all_mae = [], [], []
        fvr_vals, pt_infeasible = [], []
        per_class = defaultdict(lambda: {"kl": [], "l1": [], "mae": [], "conv": [], "pt": []})
        for sid, t in ((s.sample_id, s) for s in test_targets):
            tp = t.teacher_aggregate.mode_probabilities
            sp = preds[name][sid]
            cls = class_of[sid]
            k = kl(tp, sp)
            l = l1(tp, sp)
            mae = abs(pt_prob(tp) - pt_prob(sp))
            all_kl.append(k)
            all_l1.append(l)
            all_mae.append(mae)
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

        # ---- accessibility sensitivity (paired within curve group) ----
        by_group = defaultdict(lambda: defaultdict(list))
        for sid in preds[name]:
            by_group[curve_of[sid]][class_of[sid]].append(pt_prob(preds[name][sid]))
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
        report["models"][name] = m_report

    # teacher reference for sensitivity/monotonicity/fidelity bounds
    t_report = {"pt_prob_by_class": {}, "sensitivity": {}, "monotonicity": {}}
    by_group = defaultdict(lambda: defaultdict(list))
    for sid, p in preds["Teacher"].items():
        by_group[curve_of[sid]][class_of[sid]].append(pt_prob(p))
    for cls in CLASS_ORDER:
        vals = [pt_prob(p) for sid, p in preds["Teacher"].items() if class_of[sid] == cls]
        if vals:
            t_report["pt_prob_by_class"][CLASS_LABELS[cls]] = _bootstrap(np.array(vals))
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
    t_report["sensitivity"] = {"delta_P_pt_best_minus_worst": _bootstrap(np.array(deltas)),
                               "n_groups": len(deltas)}
    t_report["monotonicity"] = {"pair_agreement": _bootstrap(np.array(mono_pairs)),
                                "triplet_agreement": _bootstrap(np.array(mono_triplets)),
                                "n_pairs": len(mono_pairs), "n_triplets": len(mono_triplets)}
    report["models"]["Teacher"] = t_report

    # ---- deltas B1_S8 vs B0_S7W3 (paired over states/groups) ----
    rng = np.random.default_rng(SEED)
    def _paired_delta(a: np.ndarray, b: np.ndarray) -> dict:
        d = np.empty(B)
        for i in range(B):
            idx = rng.integers(0, len(a), size=len(a))
            d[i] = (a[idx] - b[idx]).mean()
        return {"mean": round(float((a - b).mean()), 4),
                "ci_low": round(float(np.percentile(d, 2.5)), 4),
                "ci_high": round(float(np.percentile(d, 97.5)), 4),
                "ci_excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}

    sid_order = [t.sample_id for t in test_targets]
    b0, b1 = preds["B0_S7W3"], preds["B1_S8"]
    report["deltas_s8_vs_s7w3"] = {
        "pt_probability_mae_delta": _paired_delta(
            np.array([abs(pt_prob(t.teacher_aggregate.mode_probabilities) - pt_prob(b1[s])) for s, t in ((x.sample_id, x) for x in test_targets)]),
            np.array([abs(pt_prob(t.teacher_aggregate.mode_probabilities) - pt_prob(b0[s])) for s, t in ((x.sample_id, x) for x in test_targets)])),
        "mean_P_pt_infeasible_delta": _paired_delta(
            np.array([pt_prob(b1[s]) for s in sid_order if class_of[s] == CLASS_E]),
            np.array([pt_prob(b0[s]) for s in sid_order if class_of[s] == CLASS_E])),
        "fvr_rate_delta": _paired_delta(
            np.array([1.0 if max(b1[s], key=b1[s].get) == "pt" else 0.0 for s in sid_order if class_of[s] == CLASS_E]),
            np.array([1.0 if max(b0[s], key=b0[s].get) == "pt" else 0.0 for s in sid_order if class_of[s] == CLASS_E])),
        "pair_monotonicity_delta": _paired_delta(
            np.array(report["models"]["B1_S8"]["monotonicity"]["raw_pairs"]),
            np.array(report["models"]["B0_S7W3"]["monotonicity"]["raw_pairs"])),
        "note": "negative MAE/P(PT|inf)/FVR delta = S8 better; monotonicity delta is "
                "paired over per-group 0/1 pair-agreement indicators",
    }
    report["unseen_od"] = {
        "guarantee": "test split uses ONLY test-ODs and test-personas (asserted); train ODs never appear in test",
        "n_test_ods": len(test_ods),
        "od_holdout_verified": True,
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:6000])
    print(f"\nwrote {out / 'eval_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
