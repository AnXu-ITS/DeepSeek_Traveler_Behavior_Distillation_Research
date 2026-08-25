#!/usr/bin/env python
"""S6 causal mechanism audit evaluation.

For each (persona, trip, axis) four-state group {A,B,C,D}, compute the causal
mechanism diagnostics:

  E_natural  = ||P_B - P_A||_1      (natural intervention effect)
  E_broken   = ||P_C - P_A||_1      (broken-path effect; shortcut evidence)
  E_mediator = ||P_D - P_A||_1      (mediator-only effect; mechanism evidence)
  R_shortcut = E_broken  / (E_natural + eps)    (~0 = mechanism, ~1 = shortcut)
  R_mediator = E_mediator / (E_natural + eps)   (~1 = mediator-driven, ~0 = not)

and the target-mode delta P (P_X(mode) - P_A(mode) for X in B/C/D).

Models compared: Teacher (from state targets; A/B reused from S3, C/D filled by
run_teacher_causal_audit.py) and Students C0 (pre-S5) / C1 (S5 multi-axis).

Usage:
    python scripts/eval_causal_mechanism.py \
        --states data/causal_audit/states.jsonl \
        --c0 outputs/student_v0_3_s3_c/checkpoints/best.pt \
        --c1 outputs/student_s5_joint_m2/checkpoints/best.pt \
        --output outputs/causal_audit/eval_metrics.json
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

from traveler_distillation.schemas.state import UniversalTravelerState
from traveler_distillation.student import FeatureExtractor, TravelerStudent, collate_batch

_EPS = 1e-6


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    extractor = FeatureExtractor().from_state_dict(ckpt["extractor_state"])
    model = TravelerStudent(ckpt.get("config", {}), ckpt["feature_spec"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, extractor


@torch.no_grad()
def _predict(state: UniversalTravelerState, model, extractor, device) -> dict[str, float]:
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
    s_out = model(batch)["mode_probabilities"][0].cpu().numpy()
    return {alt.mode: float(s_out[i]) for i, alt in enumerate(state.alternatives) if alt.available}


def _l1(p, q) -> float:
    modes = set(p) | set(q)
    return sum(abs(p.get(m, 0.0) - q.get(m, 0.0)) for m in modes)


def _effect(p_from, p_base) -> float:
    return _l1(p_from, p_base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="data/causal_audit/states.jsonl")
    ap.add_argument("--c0", default="outputs/student_v0_3_s3_c/checkpoints/best.pt")
    ap.add_argument("--c1", default="outputs/student_s5_joint_m2/checkpoints/best.pt")
    ap.add_argument("--output", default="outputs/causal_audit/eval_metrics.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = _load_jsonl(Path(args.states))

    # group by audit_group_id -> {state_type: record}
    groups = defaultdict(dict)
    for r in records:
        groups[r["audit_group_id"]][r["state_type"]] = r

    # student models
    students = {
        "C0_preS5": _load_model(Path(args.c0), device),
        "C1_S5": _load_model(Path(args.c1), device),
    }

    # per-axis accumulators
    def _new_accum():
        return {"E_natural": [], "E_broken": [], "E_mediator": [],
                "R_shortcut": [], "R_mediator": [], "dP_target": {"B": [], "C": [], "D": []}}

    model_accum = defaultdict(lambda: defaultdict(_new_accum))  # model -> axis -> accum

    n_teacher_complete = 0
    n_groups = 0
    n_skipped_unavailable = 0
    for gid, states in groups.items():
        if not all(k in states for k in ("baseline", "natural", "broken", "mediator")):
            continue
        A = UniversalTravelerState.model_validate(states["baseline"]["state"])
        B = UniversalTravelerState.model_validate(states["natural"]["state"])
        C = UniversalTravelerState.model_validate(states["broken"]["state"])
        D = UniversalTravelerState.model_validate(states["mediator"]["state"])
        axis = states["baseline"]["axis_id"]
        tmode = states["baseline"]["target_mode"]

        # The audit is only meaningful when the target mode is an available
        # option for this persona: a persona without a car has E_natural == 0
        # under parking/congestion interventions, which explodes the ratios.
        if tmode not in A.available_modes:
            n_skipped_unavailable += 1
            continue
        n_groups += 1

        # --- teacher (from targets) ---
        tA = states["baseline"]["teacher_aggregate"]
        tB = states["natural"]["teacher_aggregate"]
        tC = states["broken"]["teacher_aggregate"]
        tD = states["mediator"]["teacher_aggregate"]
        if all(t is not None for t in (tA, tB, tC, tD)):
            n_teacher_complete += 1
            PA = tA["mode_probabilities"]; PB = tB["mode_probabilities"]
            PC = tC["mode_probabilities"]; PD = tD["mode_probabilities"]
            _accumulate(model_accum["Teacher"][axis], PA, PB, PC, PD, tmode)

        # --- students ---
        for name, (model, extractor) in students.items():
            PA = _predict(A, model, extractor, device)
            PB = _predict(B, model, extractor, device)
            PC = _predict(C, model, extractor, device)
            PD = _predict(D, model, extractor, device)
            _accumulate(model_accum[name][axis], PA, PB, PC, PD, tmode)

    def _mean(v):
        return round(sum(v) / len(v), 4) if v else None

    # aggregate into report
    report = {"n_groups": n_groups, "n_skipped_unavailable": n_skipped_unavailable,
              "n_teacher_complete": n_teacher_complete, "models": {}}
    for model_name, axis_map in model_accum.items():
        model_out = {}
        for axis, acc in axis_map.items():
            model_out[axis] = {
                "E_natural": _mean(acc["E_natural"]),
                "E_broken": _mean(acc["E_broken"]),
                "E_mediator": _mean(acc["E_mediator"]),
                "R_shortcut": _mean(acc["R_shortcut"]),
                "R_mediator": _mean(acc["R_mediator"]),
                "dP_target_B": _mean(acc["dP_target"]["B"]),
                "dP_target_C": _mean(acc["dP_target"]["C"]),
                "dP_target_D": _mean(acc["dP_target"]["D"]),
                "n": len(acc["E_natural"]),
            }
        report["models"][model_name] = model_out

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {out}")
    return 0


def _accumulate(acc, PA, PB, PC, PD, tmode):
    En = _effect(PB, PA)
    Eb = _effect(PC, PA)
    Em = _effect(PD, PA)
    acc["E_natural"].append(En)
    acc["E_broken"].append(Eb)
    acc["E_mediator"].append(Em)
    acc["R_shortcut"].append(Eb / (En + _EPS))
    acc["R_mediator"].append(Em / (En + _EPS))
    acc["dP_target"]["B"].append(PB.get(tmode, 0.0) - PA.get(tmode, 0.0))
    acc["dP_target"]["C"].append(PC.get(tmode, 0.0) - PA.get(tmode, 0.0))
    acc["dP_target"]["D"].append(PD.get(tmode, 0.0) - PA.get(tmode, 0.0))


if __name__ == "__main__":
    raise SystemExit(main())
