#!/usr/bin/env python
"""S7 causal-repair evaluation: test-only quadruplets + paired bootstrap CI.

HARD CONSTRAINTS (user-mandated):
  * Only split == "test" quadruplets are evaluated here (8 car-available groups
    per axis). This set was never used for training or hyperparameter/model
    selection.
  * Teacher targets are the S6 targets reused verbatim (no API calls).
  * PRIMARY metrics = Teacher-Student effect gaps G_nat / G_broken / G_med
    (S7 instruction §21, elevated to primary by the user);
    SECONDARY metrics = R_shortcut / R_mediator and their gaps.
  * All reported aggregates carry 95% paired bootstrap CIs (B=2000 over
    quadruplet groups, fixed seed) — user-mandated.

Models: Teacher (S6 targets), C0 (pre-S5), C1 (S5 joint M2 = S7 start point)
and every --variant name=path (S7 fine-tunes).

Usage:
    python scripts/eval_s7_causal_repair.py \
        --mechanism-dataset data/student_s7_mechanism/quadruplets.jsonl \
        --c0 outputs/student_v0_3_s3_c/checkpoints/best.pt \
        --c1 outputs/student_s5_joint_m2/checkpoints/best.pt \
        --variant W1=outputs/student_s7_w1/checkpoints/best.pt \
        --variant W2=outputs/student_s7_w2/checkpoints/best.pt \
        --variant W3=outputs/student_s7_w3/checkpoints/best.pt \
        --variant mech_only=outputs/student_s7_mech_only/checkpoints/best.pt \
        --variant broken_only=outputs/student_s7_broken_only/checkpoints/best.pt \
        --output outputs/s7_causal_eval/eval_metrics.json
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

from traveler_distillation.student import (
    FeatureExtractor,
    TravelerStudent,
    collate_batch,
    load_mechanism_quadruplets,
)

_EPS = 1e-6
B = 2000
SEED = 42


def _load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    extractor = FeatureExtractor().from_state_dict(ckpt["extractor_state"])
    model = TravelerStudent(ckpt.get("config", {}), ckpt["feature_spec"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, extractor


@torch.no_grad()
def _predict(state, model, extractor, device) -> dict[str, float]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism-dataset", default="data/student_s7_mechanism/quadruplets.jsonl")
    ap.add_argument("--c0", default="outputs/student_v0_3_s3_c/checkpoints/best.pt")
    ap.add_argument("--c1", default="outputs/student_s5_joint_m2/checkpoints/best.pt")
    ap.add_argument("--variant", action="append", default=[],
                    help="name=path of an S7 checkpoint (repeatable)")
    ap.add_argument("--output", default="outputs/s7_causal_eval/eval_metrics.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    quads = load_mechanism_quadruplets(args.mechanism_dataset, split="test")
    assert all(q.split == "test" for q in quads), "non-test quadruplets leaked into final eval"
    assert len(quads) == 16, f"unexpected test quadruplet count: {len(quads)}"

    models: dict[str, object] = {
        "C0_preS5": _load_model(Path(args.c0), device),
        "C1_S5": _load_model(Path(args.c1), device),
    }
    for spec in args.variant:
        name, _, path = spec.partition("=")
        models[name] = _load_model(Path(path), device)

    # per-group per-model metrics (quadruplet group = bootstrap unit)
    group_metrics = defaultdict(lambda: defaultdict(dict))  # model -> axis -> {gid: dict}
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
        tmode = max(PA_t, key=PA_t.get)
        for name, (model, extractor) in models.items():
            PA = _predict(members["baseline"].state, model, extractor, device)
            PB = _predict(members["natural"].state, model, extractor, device)
            PC = _predict(members["broken"].state, model, extractor, device)
            PD = _predict(members["mediator"].state, model, extractor, device)
            En = _l1(PB, PA)
            Eb = _l1(PC, PA)
            Em = _l1(PD, PA)
            Rs = Eb / (En + _EPS)
            Rm = Em / (En + _EPS)
            m = {
                "E_natural": En, "E_broken": Eb, "E_mediator": Em,
                "R_shortcut": Rs, "R_mediator": Rm,
                "dP_target_B": PB.get(tmode, 0.0) - PA.get(tmode, 0.0),
                "dP_target_C": PC.get(tmode, 0.0) - PA.get(tmode, 0.0),
                "dP_target_D": PD.get(tmode, 0.0) - PA.get(tmode, 0.0),
            }
            group_metrics[name][axis][gid] = m
        # teacher entry
        En_t = _l1(PB_t, PA_t)
        Eb_t = _l1(PC_t, PA_t)
        Em_t = _l1(PD_t, PA_t)
        group_metrics["Teacher"][axis][gid] = {
            "E_natural": En_t, "E_broken": Eb_t, "E_mediator": Em_t,
            "R_shortcut": Eb_t / (En_t + _EPS), "R_mediator": Em_t / (En_t + _EPS),
            "dP_target_B": PB_t.get(tmode, 0.0) - PA_t.get(tmode, 0.0),
            "dP_target_C": PC_t.get(tmode, 0.0) - PA_t.get(tmode, 0.0),
            "dP_target_D": PD_t.get(tmode, 0.0) - PA_t.get(tmode, 0.0),
        }

    GAP_FIELDS = ("G_nat", "G_broken", "G_med", "Gap_shortcut", "Gap_mediator")
    report = {"n_test_groups_per_axis": {ax: len(g) for ax, g in axes.items()},
              "bootstrap": {"B": B, "seed": SEED, "unit": "quadruplet group (paired over groups)"},
              "models": {}, "deltas_vs_c1": {}}
    for name in list(models) + ["Teacher"]:
        model_report = {}
        for axis, gids in axes.items():
            rows = [group_metrics[name][axis][g] for g in gids]
            # effect + ratio aggregates (secondary for ratios)
            out = {}
            for field in ("E_natural", "E_broken", "E_mediator", "R_shortcut", "R_mediator"):
                out[field] = _bootstrap(np.array([r[field] for r in rows]))
            # PRIMARY: teacher-student effect gaps
            t_rows = [group_metrics["Teacher"][axis][g] for g in gids]
            gap = {
                "G_nat": np.abs(np.array([r["E_natural"] for r in rows]) - np.array([r["E_natural"] for r in t_rows])),
                "G_broken": np.abs(np.array([r["E_broken"] for r in rows]) - np.array([r["E_broken"] for r in t_rows])),
                "G_med": np.abs(np.array([r["E_mediator"] for r in rows]) - np.array([r["E_mediator"] for r in t_rows])),
                "Gap_shortcut": np.abs(np.array([r["R_shortcut"] for r in rows]) - np.array([r["R_shortcut"] for r in t_rows])),
                "Gap_mediator": np.abs(np.array([r["R_mediator"] for r in rows]) - np.array([r["R_mediator"] for r in t_rows])),
            }
            for field in GAP_FIELDS:
                out[field] = _bootstrap(gap[field])
            for field in ("dP_target_B", "dP_target_C", "dP_target_D"):
                out[field] = _bootstrap(np.array([r[field] for r in rows]))
            out["n_groups"] = len(rows)
            model_report[axis] = out
        report["models"][name] = model_report

    # paired deltas vs C1 (improvement evidence; same groups in both models)
    c1_rows = {axis: [group_metrics["C1_S5"][axis][g] for g in gids] for axis, gids in axes.items()}
    for name in models:
        if name in ("C0_preS5", "C1_S5", "Teacher"):
            continue
        model_report = {}
        for axis, gids in axes.items():
            rows = [group_metrics[name][axis][g] for g in gids]
            base = c1_rows[axis]
            t_rows = [group_metrics["Teacher"][axis][g] for g in gids]
            out = {}
            for field in GAP_FIELDS:
                if field.startswith("Gap_"):
                    a = np.abs(np.array([r[field.replace("Gap_", "R_")] for r in rows]) - np.array([r[field.replace("Gap_", "R_")] for r in t_rows]))
                    b = np.abs(np.array([r[field.replace("Gap_", "R_")] for r in base]) - np.array([r[field.replace("Gap_", "R_")] for r in t_rows]))
                else:
                    src = {"G_nat": "E_natural", "G_broken": "E_broken", "G_med": "E_mediator"}[field]
                    a = np.abs(np.array([r[src] for r in rows]) - np.array([r[src] for r in t_rows]))
                    b = np.abs(np.array([r[src] for r in base]) - np.array([r[src] for r in t_rows]))
                out[field] = _bootstrap(a - b)  # negative = gap shrank vs C1
            out["E_natural"] = _bootstrap(np.array([r["E_natural"] for r in rows]) - np.array([r["E_natural"] for r in base]))
            out["R_shortcut"] = _bootstrap(np.array([r["R_shortcut"] for r in rows]) - np.array([r["R_shortcut"] for r in base]))
            out["R_mediator"] = _bootstrap(np.array([r["R_mediator"] for r in rows]) - np.array([r["R_mediator"] for r in base]))
            model_report[axis] = out
        report["deltas_vs_c1"][name] = model_report

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
