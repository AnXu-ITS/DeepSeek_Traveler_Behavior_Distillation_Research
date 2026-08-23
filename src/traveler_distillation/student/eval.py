"""Shared evaluation helpers used by the train scripts.

Keeps per-persona generalization breakdown and counterfactual sign-agreement
in one place instead of duplicating them across train_student_v0_2_{a,b,c}.py.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from .dataset import collate_batch


def predict_probs(sample, model, extractor, device) -> dict[str, float]:
    """Student preference distribution for one sample (available modes only)."""
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


@torch.no_grad()
def persona_breakdown(samples, model, extractor, device) -> list[dict]:
    """Per-persona test metrics: mode accuracy + probability L1.

    Used to check whether the student generalizes uniformly across personas or
    only to the personas seen in training (persona-holdout evaluation).
    """
    model.eval()
    groups: dict[str, list] = defaultdict(list)
    for s in samples:
        groups[s.state.persona.persona_id].append(s)

    out = []
    for pid in sorted(groups):
        ss = groups[pid]
        acc = 0
        l1 = 0.0
        for s in ss:
            t = s.teacher_aggregate.mode_probabilities
            p = predict_probs(s, model, extractor, device)
            modes = set(t) | set(p)
            pred_mode = max(p, key=p.get)
            acc += 1 if pred_mode == s.teacher_aggregate.selected_mode else 0
            l1 += sum(abs(t.get(m, 0.0) - p.get(m, 0.0)) for m in modes)
        n = len(ss)
        out.append(
            {
                "persona_id": pid,
                "n_samples": n,
                "mode_accuracy": round(acc / n, 4),
                "probability_l1": round(l1 / n, 4),
            }
        )
    return out


@torch.no_grad()
def counterfactual_sign_agreement(samples, model, extractor, device) -> dict:
    """Per-mode sign agreement of DeltaP between teacher and student (evaluation).

    Resolves each counterfactual's baseline via ``baseline_sample_id``, computes
    teacher DeltaP and student DeltaP per mode, and measures how often their
    signs agree (teacher-not-moved modes count as agreeing).
    """
    model.eval()
    by_id = {s.sample_id: s for s in samples}
    totals = {"agree": 0, "n": 0}
    per_pair = []
    for s in samples:
        if s.perturbation.axis == "baseline":
            continue
        base = by_id.get(s.baseline_sample_id)
        if base is None:
            continue
        t_base = base.teacher_aggregate.mode_probabilities
        t_cf = s.teacher_aggregate.mode_probabilities
        s_base = predict_probs(base, model, extractor, device)
        s_cf = predict_probs(s, model, extractor, device)
        modes = set(t_base) | set(t_cf) | set(s_base) | set(s_cf)
        agree = 0
        for m in modes:
            dt = t_cf.get(m, 0.0) - t_base.get(m, 0.0)
            ds = s_cf.get(m, 0.0) - s_base.get(m, 0.0)
            if abs(dt) < 1e-3 or (dt > 0) == (ds > 0):
                agree += 1
        pair_agree = agree / len(modes) if modes else 1.0
        totals["agree"] += pair_agree
        totals["n"] += 1
        per_pair.append(
            {
                "sample_id": s.sample_id,
                "axis": s.perturbation.axis,
                "level": s.perturbation.level,
                "sign_agreement": round(pair_agree, 4),
            }
        )

    mean = totals["agree"] / totals["n"] if totals["n"] else 0.0
    return {"mean_sign_agreement": round(mean, 4), "n_pairs": totals["n"], "per_pair": per_pair}
