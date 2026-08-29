#!/usr/bin/env python
"""E1 S2 — estimate MNL-B on the frozen S9 Singapore train split (234 states).

Fits the three pre-registered specs (S1/S2/S3) by maximizing the expected
log-likelihood under the frozen Teacher mean-probability labels, selects the
spec on the frozen VAL split (51 states), runs the G1 estimation gate and the
G4 determinism gate, then freezes the coefficients to
outputs/e1_mnl/mnl_b_coefs.json.

Usage:
    python scripts/estimate_mnl.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from traveler_distillation.baselines.mnl import build_design, estimate, loglik_value
from traveler_distillation.dataset.aggregation import AggregatedTeacherTarget

DATASET = _ROOT / "data" / "singapore_accessibility" / "states_with_teacher.jsonl"
SPLIT_MANIFEST = _ROOT / "data" / "singapore_accessibility" / "split_manifest.json"
OUT_DIR = _ROOT / "outputs" / "e1_mnl"
SPECS = ("S1", "S2", "S3")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _load(path: Path) -> list[AggregatedTeacherTarget]:
    return [AggregatedTeacherTarget.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    targets = _load(DATASET)
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    train_personas = set(manifest["persona_split"]["train"])
    val_personas = set(manifest["persona_split"]["val"])
    train = [t for t in targets if t.persona_group_id in train_personas]
    val = [t for t in targets if t.persona_group_id in val_personas]
    assert len(train) == 234 and len(val) == 51, f"unexpected splits: {len(train)}/{len(val)}"
    print(f"train={len(train)} val={len(val)} (frozen S9 splits)")

    fits = {}
    for spec in SPECS:
        print(f"--- spec {spec} ---", flush=True)
        r = estimate(train, spec)
        _, names, _ = build_design(train[0].state, spec)
        val_ll = loglik_value(val, spec, r["theta"])
        train_ll = -r["loss"]
        print(f"  k={r['k']} train_ll={train_ll:.6f} val_ll={val_ll:.6f} "
              f"spread={r['restart_spread_max_abs']:.3e} cond={r['hess_cond']:.2e} "
              f"se_finite={r['se_finite']} pd={r['hess_pd']} blowup={r['blowup']}")
        fits[spec] = {**r, "names": names, "val_ll": val_ll, "train_ll": train_ll}

    # spec selection on VAL log-likelihood only, restricted to specs that pass
    # the identifiability checks (pd Hessian, finite SEs, no blowup) —
    # pre-registered rule; e.g. S2's commute terms are constant/absent in the
    # frozen data and must not compete.
    def _identifiable(r: dict) -> bool:
        return bool(r["hess_pd"] and r["se_finite"] and not r["blowup"])

    valid = {s: fits[s] for s in SPECS if _identifiable(fits[s])}
    assert valid, "no identifiable spec — review the specification"
    selected = max(valid, key=lambda s: fits[s]["val_ll"])
    invalid = {s: {"hess_pd": fits[s]["hess_pd"], "se_finite": fits[s]["se_finite"],
                   "blowup": fits[s]["blowup"]} for s in SPECS if s not in valid}
    print(f"selected spec by val log-likelihood: {selected} (identifiable specs: "
          f"{sorted(valid)}; excluded: {invalid})")

    # G4 determinism: re-run the selected spec once and compare theta exactly
    rerun = estimate(train, selected)
    det_max_abs = float(abs(rerun["theta"] - fits[selected]["theta"]).max())
    g4_pass = bool(det_max_abs == 0.0)

    r = fits[selected]
    gate = {
        "G1_hessian_pd": r["hess_pd"],
        "G1_se_finite": r["se_finite"],
        "G1_no_blowup": not r["blowup"],
        "G1_restarts_converged": r["restarts_converged"],
        "G4_determinism_bitwise": g4_pass,
        "G4_determinism_max_abs_diff": det_max_abs,
    }
    gate_pass = all(v for k, v in gate.items() if isinstance(v, bool))
    print(f"gate: {gate} -> {'PASS' if gate_pass else 'FAIL'}")

    out = {
        "model": "MNL-B",
        "spec": selected,
        "feature_names": r["names"],
        "theta": {n: float(v) for n, v in zip(r["names"], r["theta"])},
        "se": {n: float(v) for n, v in zip(r["names"], r["se"])},
        "z": {n: float(t / s if s > 0 else float("nan")) for n, t, s in
              zip(r["names"], r["theta"], r["se"])},
        "k_params": r["k"],
        "train_n": len(train),
        "val_n": len(val),
        "train_ll": r["train_ll"],
        "val_ll": r["val_ll"],
        "val_ll_by_spec": {s: fits[s]["val_ll"] for s in SPECS},
        "excluded_specs": {s: {"hess_pd": fits[s]["hess_pd"], "se_finite": fits[s]["se_finite"],
                              "blowup": fits[s]["blowup"], "reason": "identifiability check failed"}
                          for s in SPECS if s != selected and not _identifiable(fits[s])},
        "restart_spread_max_abs": r["restart_spread_max_abs"],
        "restart_losses": r["restart_losses"],
        "hess_eig_min": r["hess_eig_min"],
        "hess_cond": r["hess_cond"],
        "estimation": {
            "method": "expected log-likelihood under Teacher mean-probability labels (soft targets)",
            "optimizer": "exact Newton + Armijo backtracking, 10 fixed-seed restarts (torch L-BFGS rejected: strong_wolfe line-search failures)",
            "objective_convex": True,
            "se_method": "inverse Hessian of expected log-likelihood (M-estimator)",
            "departure_scope": "mode choice only; departure shift = 0 by design",
        },
        "dataset": str(DATASET),
        "dataset_sha256": sha256(DATASET),
        "split_manifest": str(SPLIT_MANIFEST),
        "gate": gate,
        "gate_pass": gate_pass,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coef_path = OUT_DIR / "mnl_b_coefs.json"
    coef_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"froze coefficients -> {coef_path}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
