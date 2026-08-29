"""MNL-B baseline: classical multinomial logit fit on Teacher supervision.

E1 (TRC_AIT_5_EXPERIMENT_PLAN §E1; design TRC_AIT_5_E1_MNL_BASELINE_DESIGN.md
v0.2, MNL-B only).

Scope: mode choice only; departure shift == 0 (documented boundary). Utility is
linear-in-parameters over alternative attributes (travel_time, monetary_cost,
access_time, transfers) plus alternative-specific constants and pre-registered
persona/trip interactions.
Estimated by maximizing the expected log-likelihood under the frozen Teacher
mean-probability labels (soft targets, K=3/5 aggregation) — the same
supervision the student's distribution loss used.

Pre-registered specs (selected on the frozen VAL split, never on test):
  S1: base, 15 parameters
  S2: S1 + purpose=commute x {car, pt, bike}, 18 parameters
  S3: S1 + schedule_flexibility ordinal x {car, pt}, 17 parameters

G1 identifiability corrections (v0.3, diagnosed from the frozen estimation
data; all are classical MNL absorb-into-ASC conventions):
  1. `weather_exposure` (constant within mode: 0.05/0.4/0.9/1.0) and
     `reliability_delay_min` (car=3.0, all others 0.0) are constant within
     mode, hence perfectly collinear with the mode ASCs -> removed.
     Rain/C3/C4 responses still enter through travel_time (Phase C formulas
     scale/add tt); reliability increments in C3/C4 co-vary exactly with tt.
  2. `car_own_car` = I[car] x car_ownership is IDENTICAL to asc_car in the
     estimation data: unavailable car alternatives are excluded from the
     design and every AVAILABLE car is owned (availability = license AND
     ownership) -> removed, absorbed into asc_car. Ownership heterogeneity
     still enters behavior through car availability.

Estimation: torch L-BFGS (strong Wolfe), 10 fixed-seed restarts on a strictly
convex objective; asymptotic SEs from the inverse Hessian of the expected
log-likelihood (M-estimator). Deterministic given the frozen data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..schemas.state import UniversalTravelerState

LIM_ORDINAL = {"none": 0, "mild": 1, "significant": 2}
FLEX_ORDINAL = {"low": 0, "medium": 1, "high": 2}


# ----------------------------------------------------------------------------
# Pre-registered specifications
# ----------------------------------------------------------------------------
def _feat(state: UniversalTravelerState, mode: str, spec: str) -> tuple[list[str], np.ndarray]:
    """Feature names + vector for one ALTERNATIVE (mode)."""
    persona, trip = state.persona, state.trip
    alt = next(a for a in state.alternatives if a.mode == mode)

    def _flag(b: bool) -> float:
        return 1.0 if b else 0.0

    names: list[str] = []
    vals: list[float] = []

    def add(name: str, v: float):
        names.append(name)
        vals.append(float(v))

    add("asc_car", _flag(mode == "car"))
    add("asc_pt", _flag(mode == "pt"))
    add("asc_bike", _flag(mode == "bike"))
    add("tt", float(alt.travel_time_min))
    add("cost", float(alt.monetary_cost))
    add("cost_low", float(alt.monetary_cost) if persona.income_group == "low" else 0.0)
    add("cost_high", float(alt.monetary_cost) if persona.income_group == "high" else 0.0)
    add("access", float(alt.access_time_min))
    add("transfers", float(alt.transfers))
    add("pass_pt", _flag(mode == "pt" and persona.transit_pass))
    add("habit", _flag(persona.habitual_mode == mode))
    add("lim_wb", float(LIM_ORDINAL.get(persona.mobility_limitation, 0)) if mode in ("walk", "bike") else 0.0)
    add("age65_wb", _flag(mode in ("walk", "bike") and persona.age_group == "65+"))
    add("hard_car", _flag(mode == "car" and trip.time_constraint == "hard"))
    add("hard_pt", _flag(mode == "pt" and trip.time_constraint == "hard"))

    if spec == "S2":
        add("commute_car", _flag(mode == "car" and trip.purpose == "commute"))
        add("commute_pt", _flag(mode == "pt" and trip.purpose == "commute"))
        add("commute_bike", _flag(mode == "bike" and trip.purpose == "commute"))
    elif spec == "S3":
        f = float(FLEX_ORDINAL.get(persona.schedule_flexibility, 0))
        add("flex_car", f if mode == "car" else 0.0)
        add("flex_pt", f if mode == "pt" else 0.0)
    return names, np.asarray(vals, dtype=np.float64)


def build_design(state: UniversalTravelerState, spec: str) -> tuple[list[str], list[str], np.ndarray]:
    """(available modes, feature names, X rows for available modes)."""
    avail = [a.mode for a in state.alternatives if a.available]
    names0: list[str] | None = None
    rows = []
    for m in avail:
        names, v = _feat(state, m, spec)
        if names0 is None:
            names0 = names
        else:
            assert names == names0, "feature name order differs across modes"
        rows.append(v)
    return avail, list(names0), np.vstack(rows)


# ----------------------------------------------------------------------------
# Objective
# ----------------------------------------------------------------------------
def neg_loglik(theta: torch.Tensor, Xs: list[torch.Tensor], Ps: list[torch.Tensor]) -> torch.Tensor:
    """Negative expected log-likelihood under soft teacher labels."""
    total = torch.zeros((), dtype=theta.dtype)
    for X, p in zip(Xs, Ps):
        v = X @ theta
        logz = torch.logsumexp(v, dim=0)
        total = total - torch.sum(p * (v - logz))
    return total


def loglik_value(samples: list, spec: str, theta: np.ndarray) -> float:
    """Expected log-likelihood on a list of AggregatedTeacherTarget samples."""
    total = 0.0
    for s in samples:
        avail, _, X = build_design(s.state, spec)
        tp = s.teacher_aggregate.mode_probabilities
        p = np.asarray([tp.get(m, 0.0) for m in avail], dtype=np.float64)
        p = p / p.sum()
        v = X @ theta
        v = v - v.max()
        total += float(np.sum(p * (v - np.log(np.exp(v).sum()))))
    return total


# ----------------------------------------------------------------------------
# Estimation
# ----------------------------------------------------------------------------
def _prepare(samples: list, spec: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    Xs, Ps = [], []
    for s in samples:
        avail, _, X = build_design(s.state, spec)
        tp = s.teacher_aggregate.mode_probabilities
        p = np.asarray([tp.get(m, 0.0) for m in avail], dtype=np.float64)
        p = p / p.sum()
        Xs.append(torch.tensor(X, dtype=torch.float64))
        Ps.append(torch.tensor(p, dtype=torch.float64))
    return Xs, Ps


def _prepare_np(samples: list, spec: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Design matrices (numpy) + soft teacher targets for available modes."""
    Xs, Ps = [], []
    for s in samples:
        avail, _, X = build_design(s.state, spec)
        tp = s.teacher_aggregate.mode_probabilities
        p = np.asarray([tp.get(m, 0.0) for m in avail], dtype=np.float64)
        p = p / p.sum()
        Xs.append(np.asarray(X, dtype=np.float64))
        Ps.append(p)
    return Xs, Ps


def _loss_np(theta: np.ndarray, Xs: list[np.ndarray], Ps: list[np.ndarray]) -> float:
    total = 0.0
    for X, p in zip(Xs, Ps):
        v = X @ theta
        v = v - v.max()
        e = np.exp(v)
        total += -float(p @ (v - np.log(e.sum())))
    return total


def _fgh_np(theta: np.ndarray, Xs: list[np.ndarray], Ps: list[np.ndarray]):
    """Closed-form loss / gradient / Hessian of the soft-target log-likelihood.

    g = sum_n X_n^T (q_n - p_n);  H = sum_n X_n^T (diag(q_n) - q_n q_n^T) X_n
    with q_n = softmax(X_n theta).
    """
    K = len(theta)
    g = np.zeros(K)
    H = np.zeros((K, K))
    total = 0.0
    for X, p in zip(Xs, Ps):
        v = X @ theta
        v = v - v.max()
        e = np.exp(v)
        q = e / e.sum()
        total += -float(p @ (v - np.log(e.sum())))
        diff = q - p
        g += X.T @ diff
        xq = X.T * q
        H += xq @ X - np.outer(xq.sum(axis=1), xq.sum(axis=1))
    return total, g, H


def estimate(samples: list, spec: str, n_restarts: int = 10, base_seed: int = 0,
             max_iter: int = 200) -> dict:
    """Fit spec on samples (soft labels) with exact-Newton + Armijo backtracking.

    All quantities are closed-form numpy expressions of the soft-target
    expected log-likelihood; the objective is strictly convex after the
    identifiability corrections, so exact Newton converges quadratically and
    the 10 restarts verify the unique optimum (G1/G4).
    torch L-BFGS was rejected during E1 bring-up: its strong_wolfe line search
    repeatedly failed on this objective (stale points with |grad| ~ 1e2).
    """
    Xs, Ps = _prepare_np(samples, spec)
    K = Xs[0].shape[1]

    solutions = []
    for r in range(n_restarts):
        rng = np.random.default_rng(base_seed + r)
        theta = rng.uniform(-0.05, 0.05, size=K)
        loss = float("inf")
        iters = 0
        for it in range(max_iter):
            loss, g, H = _fgh_np(theta, Xs, Ps)
            if not (np.isfinite(loss) and np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
                break
            if float(np.abs(g).max()) < 1e-10:
                iters = it
                break
            try:
                d = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                d = -g / max(float(np.abs(g).max()), 1e-12) * 0.05  # fallback GD
            gd = float(g @ d)
            if not (np.all(np.isfinite(d)) and gd < 0):
                d = -g / max(float(np.abs(g).max()), 1e-12) * 0.05
                gd = float(g @ d)
            alpha = 1.0
            accepted = False
            for _ in range(60):
                cand = theta + alpha * d
                cand_loss = _loss_np(cand, Xs, Ps)
                if np.isfinite(cand_loss) and cand_loss <= loss + 1e-4 * alpha * gd:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
            theta = theta + alpha * d
            iters = it + 1
        loss, g, H = _fgh_np(theta, Xs, Ps)
        solutions.append({"theta": theta, "loss": loss, "grad_max": float(np.abs(g).max()),
                          "iters": iters, "hess": H})

    losses = [s["loss"] for s in solutions]
    best = solutions[int(np.argmin(losses))]
    thetas = np.stack([s["theta"] for s in solutions])
    spread = float(np.abs(thetas - best["theta"]).max())
    converged = spread < 1e-6

    theta_star = best["theta"]
    H = best["hess"]
    eig = np.linalg.eigvalsh(H)
    cond = float(np.linalg.cond(H))
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
        se_finite = bool(np.all(np.isfinite(se)))
    except np.linalg.LinAlgError:
        se = np.full(K, np.nan)
        se_finite = False

    return {
        "spec": spec,
        "k": K,
        "loss": best["loss"],
        "grad_max": best["grad_max"],
        "theta": theta_star,
        "se": se,
        "hess_eig_min": float(eig.min()),
        "hess_eig_max": float(eig.max()),
        "hess_cond": cond,
        "hess_pd": bool(eig.min() > 0.0),
        "se_finite": se_finite,
        "restart_spread_max_abs": spread,
        "restarts_converged": converged,
        "restart_losses": [round(s["loss"], 8) for s in solutions],
        "restart_grad_max": [round(s["grad_max"], 8) for s in solutions],
        "blowup": bool(np.abs(theta_star).max() >= 100.0),
    }


# ----------------------------------------------------------------------------
# Decision maker
# ----------------------------------------------------------------------------
class MNLModel:
    """MNL-B decision maker with frozen coefficients."""

    def __init__(self, spec: str, feature_names: list[str], theta: np.ndarray):
        self.spec = spec
        self.feature_names = list(feature_names)
        self.theta = np.asarray(theta, dtype=np.float64)
        assert len(self.feature_names) == len(self.theta)

    def predict_probs(self, state: UniversalTravelerState) -> dict[str, float]:
        avail, names, X = build_design(state, self.spec)
        assert names == self.feature_names, "feature names do not match frozen spec"
        v = X @ self.theta
        v = v - v.max()
        e = np.exp(v)
        p = e / e.sum()
        return {m: float(p[i]) for i, m in enumerate(avail)}

    def decide(self, state: UniversalTravelerState) -> dict:
        probs = self.predict_probs(state)
        chosen = max(probs, key=probs.get)
        return {
            "mode": chosen,
            "mode_probabilities": probs,
            "departure_time_shift_min": 0.0,
        }

    def to_json(self) -> dict:
        return {
            "model": "MNL-B",
            "spec": self.spec,
            "feature_names": self.feature_names,
            "theta": {n: float(v) for n, v in zip(self.feature_names, self.theta)},
        }

    @classmethod
    def from_json(cls, payload: dict) -> "MNLModel":
        names = payload["feature_names"]
        theta = np.asarray([payload["theta"][n] for n in names], dtype=np.float64)
        return cls(payload["spec"], names, theta)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
