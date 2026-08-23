"""Audit metrics and deterministic state hashing."""
from __future__ import annotations

import hashlib
import json

from ..schemas.state import UniversalTravelerState


def state_hash(state: UniversalTravelerState) -> str:
    """Deterministic SHA-256 hash of a state (excludes any timestamps/metadata)."""
    canonical = json.dumps(
        state.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """Deterministic SHA-256 hash of the exact behavioral prompt text.

    Covers only the text sent to the model (system + user). Request-level
    metadata such as audit nonces, cache-bypass fields and timestamps are NOT
    included.
    """
    canonical = system_prompt + "\n\n" + user_prompt
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    """Sample standard deviation (ddof=1); 0 for fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def value_range(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def l1(p1: dict, p2: dict) -> float:
    """L1 distance between two probability vectors (missing keys treated as 0)."""
    keys = set(p1) | set(p2)
    return sum(abs(p1.get(k, 0.0) - p2.get(k, 0.0)) for k in keys)


def mean_pairwise_l1(dists: list[dict]) -> float:
    """Mean pairwise L1 distance over a list of probability distributions."""
    if len(dists) < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            total += l1(dists[i], dists[j])
            n += 1
    return total / n


def max_pairwise_l1(dists: list[dict]) -> float:
    """Maximum pairwise L1 distance over a list of probability distributions."""
    if len(dists) < 2:
        return 0.0
    best = 0.0
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            best = max(best, l1(dists[i], dists[j]))
    return best


def aggregate_distribution(dists: list[dict]) -> dict:
    """Element-wise mean probability vector over K distributions."""
    keys = set()
    for d in dists:
        keys |= set(d)
    n = len(dists) or 1
    return {k: sum(d.get(k, 0.0) for d in dists) / n for k in keys}


def mean_single_to_aggregate_l1(dists: list[dict]) -> float:
    """Mean L1 distance of each single distribution to the K-call aggregate."""
    if not dists:
        return 0.0
    agg = aggregate_distribution(dists)
    return sum(l1(d, agg) for d in dists) / len(dists)


def jaccard_similarity(a: list, b: list) -> float:
    """Jaccard similarity between two sets/lists (for reason-code overlap)."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _rank(vals: list[float]) -> list[float]:
    """Average ranks (1-based), ties share the average rank."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def spearman_rank(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (rank-then-Pearson)."""
    if len(x) < 2:
        return 0.0
    return _pearson(_rank(x), _rank(y))


def direction_reversals(values: list[float], eps: float = 1e-9) -> int:
    """Count sign changes in consecutive non-zero differences."""
    signs = []
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        if abs(d) < eps:
            continue
        signs.append(1 if d > 0 else -1)
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


def selected_mode_agreement(modes: list[str]) -> float:
    """Fraction of repeats agreeing on the most common selected mode."""
    if not modes:
        return 0.0
    from collections import Counter

    return max(Counter(modes).values()) / len(modes)
