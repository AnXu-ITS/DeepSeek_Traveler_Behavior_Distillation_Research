"""Group-aware train/val/test split (no cross-split group leakage)."""
from __future__ import annotations

import random
from collections import defaultdict


def group_aware_split(
    samples: list,
    group_field: str = "counterfactual_group_id",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split samples by group so every group lives in exactly one split.

    Samples with a ``None`` group id are treated as their own singleton group.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    groups: dict[str, list] = defaultdict(list)
    for s in samples:
        gid = getattr(s, group_field, None) or f"__singleton__{s.sample_id}"
        groups[gid].append(s)

    gids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(gids)

    n = len(gids)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    # ensure at least one test group when possible
    n_train = min(n_train, n - 2) if n > 2 else n_train
    n_val = min(n_val, n - n_train - 1) if n - n_train > 1 else n_val

    train_ids = set(gids[:n_train])
    val_ids = set(gids[n_train : n_train + n_val])
    test_ids = set(gids[n_train + n_val :])

    train = [s for gid, ss in groups.items() if gid in train_ids for s in ss]
    val = [s for gid, ss in groups.items() if gid in val_ids for s in ss]
    test = [s for gid, ss in groups.items() if gid in test_ids for s in ss]

    return train, val, test


def group_overlap(*splits: list, group_field: str = "counterfactual_group_id") -> bool:
    """Return True if any group id appears in more than one split."""
    seen: dict[str, int] = {}
    for i, split in enumerate(splits):
        for s in split:
            gid = getattr(s, group_field, None) or f"__singleton__{s.sample_id}"
            if gid in seen and seen[gid] != i:
                return True
            seen[gid] = i
    return False
