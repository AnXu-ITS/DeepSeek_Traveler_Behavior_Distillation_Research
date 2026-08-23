"""Training dataset: variable-choice-set batching with padding + availability mask."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..dataset.aggregation import AggregatedTeacherTarget
from .features import FeatureExtractor


def _target_vector(sample: AggregatedTeacherTarget) -> tuple[list[float], int, float]:
    """Dense teacher-probability vector aligned with the sample's alternatives.

    Available alternatives carry the aggregated teacher probability for their
    mode; unavailable/padded entries carry 0. The target mode index is the
    argmax position over this dense vector.
    """
    probs = []
    for alt in sample.state.alternatives:
        if alt.available:
            probs.append(float(sample.teacher_aggregate.mode_probabilities.get(alt.mode, 0.0)))
        else:
            probs.append(0.0)
    mode_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    departure = float(sample.teacher_aggregate.departure_time_shift_min)
    return probs, mode_idx, departure


def encode_sample(sample: AggregatedTeacherTarget, extractor: FeatureExtractor) -> dict:
    """Encode one sample into model-ready tensors (features + targets)."""
    feats = extractor.encode(sample.state)
    probs, mode_idx, departure = _target_vector(sample)
    return {
        "global_cat": torch.tensor(feats["global_cat"], dtype=torch.long),
        "global_num": torch.tensor(feats["global_num"], dtype=torch.float32),
        "alt_mode_idx": torch.tensor(feats["alt_mode_idx"], dtype=torch.long),
        "alt_num": torch.tensor(feats["alt_num"], dtype=torch.float32),
        "alt_mask": torch.tensor(feats["alt_available"], dtype=torch.float32),
        "target_probs": torch.tensor(probs, dtype=torch.float32),
        "target_mode_idx": torch.tensor(mode_idx, dtype=torch.long),
        "target_departure": torch.tensor(departure, dtype=torch.float32),
    }


class AggregatedTeacherDataset(Dataset):
    def __init__(self, samples: list[AggregatedTeacherTarget], extractor: FeatureExtractor):
        self.samples = samples
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return encode_sample(self.samples[idx], self.extractor)


class CounterfactualPairDataset(Dataset):
    """(baseline, counterfactual) pairs for behavioral-elasticity training.

    Each item is a counterfactual sample paired with its baseline (resolved via
    ``baseline_sample_id``). Because a baseline and its counterfactuals share
    the same persona+trip, their alternative lists are identical, so the dense
    probability vectors are mode-aligned and the elasticity loss can subtract
    them directly.
    """

    def __init__(self, pairs: list[tuple], extractor: FeatureExtractor):
        self.pairs = pairs
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        base, cf = self.pairs[idx]
        return {
            "base": encode_sample(base, self.extractor),
            "cf": encode_sample(cf, self.extractor),
        }


def collate_pairs(batch: list[dict]) -> dict:
    """Collate a batch of (base, cf) pairs into two separately-padded tensors."""
    return {
        "base": collate_batch([b["base"] for b in batch]),
        "cf": collate_batch([b["cf"] for b in batch]),
    }


def build_baseline_index(samples: list) -> dict:
    """Map sample_id -> baseline sample (perturbation.axis == 'baseline')."""
    return {s.sample_id: s for s in samples if s.perturbation.axis == "baseline"}


def make_counterfactual_pairs(samples: list, baseline_index: dict) -> list[tuple]:
    """Return (baseline, counterfactual) pairs for every CF sample with a baseline."""
    pairs = []
    for s in samples:
        if s.perturbation.axis == "baseline":
            continue
        base = baseline_index.get(s.baseline_sample_id)
        if base is not None:
            pairs.append((base, s))
    return pairs


def make_persona_contrast_pairs(
    samples: list, max_pairs_per_group: int | None = None
) -> list[tuple]:
    """(persona A, persona B) pairs in the SAME travel situation.

    Groups samples by (trip_id, perturbation axis, level); within a group, pairs
    every distinct persona with every other distinct persona (one sample per
    persona per group). Pairing is symmetric only (A,B) once, and only samples
    with different persona ids are paired.

    All alternatives are emitted in the fixed order [car, pt, bike, walk], so
    the dense target vectors of two personas are mode-aligned even when their
    availability differs; the loss uses the union availability mask.
    """
    from collections import defaultdict as _dd

    groups: dict[tuple, list] = _dd(list)
    for s in samples:
        key = (
            s.state.trip.trip_id,
            s.perturbation.axis,
            s.perturbation.level,
        )
        groups[key].append(s)

    pairs = []
    for members in groups.values():
        by_persona: dict[str, list] = _dd(list)
        for s in members:
            by_persona[s.state.persona.persona_id].append(s)
        pids = sorted(by_persona)
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                pairs.append((by_persona[pids[i]][0], by_persona[pids[j]][0]))
                if max_pairs_per_group is not None and len(pairs) >= max_pairs_per_group:
                    return pairs
    return pairs


class PersonaContrastPairDataset(Dataset):
    """Persona-contrast pairs for behavioral-heterogeneity training.

    Each item is a pair of samples from DIFFERENT personas in the SAME
    situation (same trip, same perturbation axis and level).
    """

    def __init__(self, pairs: list[tuple], extractor: FeatureExtractor):
        self.pairs = pairs
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        a, b = self.pairs[idx]
        return {
            "a": encode_sample(a, self.extractor),
            "b": encode_sample(b, self.extractor),
        }


def collate_contrast_pairs(batch: list[dict]) -> dict:
    """Collate a batch of (a, b) persona-contrast pairs into two padded tensors."""
    return {
        "a": collate_batch([b["a"] for b in batch]),
        "b": collate_batch([b["b"] for b in batch]),
    }


def collate_batch(batch: list[dict]) -> dict:
    """Pad the variable-length alternative axis to the batch max."""
    max_alt = max(b["alt_mode_idx"].shape[0] for b in batch)

    def pad_1d(t, pad, dtype):
        out = torch.zeros(max_alt, dtype=dtype)
        out[: t.shape[0]] = t
        return out

    def pad_2d(t):
        n = t.shape[0]
        out = torch.zeros(max_alt, t.shape[1], dtype=t.dtype)
        out[:n] = t
        return out

    out = {
        "global_cat": torch.stack([b["global_cat"] for b in batch]),
        "global_num": torch.stack([b["global_num"] for b in batch]),
        "alt_mode_idx": torch.stack([pad_1d(b["alt_mode_idx"], 0, torch.long) for b in batch]),
        "alt_num": torch.stack([pad_2d(b["alt_num"]) for b in batch]),
        "alt_mask": torch.stack([pad_1d(b["alt_mask"], 0.0, torch.float32) for b in batch]),
    }
    if "target_probs" in batch[0]:
        out["target_probs"] = torch.stack(
            [pad_1d(b["target_probs"], 0.0, torch.float32) for b in batch]
        )
        out["target_mode_idx"] = torch.stack([b["target_mode_idx"] for b in batch])
        out["target_departure"] = torch.stack([b["target_departure"] for b in batch])
    return out
