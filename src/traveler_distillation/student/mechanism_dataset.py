"""S7 mechanism quadruplet dataset: A/B/C/D causal-audit states as one item.

Each S6 audit group (persona::trip::axis) contributes one quadruplet
{A=baseline, B=natural, C=broken, D=mediator}. All four members share the same
persona+trip, so their alternative lists are identical and the dense
probability vectors are mode-aligned — the mechanism losses can subtract them
directly.

The quadruplet JSONL is produced by ``scripts/build_s7_mechanism_dataset.py``
from ``data/causal_audit/states_with_teacher.jsonl`` (S6 data, reused verbatim:
no new API calls). Rows have the shape::

    {
      "audit_group_id": "P000001::T000001::parking_cost",
      "axis_id": "parking_cost",
      "persona_id": "P000001",
      "trip_id": "T000001",
      "split": "train" | "val" | "test",      # S3-C persona holdout, verbatim
      "members": {
        "baseline": {"state": {...}, "teacher_probs": {...},
                     "teacher_departure": 0.0, "teacher_k": 3},
        "natural":  {...}, "broken": {...}, "mediator": {...}
      }
    }

Split integrity is a hard S7 constraint: the final causal test set (split ==
"test") must never enter training or hyperparameter/model selection. The build
script asserts the train/val/test persona sets are disjoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..schemas.state import UniversalTravelerState
from .dataset import collate_batch
from .features import FeatureExtractor

ROLES = ("baseline", "natural", "broken", "mediator")


@dataclass
class QuadMember:
    """One state of a mechanism quadruplet with its reused S6 teacher target."""

    state: UniversalTravelerState
    teacher_probs: dict[str, float]
    teacher_departure: float
    teacher_k: int


@dataclass
class MechanismQuadruplet:
    audit_group_id: str
    axis_id: str
    persona_id: str
    trip_id: str
    split: str
    members: dict[str, QuadMember]  # keyed by ROLES

    @property
    def A(self) -> QuadMember:
        return self.members["baseline"]

    @property
    def B(self) -> QuadMember:
        return self.members["natural"]

    @property
    def C(self) -> QuadMember:
        return self.members["broken"]

    @property
    def D(self) -> QuadMember:
        return self.members["mediator"]


def load_mechanism_quadruplets(
    path: str | Path,
    split: str | None = None,
    axes: list[str] | None = None,
) -> list[MechanismQuadruplet]:
    """Load quadruplets from the S7 mechanism dataset file.

    ``split`` (train/val/test) and ``axes`` (e.g. ["congestion", "parking_cost"])
    optionally filter the rows; incomplete quadruplets raise loudly.
    """
    quads: list[MechanismQuadruplet] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split is not None and row["split"] != split:
            continue
        if axes is not None and row["axis_id"] not in axes:
            continue
        members = {}
        for role in ROLES:
            if role not in row["members"]:
                raise ValueError(f"quadruplet {row['audit_group_id']} missing member {role!r}")
            m = row["members"][role]
            members[role] = QuadMember(
                state=UniversalTravelerState.model_validate(m["state"]),
                teacher_probs=dict(m["teacher_probs"]),
                teacher_departure=float(m["teacher_departure"]),
                teacher_k=int(m["teacher_k"]),
            )
        quads.append(
            MechanismQuadruplet(
                audit_group_id=row["audit_group_id"],
                axis_id=row["axis_id"],
                persona_id=row["persona_id"],
                trip_id=row["trip_id"],
                split=row["split"],
                members=members,
            )
        )
    return quads


def _target_vector(member: QuadMember) -> tuple[list[float], int, float]:
    """Dense teacher-probability vector aligned with the member's alternatives.

    Available alternatives carry the teacher probability for their mode;
    unavailable entries carry 0. ``mode_idx`` is the argmax position.
    """
    probs = [
        float(member.teacher_probs.get(alt.mode, 0.0)) if alt.available else 0.0
        for alt in member.state.alternatives
    ]
    mode_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return probs, mode_idx, member.teacher_departure


def encode_quad_member(member: QuadMember, extractor: FeatureExtractor) -> dict:
    """Encode one quadruplet member into model-ready tensors (features + targets)."""
    feats = extractor.encode(member.state)
    probs, mode_idx, departure = _target_vector(member)
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


class MechanismQuadrupletDataset(Dataset):
    """One item per A/B/C/D mechanism quadruplet (all four members encoded)."""

    def __init__(self, quadruplets: list[MechanismQuadruplet], extractor: FeatureExtractor):
        self.quadruplets = quadruplets
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.quadruplets)

    def __getitem__(self, idx: int):
        quad = self.quadruplets[idx]
        return {
            "A": encode_quad_member(quad.A, self.extractor),
            "B": encode_quad_member(quad.B, self.extractor),
            "C": encode_quad_member(quad.C, self.extractor),
            "D": encode_quad_member(quad.D, self.extractor),
        }


def collate_quadruplets(batch: list[dict]) -> dict:
    """Collate a batch of quadruplets into four separately-padded sub-batches."""
    return {
        "A": collate_batch([b["A"] for b in batch]),
        "B": collate_batch([b["B"] for b in batch]),
        "C": collate_batch([b["C"] for b in batch]),
        "D": collate_batch([b["D"] for b in batch]),
    }
