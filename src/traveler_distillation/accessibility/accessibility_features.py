"""S8 — Student-S8 model and extended feature extractor (Case B).

Case B (S8 instructions §5): the alternative feature vector grows from 6 to 12
numeric fields, so the alt_encoder input dimension changes and S8 is a NEW
architecture version initialized from the frozen S7-W3 weights:

- global encoder / scorer / departure head / embeddings: copied verbatim;
- ``alt_encoder.0.weight`` (32, 14) -> (32, 20): first 14 columns copied from
  S7-W3, the 6 new columns zero-initialized (so S8-at-init reproduces S7-W3
  behavior exactly, given z-scored inputs — an initialization invariant that
  the sanity checks verify);
- the frozen S7-W3 release directory is never modified.

Parameter count: 24,370 (S7-W3) -> 24,562 (S8).
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from ..schemas.alternative import TravelAlternative
from ..schemas.state import UniversalTravelerState
from ..student.features import (
    ALT_NUM,
    CONTEXT_CAT,
    CONTEXT_NUM,
    GLOBAL_CAT,
    GLOBAL_NUM,
    PERSONA_CAT,
    PERSONA_NUM,
    TRIP_CAT,
    TRIP_NUM,
    FeatureExtractor,
    UNK,
)
from ..student.model import TravelerStudent

# S8 added alternative-level numeric fields (city-independent accessibility attributes)
S8_NEW_ALT_NUM = [
    "pt_feasible",
    "egress_time_min",
    "wait_time_min",
    "in_vehicle_time_min",
    "transfer_time_min",
    "coverage_ratio",
]
S8_ALT_NUM = ALT_NUM + S8_NEW_ALT_NUM
ARCH_VERSION = "student_s8_v1"
S8_PARAM_COUNT = 24562


class S8FeatureExtractor(FeatureExtractor):
    """Extended extractor: 12 alternative numeric fields (6 S7 + 6 S8)."""

    ALT_NUM_ORDER = S8_ALT_NUM

    @classmethod
    def from_s7_and_train(cls, s7_extractor_state: dict, train_states: list[UniversalTravelerState]) -> "S8FeatureExtractor":
        """Reuse the frozen S7 vocabularies/stats verbatim; fit ONLY the 6 new
        alternative fields on the S8 train states."""
        ext = cls()
        ext.cat_vocabs = copy.deepcopy(s7_extractor_state["cat_vocabs"])
        ext.mode_vocab = copy.deepcopy(s7_extractor_state["mode_vocab"])
        ext.num_mean = copy.deepcopy(s7_extractor_state["num_mean"])
        ext.num_std = copy.deepcopy(s7_extractor_state["num_std"])

        values: dict[str, list[float]] = {name: [] for name in S8_NEW_ALT_NUM}
        for state in train_states:
            for alt in state.alternatives:
                for name in S8_NEW_ALT_NUM:
                    values[name].append(float(getattr(alt, name)))
        for name in S8_NEW_ALT_NUM:
            vals = values[name]
            mean = sum(vals) / len(vals) if vals else 0.0
            var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
            std = var ** 0.5
            ext.num_mean[name] = mean
            ext.num_std[name] = std if std > 1e-8 else 1.0
        ext.fitted = True
        return ext

    def _alt_num_row(self, alt: TravelAlternative) -> list[float]:
        return [
            (float(getattr(alt, name)) - self.num_mean[name]) / self.num_std[name]
            for name in S8_ALT_NUM
        ]

    def encode(self, state: UniversalTravelerState) -> dict:
        if not self.fitted:
            raise RuntimeError("S8FeatureExtractor must be fit before encode")
        global_cat = []
        for name in GLOBAL_CAT:
            v = str(self._get_cat(state, name))
            global_cat.append(self.cat_vocabs[name].get(v, 0))
        global_num = []
        for name in GLOBAL_NUM:
            x = self._get_num(state, name)
            global_num.append((x - self.num_mean[name]) / self.num_std[name])
        alt_mode_idx = []
        alt_num = []
        alt_available = []
        for alt in state.alternatives:
            alt_mode_idx.append(self.mode_vocab.get(alt.mode, 0))
            alt_num.append(self._alt_num_row(alt))
            alt_available.append(1.0 if alt.available else 0.0)
        return {
            "global_cat": global_cat,
            "global_num": global_num,
            "alt_mode_idx": alt_mode_idx,
            "alt_num": alt_num,
            "alt_available": alt_available,
        }

    @property
    def spec(self) -> dict:
        return {
            "cat_names": list(GLOBAL_CAT),
            "cat_vocab_sizes": {name: len(v) for name, v in self.cat_vocabs.items()},
            "n_global_num": len(GLOBAL_NUM),
            "n_alt_num": len(S8_ALT_NUM),
            "mode_vocab_size": len(self.mode_vocab),
        }


class TravelerStudentS8(TravelerStudent):
    """Student-S8: same architecture family, 12 alternative numeric fields.

    Built by passing an S8 feature spec (n_alt_num = 12) to the S7 model
    constructor; the alt_encoder input becomes 8 + 12 = 20. Everything else is
    structurally identical to S7-W3.
    """

    ARCH_VERSION = ARCH_VERSION

    @classmethod
    def from_s7_weights(cls, s7_checkpoint: str | Path, feature_spec: dict) -> "TravelerStudentS8":
        """Initialize from the frozen S7-W3 checkpoint.

        Copies all shared weights verbatim; extends ``alt_encoder.0.weight``
        with zero columns for the 6 new alternative fields.
        """
        ck = torch.load(Path(s7_checkpoint), map_location="cpu", weights_only=False)
        model = cls(ck["config"], feature_spec)
        s7_sd = ck["model_state"]
        n_old = len(ALT_NUM)
        n_new = len(S8_NEW_ALT_NUM)
        old_alt_in = model.mode_embedding.embedding_dim + n_old
        new_alt_in = model.mode_embedding.embedding_dim + n_old + n_new
        assert s7_sd["alt_encoder.0.weight"].shape[1] == old_alt_in
        assert model.alt_encoder[0].weight.shape[1] == new_alt_in
        sd = {}
        for key, val in s7_sd.items():
            if key == "alt_encoder.0.weight":
                extended = torch.zeros((val.shape[0], new_alt_in), dtype=val.dtype)
                extended[:, :old_alt_in] = val
                sd[key] = extended
            else:
                sd[key] = val.clone()
        model.load_state_dict(sd)
        model.eval()
        return model


def load_s8(extractor_state: dict, model_state: dict, config: dict, feature_spec: dict) -> tuple[S8FeatureExtractor, TravelerStudentS8]:
    """Restore a trained Student-S8 from checkpoint parts."""
    ext = S8FeatureExtractor().load_state_dict(extractor_state)
    model = TravelerStudentS8(config, feature_spec)
    model.load_state_dict(model_state)
    model.eval()
    return ext, model
