"""S8-aware MATSim adapter (Phase C infrastructure, built for the S8 freeze).

The base :class:`MATSimAdapter` loads S7-W3 checkpoints (6 alternative numeric
fields). The frozen S8 checkpoint is architecture ``student_s8_v1`` with 12
alternative numeric fields, so it needs the S8 feature extractor and model
class. This module provides :class:`S8MATSimAdapter`, which:

- hard-asserts the checkpoint is ``student_s8_v1`` (no silent schema mismatch);
- encodes states with the 12-field S8 extractor;
- builds real-network scenarios where the pt alternative carries REAL
  transit-accessibility attributes computed by the validated PT planner
  (:func:`plan_accessibility`), i.e. the same pipeline the S8 dataset used.

Phase C rules honored here: load-only (never modifies the frozen checkpoint),
no optimizer, no retraining, no schema/normalization changes.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..accessibility.accessibility_dataset import build_real_alternatives
from ..accessibility.accessibility_features import (
    ARCH_VERSION,
    S8FeatureExtractor,
    TravelerStudentS8,
)
from ..accessibility.gtfs_accessibility import SupplyIndex, plan_accessibility
from ..schemas.context import DynamicContext
from ..schemas.state import UniversalTravelerState
from ..student.release_guard import assert_not_frozen_output
from .adapter import MATSimAdapter

# expected alternative encoder input: mode embedding dim 8 + 12 numeric fields
_EXPECTED_ALT_IN = 8 + 12


class S8MATSimAdapter(MATSimAdapter):
    """MATSim adapter for the FROZEN S8 Supply-Aware Traveler Agent v1.0."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        ckpt_path = Path(checkpoint_path)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck.get("config") or {}
        if cfg.get("arch_version") != ARCH_VERSION:
            raise RuntimeError(
                f"S8MATSimAdapter requires arch_version={ARCH_VERSION!r}, "
                f"checkpoint has {cfg.get('arch_version')!r} ({ckpt_path})"
            )
        if cfg.get("n_alt_num") != 12 or ck.get("feature_spec", {}).get("n_alt_num") != 12:
            raise RuntimeError(
                f"S8MATSimAdapter requires n_alt_num=12 (checkpoint has "
                f"config.n_alt_num={cfg.get('n_alt_num')}, "
                f"feature_spec.n_alt_num={ck.get('feature_spec', {}).get('n_alt_num')})"
            )
        if "extractor_state" not in ck:
            raise RuntimeError("S8 checkpoint lacks extractor_state")
        self.extractor = S8FeatureExtractor().from_state_dict(ck["extractor_state"])
        self.s_cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TravelerStudentS8(cfg, ck["feature_spec"]).to(self.device)
        self.model.load_state_dict(ck["model_state"])
        self.model.eval()
        # schema-match hard assertion: encoder input must be 8 + 12
        alt_in = self.model.alt_encoder[0].weight.shape[1]
        if alt_in != _EXPECTED_ALT_IN:
            raise RuntimeError(
                f"schema mismatch: S8 alt_encoder input is {alt_in}, expected {_EXPECTED_ALT_IN}"
            )
        self._source_checkpoint = str(ckpt_path)

    # ------------------------------------------------------------------ supply
    def make_alt_factory(self, supply_paths: dict[str, str | Path]):
        """Return an ``alt_factory`` for ``MATSimAdapter.build_real_scenario``.

        For every (persona, trip, OD) the factory computes the real transit
        accessibility vector with the validated PT planner and builds the
        car/pt/bike/walk alternatives with the S8 attributes attached.
        """
        idx = SupplyIndex(supply_paths)

        def factory(persona, trip, context: DynamicContext, origin: str, dest: str):
            dep_sec = float(trip.desired_departure_min) * 60.0
            acc = plan_accessibility(idx, origin, dest, dep_sec)
            return build_real_alternatives(persona, trip, context, idx, origin, dest, acc)

        return factory

    def build_real_scenario_s8(
        self,
        personas: list,
        trips: list,
        context: DynamicContext,
        output_dir: str | Path,
        supply_paths: dict[str, str | Path],
        **kwargs,
    ) -> dict:
        """Real-network scenario with S8 decisions + real PT accessibility.

        Thin wrapper around ``MATSimAdapter.build_real_scenario`` that injects
        the S8 alternative factory. Phase C output directories must never live
        inside a frozen release.
        """
        assert_not_frozen_output(output_dir)
        return self.build_real_scenario(
            personas,
            trips,
            context,
            output_dir,
            network_path=supply_paths["network"],
            schedule_path=supply_paths["schedule"],
            vehicles_path=supply_paths["vehicles"],
            stops_path=supply_paths["stops"],
            snap_report_path=supply_paths["snapping"],
            trips_by_stop_path=supply_paths["trips_by_stop"],
            activity_nodes_path=supply_paths["activity_nodes"],
            alt_factory=self.make_alt_factory(supply_paths),
            **kwargs,
        )


def smoke_schema_check(adapter: S8MATSimAdapter, state: UniversalTravelerState) -> dict:
    """Assert the S8 encode/model round-trip is schema-consistent (12 fields)."""
    feats = adapter.extractor.encode(state)
    n_alt = len(state.alternatives)
    assert len(feats["alt_num"]) == n_alt, "alt_num rows != alternatives"
    assert all(len(r) == 12 for r in feats["alt_num"]), "alt_num row is not 12 fields"
    from ..student.dataset import collate_batch

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
    with torch.no_grad():
        out = adapter.model(batch)
    probs = out["mode_probabilities"][0].cpu().numpy()
    return {
        "n_alternatives": n_alt,
        "alt_num_width": len(feats["alt_num"][0]),
        "mode_probabilities": {
            alt.mode: round(float(probs[i]), 6)
            for i, alt in enumerate(state.alternatives)
            if alt.available
        },
        "departure_time_shift_min": round(float(out["departure_time_shift_min"][0].item()), 4),
        "schema_match": True,
    }
