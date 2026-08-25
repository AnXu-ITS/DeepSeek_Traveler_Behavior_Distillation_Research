"""S5 multi-axis / joint counterfactual generator.

Derives joint states (TWO or more context axes changed simultaneously) from an
existing baseline state. Joint levels are applied onto the baseline context and
alternatives are regenerated so the state stays internally consistent (the
alternative generator already composes all axes).

A joint sample records its decomposition as ``joint_axes`` so downstream
evaluation can link it back to its single-axis counterparts and compute the
interaction effect  I_ij = P_ij - P_i - P_j + P_0.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schemas.state import UniversalTravelerState
from .alternative_generator import AlternativeGenerator
from .context_perturbation import perturb_context_multi


def resolve_combination(joint_axes: list[dict], combos: list[dict]) -> dict | None:
    """Resolve a joint sample's ``joint_axes`` back to its combination config.

    Matches by the ordered axis list (``joint_axes`` entries carry
    ``{"axis": ..., "level": ...}``). Returns the combination dict (with
    ``combination_id`` / ``seen_in_training``) or ``None``.
    """
    axes = tuple(j["axis"] for j in joint_axes)
    for c in combos:
        if tuple(c["axes"]) == axes:
            return c
    return None


@dataclass
class JointSample:
    combination_id: str
    level_index: int
    axes: list[tuple[str, float | int | bool]]
    state: UniversalTravelerState


class JointContextGenerator:
    """Produce joint (multi-axis) counterfactual states from a baseline."""

    def __init__(self, config: dict | None = None):
        self.alt_gen = AlternativeGenerator(config)
        self.combinations = (config or {}).get("joint_combinations", []) if isinstance(config, dict) else []

    def generate(self, baseline_state: UniversalTravelerState, combination: dict) -> list[JointSample]:
        axes_names: list[str] = combination["axes"]
        results: list[JointSample] = []
        for level_index, level_values in enumerate(combination.get("joint_levels", [])):
            axes = list(zip(axes_names, level_values))
            new_ctx = perturb_context_multi(baseline_state.context, axes)
            cid = combination["combination_id"]
            new_ctx.context_id = (
                f"C_JOINT_{cid}_{level_index}_{baseline_state.persona.persona_id}_"
                f"{baseline_state.trip.trip_id}"
            )
            state = UniversalTravelerState(
                persona=baseline_state.persona,
                trip=baseline_state.trip,
                context=new_ctx,
                alternatives=self.alt_gen.generate(baseline_state.persona, baseline_state.trip, new_ctx),
            )
            results.append(JointSample(combination_id=cid, level_index=level_index, axes=axes, state=state))
        return results
